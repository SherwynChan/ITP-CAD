import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


class CNCAnalyzer:
    """STL analyzer focused specifically on CNC milling manufacturability."""

    def __init__(self, stl_path):
        """Initialize with STL file path."""
        self.stl_path = stl_path
        self.mesh = None
        self.manufacturability_score = 100  # Start at 100 (perfect), subtract penalties
        self.problem_regions = []

    def load_stl(self):
        """Load STL file and return success status."""
        try:
            self.mesh = trimesh.load(self.stl_path)
            print(f"✓ Loaded STL: {len(self.mesh.faces)} faces, {len(self.mesh.vertices)} vertices")
            return True
        except Exception as e:
            print(f"✗ Failed to load STL: {e}")
            return False

    def analyze_cnc_manufacturability(self):
        """
        Analyze manufacturability for CNC milling.
        Returns score 0-100 (100 = perfect for CNC, 0 = impossible)
        """
        if self.mesh is None:
            return 0

        self.manufacturability_score = 100
        self.problem_regions = []

        # 1. Check for undercuts (biggest CNC problem)
        undercuts = self._find_undercuts()
        if len(undercuts) > 0:
            penalty = min(50, len(undercuts) * 0.1)  # Up to 50 point penalty
            self.manufacturability_score -= penalty
            self.problem_regions.append(("Undercuts (Cannot machine)", undercuts))
            print(f"Found {len(undercuts)} undercut faces - Major CNC problem")

        # 2. Check for enclosed internal volumes (impossible to machine)
        internal_volumes = self._find_internal_volumes()
        if internal_volumes > 0:
            self.manufacturability_score -= 40
            print("Found internal enclosed volumes - Cannot machine without assembly")

        # 3. Check for very narrow channels (tool access limited)
        narrow_channels = self._find_narrow_channels()
        if len(narrow_channels) > 0:
            penalty = min(25, len(narrow_channels) * 0.05)
            self.manufacturability_score -= penalty
            self.problem_regions.append(("Narrow Channels", narrow_channels))
            print(f"Found {len(narrow_channels)} faces in narrow channels")

        # 4. Check for very deep pockets (tool length/rigidity issues)
        deep_pockets = self._find_deep_pockets()
        if len(deep_pockets) > 0:
            penalty = min(20, len(deep_pockets) * 0.03)
            self.manufacturability_score -= penalty
            self.problem_regions.append(("Deep Pockets", deep_pockets))
            print(f"Found {len(deep_pockets)} faces in deep pockets")

        # 5. Check for small features (below minimum tool size)
        small_features = self._find_small_features()
        if small_features:
            self.manufacturability_score -= 15
            print("Found very small features - may need micro tools")

        self.manufacturability_score = max(0, self.manufacturability_score)
        return self.manufacturability_score

    def _find_undercuts(self):
        """
        Find true undercuts - features that cannot be reached by straight tools.
        These are re-entrant features where material wraps around.
        """
        face_normals = self.mesh.face_normals
        face_centers = self.mesh.triangles_center

        undercut_faces = []

        for i, (center, normal) in enumerate(zip(face_centers, face_normals)):
            # Check if this face can be accessed by a straight tool from any direction
            is_accessible = False

            # Test tool access from 6 primary directions (±X, ±Y, ±Z)
            tool_directions = [
                np.array([1, 0, 0]),  # +X
                np.array([-1, 0, 0]),  # -X
                np.array([0, 1, 0]),  # +Y
                np.array([0, -1, 0]),  # -Y
                np.array([0, 0, 1]),  # +Z
                np.array([0, 0, -1])  # -Z
            ]

            for tool_dir in tool_directions:
                # Face must be somewhat aligned with tool direction to be machinable
                alignment = np.dot(normal, -tool_dir)  # Negative because tool cuts into surface

                if alignment > 0.1:  # At least some favorable alignment
                    # Check if tool path from this direction is clear
                    if self._is_tool_path_clear(center, tool_dir):
                        is_accessible = True
                        break

            if not is_accessible:
                undercut_faces.append(i)

        return np.array(undercut_faces)

    def _is_tool_path_clear(self, face_center, tool_direction):
        """Check if a cutting tool can reach the face from given direction."""
        try:
            # Cast ray from far away toward the face
            ray_start = face_center - tool_direction * 100  # Start 100 units away

            # Check for intersections between start and face
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=ray_start.reshape(1, -1),
                ray_directions=tool_direction.reshape(1, -1)
            )

            if len(locations) == 0:
                return True  # No obstructions

            # Check if first intersection is at our target face
            distances = np.linalg.norm(locations - ray_start, axis=1)
            target_distance = np.linalg.norm(face_center - ray_start)

            # If first hit is at our target, path is clear
            return np.min(distances) >= target_distance - 0.1

        except:
            return True  # Assume accessible if ray casting fails

    def _find_internal_volumes(self):
        """
        Check for completely enclosed internal volumes.
        These cannot be machined without assembly or complex entry paths.
        """
        if not self.mesh.is_watertight:
            return 0  # Can't have internal volumes if not watertight

        # Simple heuristic: if convex hull volume >> actual volume,
        # there might be internal cavities
        try:
            convex_volume = self.mesh.convex_hull.volume
            actual_volume = self.mesh.volume

            if actual_volume > 0 and convex_volume > 0:
                volume_ratio = actual_volume / convex_volume
                if volume_ratio < 0.6:  # Significant volume difference
                    return 1  # Likely internal volumes
        except:
            pass

        return 0

    def _find_narrow_channels(self):
        """
        Find faces in channels too narrow for standard tooling.
        Minimum practical channel width is ~3x tool diameter.
        """
        face_centers = self.mesh.triangles_center
        face_normals = self.mesh.face_normals
        narrow_faces = []

        # Sample subset of faces for performance
        sample_size = min(500, len(face_centers))
        sample_indices = np.random.choice(len(face_centers), sample_size, replace=False)

        for idx in sample_indices:
            center = face_centers[idx]
            normal = face_normals[idx]

            # Measure channel width by casting rays perpendicular to normal
            width = self._measure_channel_width(center, normal)

            # If channel is very narrow (< 5mm assumed minimum)
            if 0 < width < 5.0:
                narrow_faces.append(idx)

        return np.array(narrow_faces)

    def _measure_channel_width(self, center, normal):
        """Measure the width of a channel at a specific point."""
        try:
            # Create two perpendicular directions to the normal
            if abs(normal[2]) < 0.9:
                perp1 = np.cross(normal, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(normal, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)

            # Measure distance to nearest surface in perpendicular directions
            distances = []
            for direction in [perp1, -perp1]:
                locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=center.reshape(1, -1),
                    ray_directions=direction.reshape(1, -1)
                )
                if len(locations) > 0:
                    dist = np.min(np.linalg.norm(locations - center, axis=1))
                    distances.append(dist)

            return sum(distances) if len(distances) == 2 else 0
        except:
            return 0

    def _find_deep_pockets(self):
        """
        Find faces in very deep pockets where tool length becomes an issue.
        Depth-to-width ratio > 5:1 is typically problematic.
        """
        face_centers = self.mesh.triangles_center
        face_normals = self.mesh.face_normals

        # Find faces that are significantly "inside" the part
        mesh_bounds = self.mesh.bounds
        mesh_center = np.mean(mesh_bounds, axis=0)

        deep_faces = []

        for i, (center, normal) in enumerate(zip(face_centers, face_normals)):
            # Estimate depth by measuring distance to part boundary
            depth = self._estimate_pocket_depth(center, normal)

            if depth > 20:  # Arbitrary threshold for "deep"
                deep_faces.append(i)

        return np.array(deep_faces)

    def _estimate_pocket_depth(self, center, normal):
        """Estimate the depth of a pocket at a specific face."""
        try:
            # Cast ray outward from face
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=center.reshape(1, -1),
                ray_directions=normal.reshape(1, -1)
            )

            if len(locations) > 0:
                depths = np.linalg.norm(locations - center, axis=1)
                return np.min(depths[depths > 0.1])  # Ignore very close hits

            return 0
        except:
            return 0

    def _find_small_features(self):
        """Check for features smaller than typical minimum tool sizes."""
        if hasattr(self.mesh, 'edges_unique_length'):
            edge_lengths = self.mesh.edges_unique_length
            # If many edges are very small (< 0.5mm), might need micro tools
            small_edges = edge_lengths[edge_lengths < 0.5]
            return len(small_edges) > len(edge_lengths) * 0.2
        return False

    def visualize(self, save_path=None):
        """Create visualization showing CNC manufacturability analysis."""
        if self.mesh is None:
            print("No mesh loaded")
            return None

        fig = plt.figure(figsize=(15, 8))

        # Left plot: Original mesh
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_mesh(ax1, self.mesh, title="Original STL Model")

        # Right plot: CNC manufacturability analysis
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_cnc_analysis(ax2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")

        plt.show()
        return fig

    def _plot_mesh(self, ax, mesh, title="Mesh", color='lightblue'):
        """Plot a mesh on given axes."""
        vertices = mesh.vertices
        faces = mesh.faces

        mesh_3d = Poly3DCollection(vertices[faces], alpha=0.7,
                                   facecolor=color, edgecolor='navy', linewidth=0.1)
        ax.add_collection3d(mesh_3d)

        # Set axis limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def _plot_cnc_analysis(self, ax):
        """Plot mesh with CNC problem regions highlighted."""
        vertices = self.mesh.vertices
        faces = self.mesh.faces

        # Default color for all faces (good for CNC)
        face_colors = ['lightgreen'] * len(faces)  # Green = good for CNC

        # Color problem regions
        colors = ['red', 'orange', 'yellow']
        for i, (region_name, face_indices) in enumerate(self.problem_regions):
            color = colors[i % len(colors)]
            for face_idx in face_indices:
                if face_idx < len(face_colors):
                    face_colors[face_idx] = color

        # Create colored mesh
        mesh_3d = Poly3DCollection(vertices[faces], alpha=0.8,
                                   facecolors=face_colors, edgecolor='black', linewidth=0.1)
        ax.add_collection3d(mesh_3d)

        # Set axis limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_title(f'CNC Manufacturability Analysis\nScore: {self.manufacturability_score}/100',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Add legend
        legend_text = ["Green: Good for CNC"]
        if self.problem_regions:
            for i, (region_name, _) in enumerate(self.problem_regions):
                legend_text.append(f"{colors[i % len(colors)]}: {region_name}")

        ax.text2D(0.02, 0.98, '\n'.join(legend_text), transform=ax.transAxes,
                  verticalalignment='top', fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def print_cnc_report(self):
        """Print a CNC-focused manufacturability report."""
        if self.mesh is None:
            print("No analysis performed yet")
            return

        print("\n" + "=" * 50)
        print("CNC MANUFACTURABILITY REPORT")
        print("=" * 50)
        print(f"STL File: {os.path.basename(self.stl_path)}")
        print(f"Faces: {len(self.mesh.faces):,}")
        print(f"Vertices: {len(self.mesh.vertices):,}")
        print(f"Volume: {self.mesh.volume:.2f} cubic units")
        print(f"Watertight: {'Yes' if self.mesh.is_watertight else 'No'}")

        print(f"\nCNC Manufacturability Score: {self.manufacturability_score}/100")

        if self.manufacturability_score >= 80:
            print("CNC Suitability: EXCELLENT - Ideal for CNC machining")
        elif self.manufacturability_score >= 60:
            print("CNC Suitability: GOOD - Suitable with standard tooling")
        elif self.manufacturability_score >= 40:
            print("CNC Suitability: MODERATE - May need special tooling/setup")
        elif self.manufacturability_score >= 20:
            print("CNC Suitability: POOR - Significant CNC challenges")
        else:
            print("CNC Suitability: VERY POOR - Not recommended for CNC")

        print(f"\nCNC Problem Areas Found: {len(self.problem_regions)}")
        for region_name, faces in self.problem_regions:
            severity = "CRITICAL" if "Undercuts" in region_name else "MODERATE"
            print(f"  - {region_name}: {len(faces)} faces ({severity})")

        print("\nCNC Recommendations:")
        if self.manufacturability_score < 40:
            print("  - Consider design changes to eliminate undercuts")
            print("  - Evaluate if part can be made in multiple pieces")
            print("  - Consider alternative manufacturing (3D printing, casting)")
        elif self.manufacturability_score < 70:
            print("  - Plan for specialized tooling")
            print("  - Consider 5-axis CNC for better access")
            print("  - Verify tool length requirements")
        else:
            print("  - Standard 3-axis CNC should work well")
            print("  - Good candidate for CNC manufacturing")


def analyze_cnc_suitability(stl_path, save_image=None):
    """
    Main function to analyze STL file for CNC manufacturing suitability.

    Args:
        stl_path: Path to STL file
        save_image: Optional path to save visualization image

    Returns:
        CNCAnalyzer object with results
    """
    analyzer = CNCAnalyzer(stl_path)

    if not analyzer.load_stl():
        return None

    score = analyzer.analyze_cnc_manufacturability()
    analyzer.print_cnc_report()
    analyzer.visualize(save_path=save_image)

    return analyzer


if __name__ == "__main__":
    # Replace with your STL file path
    stl_file = r"C:\Users\junhongs\Desktop\itp\ITP-CAD\test_file\undercut_bottle.stl"

    if os.path.exists(stl_file):
        result = analyze_cnc_suitability(stl_file, save_image="cnc_analysis.png")

        # Quick decision making
        if result and result.manufacturability_score >= 60:
            print(f"\n ACCEPT: This part is suitable for CNC (Score: {result.manufacturability_score})")
        else:
            print(f"\n REJECT: This part is not suitable for CNC (Score: {result.manufacturability_score})")
    else:
        print(f"STL file not found: {stl_file}")
        print("Please update the 'stl_file' variable with the correct path")
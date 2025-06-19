import numpy as np
import trimesh
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.spatial.distance import cdist
from scipy.ndimage import label, binary_erosion
from sklearn.cluster import DBSCAN


class ManufacturingProcess(Enum):
    CNC_3_AXIS = "3-axis CNC"
    CNC_5_AXIS = "5-axis CNC"
    MILLING = "Milling"
    TURNING = "Turning"


@dataclass
class ManufacturabilityResult:
    is_manufacturable: bool
    issues: list[str]
    difficulty_score: float  # 0-1 scale
    recommended_processes: list[ManufacturingProcess]
    minimum_tool_diameter: float
    deep_pocket_regions: list[np.ndarray]  # Regions requiring long tools
    undercut_regions: list[np.ndarray]  # Regions not accessible by straight tools
    steep_angles: list[np.ndarray]  # Regions with steep walls


@dataclass
class DeepPocket:
    """Represents a deep pocket region with its characteristics"""
    faces: np.ndarray  # Face indices belonging to this pocket
    depth: float
    width: float
    aspect_ratio: float  # depth/width
    center: np.ndarray
    accessibility_vector: np.ndarray  # Best approach direction


class ManufacturabilityAnalyzer:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.min_tool_diameter = 2.0  # mm
        self.max_depth_to_diameter_ratio = 10  # Standard machining constraint
        self.max_wall_angle = np.radians(85)  # Maximum manageable wall angle

        # Deep pocket analysis parameters
        self.min_pocket_depth = 5.0  # mm - minimum depth to consider a pocket
        self.voxel_resolution = 0.5  # mm - resolution for voxel analysis
        self.pocket_clustering_eps = 3.0  # mm - DBSCAN clustering parameter

    def analyze_manufacturability(self) -> ManufacturabilityResult:
        """
        Perform comprehensive manufacturability analysis for subtractive manufacturing.
        """
        issues = []

        # Initialize analysis results
        deep_pockets = self._identify_deep_pockets()
        undercuts = self._identify_undercuts()
        steep_walls = self._identify_steep_walls()
        min_tool_dia = self._calculate_minimum_tool_diameter()

        # Determine if the part is manufacturable
        is_manufacturable = True
        if len(undercuts) > 0:
            is_manufacturable = False
            issues.append("Part contains undercuts not accessible by straight tools")

        if min_tool_dia < self.min_tool_diameter:
            is_manufacturable = False
            issues.append(
                f"Required tool diameter ({min_tool_dia:.2f}mm) is below minimum ({self.min_tool_diameter}mm)")

        # Check deep pocket manufacturability
        problematic_pockets = [p for p in deep_pockets if p.aspect_ratio > self.max_depth_to_diameter_ratio]
        if problematic_pockets:
            is_manufacturable = False
            issues.append(f"Found {len(problematic_pockets)} deep pockets exceeding aspect ratio limits")

        # Calculate difficulty score
        difficulty_score = self._calculate_difficulty_score(
            deep_pockets, undercuts, steep_walls, min_tool_dia
        )

        # Recommend manufacturing processes
        recommended_processes = self._recommend_processes(
            deep_pockets, undercuts, steep_walls
        )

        # Convert DeepPocket objects to face arrays for backward compatibility
        deep_pocket_faces = [pocket.faces for pocket in deep_pockets]

        return ManufacturabilityResult(
            is_manufacturable=is_manufacturable,
            issues=issues,
            difficulty_score=difficulty_score,
            recommended_processes=recommended_processes,
            minimum_tool_diameter=min_tool_dia,
            deep_pocket_regions=deep_pocket_faces,
            undercut_regions=undercuts,
            steep_angles=steep_walls
        )

    def _identify_deep_pockets(self) -> list[DeepPocket]:
        """
        Identify regions that might require long tools (deep pockets).

        This implementation uses a multi-step approach:
        1. Validate mesh is suitable for pocket analysis
        2. Use surface-based analysis instead of voxelization for simple geometries
        3. Identify actual concave regions vs solid blocks
        4. Calculate proper depth and accessibility
        """
        try:
            # Step 1: Quick validation - is this even a candidate for deep pockets?
            if self._is_simple_convex_geometry():
                logging.info("Mesh appears to be simple convex geometry - no deep pockets expected")
                return []

            # Step 2: Use surface-based analysis for better accuracy
            potential_pockets = self._find_surface_based_pockets()

            if len(potential_pockets) == 0:
                logging.info("No surface-based pockets found")
                return []

            # Step 3: Validate and analyze each potential pocket
            deep_pockets = []

            for pocket_data in potential_pockets:
                pocket = self._analyze_pocket_region(pocket_data)
                if pocket and pocket.depth >= self.min_pocket_depth:
                    deep_pockets.append(pocket)

            # Step 4: Merge nearby pockets (optional clustering)
            if len(deep_pockets) > 1:
                deep_pockets = self._cluster_nearby_pockets(deep_pockets)

            logging.info(f"Identified {len(deep_pockets)} deep pocket regions")
            return deep_pockets

        except Exception as e:
            logging.error(f"Error in deep pocket identification: {e}")
            return []

    def _is_simple_convex_geometry(self) -> bool:
        """
        Check if the mesh is a simple convex shape that shouldn't have deep pockets.
        """
        try:
            # Check if mesh is convex or nearly convex
            if self.mesh.is_convex:
                return True

            # Check volume ratio - convex shapes have high volume ratio
            convex_volume = self.mesh.convex_hull.volume
            actual_volume = self.mesh.volume

            if actual_volume <= 0 or convex_volume <= 0:
                return False

            volume_ratio = actual_volume / convex_volume

            # If volume ratio is very high (>0.9), it's likely a simple solid
            if volume_ratio > 0.9:
                logging.info(f"High volume ratio ({volume_ratio:.3f}) suggests simple solid geometry")
                return True

            # Check surface area ratio
            convex_surface_area = self.mesh.convex_hull.area
            actual_surface_area = self.mesh.area

            if actual_surface_area <= 0 or convex_surface_area <= 0:
                return False

            surface_ratio = convex_surface_area / actual_surface_area

            # If surface areas are similar, it's likely convex
            if surface_ratio > 0.8:
                logging.info(f"High surface ratio ({surface_ratio:.3f}) suggests convex geometry")
                return True

            return False

        except Exception as e:
            logging.warning(f"Error checking geometry simplicity: {e}")
            return False

    def _find_surface_based_pockets(self) -> list[dict]:
        """
        Find potential pockets using surface analysis instead of voxelization.
        """
        potential_pockets = []

        try:
            # Analyze face normals to find inward-facing regions
            face_normals = self.mesh.face_normals
            face_centers = self.mesh.triangles_center

            # Find faces that might be part of internal pockets
            # Look for faces whose normals point "inward" relative to mesh center
            mesh_center = np.mean(self.mesh.vertices, axis=0)

            internal_faces = []

            for i, (face_center, face_normal) in enumerate(zip(face_centers, face_normals)):
                # Vector from mesh center to face center
                to_face = face_center - mesh_center
                to_face_normalized = to_face / (np.linalg.norm(to_face) + 1e-8)

                # If face normal points toward mesh center, it might be internal
                dot_product = np.dot(face_normal, -to_face_normalized)

                # Also check if face is significantly recessed
                is_recessed = self._is_face_recessed(i, face_center, face_normal)

                if dot_product > 0.3 or is_recessed:  # Threshold for "inward facing"
                    internal_faces.append(i)

            if len(internal_faces) == 0:
                return []

            # Cluster internal faces into pocket regions
            if len(internal_faces) > 0:
                pocket_clusters = self._cluster_internal_faces(internal_faces)

                for cluster in pocket_clusters:
                    if len(cluster) >= 3:  # Minimum faces for a meaningful pocket
                        potential_pockets.append({
                            'face_indices': cluster,
                            'centers': face_centers[cluster],
                            'normals': face_normals[cluster]
                        })

        except Exception as e:
            logging.warning(f"Surface-based pocket detection failed: {e}")

        return potential_pockets

    def _is_face_recessed(self, face_idx: int, face_center: np.ndarray, face_normal: np.ndarray) -> bool:
        """
        Check if a face is significantly recessed from the mesh surface.
        """
        try:
            # Cast a ray outward from the face in its normal direction
            ray_origins = face_center.reshape(1, -1)
            ray_directions = face_normal.reshape(1, -1)

            # Check if ray hits another part of the mesh (indicating recess)
            locations, ray_indices, face_indices = self.mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions
            )

            if len(locations) > 0:
                # If we hit another surface close by, this face is recessed
                distances = np.linalg.norm(locations - face_center, axis=1)
                min_distance = np.min(distances[distances > 0.1])  # Ignore very close hits

                # If another surface is within reasonable distance, face is recessed
                return min_distance < 10.0  # 10mm threshold

            return False

        except Exception as e:
            return False

    def _cluster_internal_faces(self, face_indices: list) -> list[np.ndarray]:
        """
        Cluster internal faces into coherent pocket regions.
        """
        if len(face_indices) < 2:
            return [np.array(face_indices)]

        face_centers = self.mesh.triangles_center[face_indices]

        # Use DBSCAN to cluster nearby faces
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=self.voxel_resolution * 5, min_samples=2)
        cluster_labels = clustering.fit_predict(face_centers)

        clusters = []
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            if label == -1:  # Noise
                continue
            cluster_mask = cluster_labels == label
            cluster_faces = np.array(face_indices)[cluster_mask]
            clusters.append(cluster_faces)

        return clusters

    def _analyze_pocket_region(self, pocket_data: dict) -> DeepPocket:
        """
        Analyze a potential pocket region to determine if it's a real deep pocket.
        """
        face_indices = pocket_data['face_indices']
        face_centers = pocket_data['centers']
        face_normals = pocket_data['normals']

        if len(face_indices) < 3:
            return None

        # Calculate pocket dimensions
        bbox_min = np.min(face_centers, axis=0)
        bbox_max = np.max(face_centers, axis=0)
        bbox_size = bbox_max - bbox_min

        # Estimate depth and width more carefully
        # Find the primary direction of the pocket
        primary_direction = self._find_pocket_primary_direction(face_centers, face_normals)

        # Calculate depth in primary direction
        depth = self._calculate_pocket_depth(face_centers, primary_direction)

        # Calculate width perpendicular to primary direction
        width = self._calculate_pocket_width(face_centers, primary_direction)

        # Validate this is actually a pocket (not just surface variation)
        if depth < self.min_pocket_depth or width < 1.0:
            return None

        # Calculate aspect ratio
        aspect_ratio = depth / max(width, 0.1)

        # Find center and accessibility
        center = np.mean(face_centers, axis=0)
        accessibility_vector = -primary_direction  # Opposite to pocket direction

        return DeepPocket(
            faces=face_indices,
            depth=depth,
            width=width,
            aspect_ratio=aspect_ratio,
            center=center,
            accessibility_vector=accessibility_vector
        )

    def _find_pocket_primary_direction(self, face_centers: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
        """
        Find the primary direction of a pocket region.
        """
        # Use the mean normal direction as primary direction
        mean_normal = np.mean(face_normals, axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
        return mean_normal

    def _calculate_pocket_depth(self, face_centers: np.ndarray, direction: np.ndarray) -> float:
        """
        Calculate the depth of a pocket in the given direction.
        """
        # Project all face centers onto the direction vector
        projections = np.dot(face_centers, direction)

        # Depth is the range of projections
        depth = np.max(projections) - np.min(projections)
        return max(depth, 0.0)

    def _calculate_pocket_width(self, face_centers: np.ndarray, direction: np.ndarray) -> float:
        """
        Calculate the width of a pocket perpendicular to the primary direction.
        """
        # Create perpendicular directions
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction, np.array([1, 0, 0]))

        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-8)

        # Calculate ranges in perpendicular directions
        proj1 = np.dot(face_centers, perp1)
        proj2 = np.dot(face_centers, perp2)

        width1 = np.max(proj1) - np.min(proj1)
        width2 = np.max(proj2) - np.min(proj2)

        # Return the smaller width (more restrictive dimension)
        return min(width1, width2)

    def _find_internal_cavities(self, voxel_grid) -> list[np.ndarray]:
        """
        Find internal cavities using morphological operations and ray casting.
        """
        # Get the voxel matrix
        matrix = voxel_grid.matrix

        # Find internal empty spaces by eroding the solid regions
        # This helps identify pockets and recesses
        eroded = binary_erosion(matrix, iterations=2)
        internal_spaces = matrix & ~eroded

        # Label connected components of internal spaces
        labeled_spaces, num_features = label(internal_spaces)

        cavities = []
        for i in range(1, num_features + 1):
            cavity_mask = labeled_spaces == i
            cavity_coords = np.argwhere(cavity_mask)

            # Convert voxel coordinates to world coordinates
            world_coords = voxel_grid.indices_to_points(cavity_coords)
            cavities.append(world_coords)

        return cavities

    def _analyze_cavity_region(self, cavity_coords: np.ndarray, voxel_grid) -> DeepPocket:
        """
        Analyze a cavity region to determine its depth, width, and accessibility.
        """
        if len(cavity_coords) < 5:  # Skip very small cavities
            return None

        # Calculate bounding box to estimate dimensions
        bbox_min = np.min(cavity_coords, axis=0)
        bbox_max = np.max(cavity_coords, axis=0)
        bbox_size = bbox_max - bbox_min

        # Estimate depth and width
        # Assume the largest dimension is length, second largest is width, smallest is depth
        sorted_dims = np.sort(bbox_size)
        depth = sorted_dims[2]  # Largest dimension (assuming vertical orientation)
        width = sorted_dims[1]  # Second largest dimension

        # Alternative depth calculation: ray casting from top
        depth_raycast = self._calculate_depth_by_raycasting(cavity_coords)
        if depth_raycast > 0:
            depth = max(depth, depth_raycast)

        # Calculate aspect ratio
        aspect_ratio = depth / max(width, 0.1)  # Avoid division by zero

        # Find center of the cavity
        center = np.mean(cavity_coords, axis=0)

        # Determine best accessibility direction (typically from above)
        accessibility_vector = self._find_best_access_direction(cavity_coords)

        # Map back to mesh faces
        faces_in_pocket = self._find_faces_in_region(cavity_coords)

        return DeepPocket(
            faces=faces_in_pocket,
            depth=depth,
            width=width,
            aspect_ratio=aspect_ratio,
            center=center,
            accessibility_vector=accessibility_vector
        )

    def _calculate_depth_by_raycasting(self, cavity_coords: np.ndarray) -> float:
        """
        Calculate cavity depth using ray casting from the top surface.
        """
        try:
            # Find the topmost points of the cavity
            max_z = np.max(cavity_coords[:, 2])
            top_points = cavity_coords[cavity_coords[:, 2] > max_z - 1.0]  # Within 1mm of top

            if len(top_points) == 0:
                return 0.0

            # Cast rays downward from top points
            depths = []
            ray_direction = np.array([0, 0, -1])  # Downward

            for point in top_points:
                # Cast ray and find intersection with mesh
                ray_origins = point.reshape(1, -1)
                ray_directions = ray_direction.reshape(1, -1)

                locations, ray_indices, face_indices = self.mesh.ray.intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions
                )

                if len(locations) >= 2:  # Entry and exit points
                    # Calculate depth as distance between first and last intersection
                    z_positions = locations[:, 2]
                    depth = np.max(z_positions) - np.min(z_positions)
                    depths.append(depth)

            return np.mean(depths) if depths else 0.0

        except Exception as e:
            logging.warning(f"Ray casting depth calculation failed: {e}")
            return 0.0

    def _find_best_access_direction(self, cavity_coords: np.ndarray) -> np.ndarray:
        """
        Find the best tool access direction for machining this cavity.
        """
        # For now, assume vertical access is best (can be improved with normal analysis)
        return np.array([0, 0, 1])  # Upward Z direction

    def _find_faces_in_region(self, region_coords: np.ndarray) -> np.ndarray:
        """
        Find mesh faces that belong to a specific region defined by coordinates.
        """
        if len(region_coords) == 0:
            return np.array([])

        # Find face centers
        face_centers = self.mesh.triangles_center

        # Find faces whose centers are close to the region
        distances = cdist(face_centers, region_coords)
        min_distances = np.min(distances, axis=1)

        # Threshold for considering a face part of the region
        threshold = self.voxel_resolution * 2
        nearby_faces = np.where(min_distances < threshold)[0]

        return nearby_faces

    def _cluster_nearby_pockets(self, pockets: list[DeepPocket]) -> list[DeepPocket]:
        """
        Cluster nearby pocket regions that might be part of the same feature.
        """
        if len(pockets) <= 1:
            return pockets

        # Extract centers for clustering
        centers = np.array([pocket.center for pocket in pockets])

        # Use DBSCAN to cluster nearby pockets
        clustering = DBSCAN(eps=self.pocket_clustering_eps, min_samples=1)
        cluster_labels = clustering.fit_predict(centers)

        # Merge pockets in the same cluster
        merged_pockets = []
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            cluster_pockets = [pockets[i] for i in range(len(pockets)) if cluster_labels[i] == label]

            if len(cluster_pockets) == 1:
                merged_pockets.append(cluster_pockets[0])
            else:
                # Merge multiple pockets
                merged_pocket = self._merge_pockets(cluster_pockets)
                merged_pockets.append(merged_pocket)

        return merged_pockets

    def _merge_pockets(self, pockets: list[DeepPocket]) -> DeepPocket:
        """
        Merge multiple pocket regions into a single pocket.
        """
        # Combine all faces
        all_faces = np.concatenate([pocket.faces for pocket in pockets])

        # Calculate merged properties
        max_depth = max(pocket.depth for pocket in pockets)
        max_width = max(pocket.width for pocket in pockets)
        merged_center = np.mean([pocket.center for pocket in pockets], axis=0)

        return DeepPocket(
            faces=all_faces,
            depth=max_depth,
            width=max_width,
            aspect_ratio=max_depth / max_width,
            center=merged_center,
            accessibility_vector=np.array([0, 0, 1])  # Default upward
        )

    def _identify_undercuts(self) -> list[np.ndarray]:
        """
        Identify regions that cannot be accessed by straight tools.

        This implementation uses multiple approaches:
        1. Draft angle analysis - faces with negative draft angles
        2. Tool accessibility ray casting from multiple directions
        3. Overhang detection - regions that extend beyond supporting geometry
        4. Re-entrant features - concave regions that trap tools
        """
        try:
            undercut_faces = []

            # Method 1: Draft angle analysis
            negative_draft_faces = self._find_negative_draft_faces()
            if len(negative_draft_faces) > 0:
                undercut_faces.extend(negative_draft_faces)

            # Method 2: Tool accessibility analysis
            inaccessible_faces = self._find_tool_inaccessible_faces()
            if len(inaccessible_faces) > 0:
                undercut_faces.extend(inaccessible_faces)

            # Method 3: Overhang detection
            overhang_faces = self._find_overhang_faces()
            if len(overhang_faces) > 0:
                undercut_faces.extend(overhang_faces)

            # Method 4: Re-entrant feature detection
            reentrant_faces = self._find_reentrant_features()
            if len(reentrant_faces) > 0:
                undercut_faces.extend(reentrant_faces)

            # Remove duplicates and cluster nearby undercut regions
            if undercut_faces:
                unique_faces = np.unique(np.concatenate(undercut_faces))
                clustered_undercuts = self._cluster_undercut_faces(unique_faces)

                logging.info(
                    f"Identified {len(clustered_undercuts)} undercut regions with {len(unique_faces)} total faces")
                return clustered_undercuts

            return []

        except Exception as e:
            logging.error(f"Error in undercut identification: {e}")
            return []

    def _find_negative_draft_faces(self) -> list[np.ndarray]:
        """
        Find faces with negative draft angles (faces that slope inward/backward).
        """
        face_normals = self.mesh.face_normals

        # Define machining directions (typically Z-up for 3-axis, but check all principal axes)
        machining_directions = [
            np.array([0, 0, 1]),  # Z-up (most common)
            np.array([0, 0, -1]),  # Z-down
            np.array([1, 0, 0]),  # X-direction
            np.array([-1, 0, 0]),  # -X direction
            np.array([0, 1, 0]),  # Y-direction
            np.array([0, -1, 0])  # -Y direction
        ]

        undercut_regions = []

        for direction in machining_directions:
            # Calculate dot product between face normals and machining direction
            dot_products = np.dot(face_normals, direction)

            # Faces with negative dot product are facing away from tool approach
            # and may create undercuts (depending on angle threshold)
            negative_draft_threshold = np.cos(np.radians(95))  # 5 degrees past vertical
            undercut_mask = dot_products < negative_draft_threshold

            if np.any(undercut_mask):
                undercut_face_indices = np.where(undercut_mask)[0]
                undercut_regions.append(undercut_face_indices)

        return undercut_regions

    def _find_tool_inaccessible_faces(self) -> list[np.ndarray]:
        """
        Use ray casting to determine which faces cannot be reached by a straight tool.
        """
        face_centers = self.mesh.triangles_center
        face_normals = self.mesh.face_normals

        # Tool approach directions (expand based on machine capabilities)
        tool_directions = [
            np.array([0, 0, 1]),  # From above
            np.array([0, 0, -1]),  # From below
            np.array([1, 0, 0]),  # From +X
            np.array([-1, 0, 0]),  # From -X
            np.array([0, 1, 0]),  # From +Y
            np.array([0, -1, 0])  # From -Y
        ]

        inaccessible_faces = []
        tool_radius = self.min_tool_diameter / 2.0

        for face_idx in range(len(face_centers)):
            face_center = face_centers[face_idx]
            face_normal = face_normals[face_idx]

            is_accessible = False

            for tool_dir in tool_directions:
                # Skip directions that are nearly parallel to face (can't machine effectively)
                alignment = abs(np.dot(face_normal, tool_dir))
                if alignment < 0.3:  # Less than ~70 degrees from parallel
                    continue

                # Cast ray from far away in tool direction toward face
                ray_start = face_center - tool_dir * 100  # Start 100mm away
                ray_direction = tool_dir

                # Check if tool path is clear
                if self._is_tool_path_clear(ray_start, ray_direction, face_center, tool_radius):
                    is_accessible = True
                    break

            if not is_accessible:
                inaccessible_faces.append(face_idx)

        return [np.array(inaccessible_faces)] if inaccessible_faces else []

    def _is_tool_path_clear(self, start_point: np.ndarray, direction: np.ndarray,
                            target_point: np.ndarray, tool_radius: float) -> bool:
        """
        Check if a cylindrical tool can reach the target point without collision.
        """
        try:
            # Cast multiple rays around the tool circumference to simulate tool volume
            num_rays = 8  # Number of rays around tool circumference
            angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

            # Create perpendicular vectors to tool direction for ray offsets
            if abs(direction[2]) < 0.9:  # Not purely vertical
                perp1 = np.cross(direction, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(direction, np.array([1, 0, 0]))

            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)

            # Check center ray and circumference rays
            all_rays_clear = True

            for angle in angles:
                # Calculate ray offset from tool center
                offset = tool_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                ray_origin = start_point + offset

                # Cast ray and check for intersections before target
                locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=ray_origin.reshape(1, -1),
                    ray_directions=direction.reshape(1, -1)
                )

                if len(locations) > 0:
                    # Check if any intersection occurs before reaching target
                    distances_to_intersections = np.linalg.norm(locations - ray_origin, axis=1)
                    distance_to_target = np.linalg.norm(target_point - ray_origin)

                    # If any intersection is closer than target, path is blocked
                    if np.any(distances_to_intersections < distance_to_target - 0.1):  # 0.1mm tolerance
                        all_rays_clear = False
                        break

            return all_rays_clear

        except Exception as e:
            logging.warning(f"Tool path clearance check failed: {e}")
            return False

    def _find_overhang_faces(self) -> list[np.ndarray]:
        """
        Detect overhanging features that create undercuts.
        """
        face_centers = self.mesh.triangles_center
        face_normals = self.mesh.face_normals

        # Find faces that are oriented more horizontally and face downward
        # These are potential overhangs
        z_component = face_normals[:, 2]  # Z-component of normal

        # Faces with normal pointing significantly downward
        overhang_threshold = -0.5  # Normal Z-component threshold
        overhang_mask = z_component < overhang_threshold

        overhang_candidates = np.where(overhang_mask)[0]

        if len(overhang_candidates) == 0:
            return []

        # Further filter by checking if there's supporting geometry below
        confirmed_overhangs = []

        for face_idx in overhang_candidates:
            face_center = face_centers[face_idx]

            # Cast ray downward to check for support
            ray_origin = face_center
            ray_direction = np.array([0, 0, -1])  # Downward

            # Check if there's geometry directly below within reasonable distance
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=ray_origin.reshape(1, -1),
                ray_directions=ray_direction.reshape(1, -1)
            )

            # If no intersection below, or intersection is very far, it's an overhang
            support_distance_threshold = 10.0  # mm

            if len(locations) == 0:
                confirmed_overhangs.append(face_idx)
            else:
                distances_below = locations[:, 2] - face_center[2]
                closest_support = np.min(distances_below[distances_below < 0])  # Below current face

                if abs(closest_support) > support_distance_threshold:
                    confirmed_overhangs.append(face_idx)

        return [np.array(confirmed_overhangs)] if confirmed_overhangs else []

    def _find_reentrant_features(self) -> list[np.ndarray]:
        """
        Detect re-entrant features (concave regions that can trap cutting tools).
        """
        # Use mesh curvature analysis to find concave regions
        try:
            # Calculate vertex curvatures (approximate using face adjacency)
            vertex_curvatures = self._calculate_vertex_curvatures()

            # Find highly concave vertices (negative curvature)
            concave_threshold = -0.1  # Adjust based on mesh scale
            concave_vertices = np.where(vertex_curvatures < concave_threshold)[0]

            if len(concave_vertices) == 0:
                return []

            # Find faces connected to concave vertices
            reentrant_faces = []
            for vertex_idx in concave_vertices:
                # Find faces that include this vertex
                faces_with_vertex = np.where(np.any(self.mesh.faces == vertex_idx, axis=1))[0]
                reentrant_faces.extend(faces_with_vertex)

            # Remove duplicates
            unique_reentrant_faces = np.unique(reentrant_faces)

            # Filter faces that are actually problematic for machining
            # (check if they form internal corners or pockets)
            filtered_faces = self._filter_problematic_reentrant_faces(unique_reentrant_faces)

            return [filtered_faces] if len(filtered_faces) > 0 else []

        except Exception as e:
            logging.warning(f"Re-entrant feature detection failed: {e}")
            return []

    def _calculate_vertex_curvatures(self) -> np.ndarray:
        """
        Calculate approximate curvature at each vertex using adjacent face normals.
        """
        vertex_curvatures = np.zeros(len(self.mesh.vertices))

        for vertex_idx in range(len(self.mesh.vertices)):
            # Find faces adjacent to this vertex
            adjacent_faces = np.where(np.any(self.mesh.faces == vertex_idx, axis=1))[0]

            if len(adjacent_faces) < 2:
                continue

            # Calculate mean curvature based on adjacent face normals
            face_normals = self.mesh.face_normals[adjacent_faces]

            # Simple curvature approximation: variance in normal directions
            if len(face_normals) > 1:
                # Calculate how much normals deviate from mean
                mean_normal = np.mean(face_normals, axis=0)
                mean_normal = mean_normal / np.linalg.norm(mean_normal)

                # Curvature approximation based on normal deviation
                deviations = np.array([np.dot(normal, mean_normal) for normal in face_normals])
                curvature = np.std(deviations)  # High std = high curvature

                # Make negative for concave regions
                if np.mean(deviations) < 0.8:  # Normals point in different directions
                    curvature = -curvature

                vertex_curvatures[vertex_idx] = curvature

        return vertex_curvatures

    def _filter_problematic_reentrant_faces(self, face_indices: np.ndarray) -> np.ndarray:
        """
        Filter re-entrant faces to keep only those that actually cause machining problems.
        """
        problematic_faces = []

        for face_idx in face_indices:
            face_normal = self.mesh.face_normals[face_idx]
            face_center = self.mesh.triangles_center[face_idx]

            # Check if this face is part of an internal corner or pocket
            # by examining local geometry

            # Simple heuristic: faces with normals pointing "inward" relative to
            # the overall mesh orientation are more likely to be problematic

            # Calculate distance to mesh centroid
            mesh_center = np.mean(self.mesh.vertices, axis=0)
            to_center_vector = mesh_center - face_center
            to_center_vector = to_center_vector / np.linalg.norm(to_center_vector)

            # If face normal points toward mesh center, it might be internal
            if np.dot(face_normal, to_center_vector) > 0.3:
                problematic_faces.append(face_idx)

        return np.array(problematic_faces)

    def _cluster_undercut_faces(self, face_indices: np.ndarray) -> list[np.ndarray]:
        """
        Cluster nearby undercut faces into coherent regions.
        """
        if len(face_indices) == 0:
            return []

        # Get face centers for clustering
        face_centers = self.mesh.triangles_center[face_indices]

        # Use DBSCAN to cluster nearby faces
        clustering = DBSCAN(eps=self.voxel_resolution * 3, min_samples=1)
        cluster_labels = clustering.fit_predict(face_centers)

        # Group faces by cluster
        clustered_regions = []
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue

            cluster_mask = cluster_labels == label
            cluster_faces = face_indices[cluster_mask]
            clustered_regions.append(cluster_faces)

        return clustered_regions

    def _identify_steep_walls(self) -> list[np.ndarray]:
        """
        Identify regions with steep wall angles that might be difficult to machine.

        This implementation analyzes:
        1. Wall angle relative to machining directions
        2. Tool deflection risk in steep walls
        3. Surface finish quality concerns
        4. Chatter and vibration susceptibility
        """
        try:
            steep_wall_regions = []

            # Method 1: Angle-based analysis
            angle_based_steep_faces = self._find_steep_angle_faces()
            if len(angle_based_steep_faces) > 0:
                steep_wall_regions.extend(angle_based_steep_faces)

            # Method 2: Tool deflection analysis
            deflection_risk_faces = self._find_tool_deflection_risk_faces()
            if len(deflection_risk_faces) > 0:
                steep_wall_regions.extend(deflection_risk_faces)

            # Method 3: Surface finish concern analysis
            finish_concern_faces = self._find_surface_finish_concern_faces()
            if len(finish_concern_faces) > 0:
                steep_wall_regions.extend(finish_concern_faces)

            # Combine and cluster steep wall regions
            if steep_wall_regions:
                unique_faces = np.unique(np.concatenate(steep_wall_regions))
                clustered_steep_walls = self._cluster_steep_wall_faces(unique_faces)

                logging.info(
                    f"Identified {len(clustered_steep_walls)} steep wall regions with {len(unique_faces)} total faces")
                return clustered_steep_walls

            return []

        except Exception as e:
            logging.error(f"Error in steep wall identification: {e}")
            return []

    def _find_steep_angle_faces(self) -> list[np.ndarray]:
        """
        Find faces with steep angles relative to primary machining directions.
        """
        face_normals = self.mesh.face_normals

        # Primary machining directions (tool approach vectors)
        machining_directions = [
            np.array([0, 0, 1]),  # Z-up (vertical machining)
            np.array([1, 0, 0]),  # X-direction (horizontal)
            np.array([0, 1, 0]),  # Y-direction (horizontal)
        ]

        steep_face_sets = []

        for direction in machining_directions:
            # Calculate angle between face normal and machining direction
            dot_products = np.abs(np.dot(face_normals, direction))
            angles = np.arccos(np.clip(dot_products, 0, 1))

            # Convert to degrees from vertical (for walls)
            wall_angles = np.pi / 2 - angles  # Angle from horizontal

            # Find faces exceeding the steep wall threshold
            steep_mask = wall_angles > self.max_wall_angle

            if np.any(steep_mask):
                steep_face_indices = np.where(steep_mask)[0]

                # Additional filtering based on wall characteristics
                filtered_faces = self._filter_steep_faces_by_characteristics(
                    steep_face_indices, wall_angles[steep_mask], direction
                )

                if len(filtered_faces) > 0:
                    steep_face_sets.append(filtered_faces)

        return steep_face_sets

    def _filter_steep_faces_by_characteristics(self, face_indices: np.ndarray,
                                               wall_angles: np.ndarray,
                                               machining_direction: np.ndarray) -> np.ndarray:
        """
        Filter steep faces based on additional geometric characteristics.
        """
        filtered_faces = []
        face_centers = self.mesh.triangles_center[face_indices]

        for i, face_idx in enumerate(face_indices):
            face_center = face_centers[i]
            wall_angle = wall_angles[i]

            # Check if this is actually a problematic steep wall
            # vs. a legitimate steep feature that's manageable

            # 1. Check wall height (tall steep walls are more problematic)
            wall_height = self._estimate_wall_height(face_idx, machining_direction)

            # 2. Check if wall is part of a deep feature
            is_in_deep_feature = self._is_face_in_deep_feature(face_idx)

            # 3. Check wall aspect ratio
            wall_aspect_ratio = self._calculate_wall_aspect_ratio(face_idx)

            # Apply filtering criteria
            is_problematic = (
                    wall_angle > np.radians(80) or  # Very steep (>80 degrees)
                    (wall_angle > np.radians(75) and wall_height > 20) or  # Steep and tall
                    (wall_angle > np.radians(70) and is_in_deep_feature) or  # Steep in deep feature
                    wall_aspect_ratio > 15  # Very high aspect ratio
            )

            if is_problematic:
                filtered_faces.append(face_idx)

        return np.array(filtered_faces)

    def _estimate_wall_height(self, face_idx: int, machining_direction: np.ndarray) -> float:
        """
        Estimate the height of a wall face in the machining direction.
        """
        face_center = self.mesh.triangles_center[face_idx]

        # Cast rays in both directions along machining axis to find wall extent
        ray_directions = [machining_direction, -machining_direction]
        wall_extents = []

        for ray_dir in ray_directions:
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=face_center.reshape(1, -1),
                ray_directions=ray_dir.reshape(1, -1)
            )

            if len(locations) > 0:
                distances = np.linalg.norm(locations - face_center, axis=1)
                nearest_distance = np.min(distances[distances > 0.1])  # Ignore very close intersections
                wall_extents.append(nearest_distance)

        # Return total wall height
        return sum(wall_extents) if wall_extents else 0.0

    def _is_face_in_deep_feature(self, face_idx: int) -> bool:
        """
        Check if a face is part of a deep pocket or cavity.
        """
        face_center = self.mesh.triangles_center[face_idx]

        # Simple heuristic: check if face is significantly below the mesh's top surface
        mesh_bbox = self.mesh.bounds
        mesh_height = mesh_bbox[1][2] - mesh_bbox[0][2]
        face_depth_ratio = (mesh_bbox[1][2] - face_center[2]) / mesh_height

        # Consider faces in bottom 30% of mesh height as potentially in deep features
        return face_depth_ratio > 0.3

    def _calculate_wall_aspect_ratio(self, face_idx: int) -> float:
        """
        Calculate the aspect ratio of a wall face (height/width).
        """
        # Get the triangle vertices
        face_vertices = self.mesh.vertices[self.mesh.faces[face_idx]]

        # Calculate edge lengths
        edge_lengths = [
            np.linalg.norm(face_vertices[1] - face_vertices[0]),
            np.linalg.norm(face_vertices[2] - face_vertices[1]),
            np.linalg.norm(face_vertices[0] - face_vertices[2])
        ]

        # Aspect ratio approximation
        max_edge = max(edge_lengths)
        min_edge = min(edge_lengths)

        return max_edge / max(min_edge, 0.1)  # Avoid division by zero

    def _find_tool_deflection_risk_faces(self) -> list[np.ndarray]:
        """
        Identify faces where tool deflection might cause machining problems.
        """
        risk_faces = []
        face_centers = self.mesh.triangles_center
        face_normals = self.mesh.face_normals

        # Calculate required tool extension for each face
        for face_idx in range(len(face_centers)):
            face_center = face_centers[face_idx]
            face_normal = face_normals[face_idx]

            # Estimate required tool extension to reach this face
            tool_extension = self._calculate_required_tool_extension(face_idx)

            # Check if extension exceeds safe limits based on tool diameter
            max_safe_extension = self.min_tool_diameter * self.max_depth_to_diameter_ratio

            if tool_extension > max_safe_extension:
                # Additional check: is this a steep wall?
                wall_angle = self._calculate_wall_angle_from_vertical(face_normal)

                if wall_angle > np.radians(60):  # Steep wall requiring long tool
                    risk_faces.append(face_idx)

        return [np.array(risk_faces)] if risk_faces else []

    def _calculate_required_tool_extension(self, face_idx: int) -> float:
        """
        Calculate the minimum tool extension required to reach a face.
        """
        face_center = self.mesh.triangles_center[face_idx]
        face_normal = self.mesh.face_normals[face_idx]

        # Find the nearest external surface that a tool holder might contact
        # This is a simplified calculation - in practice, you'd need detailed
        # tool holder geometry and clearance analysis

        # Cast ray outward from face in normal direction
        ray_origin = face_center
        ray_direction = face_normal

        locations, _, _ = self.mesh.ray.intersects_location(
            ray_origins=ray_origin.reshape(1, -1),
            ray_directions=ray_direction.reshape(1, -1)
        )

        if len(locations) > 0:
            distances = np.linalg.norm(locations - face_center, axis=1)
            # Return distance to nearest surface in normal direction
            return np.min(distances[distances > 0.1])

        # Default assumption if no intersection found
        return 50.0  # mm

    def _calculate_wall_angle_from_vertical(self, face_normal: np.ndarray) -> float:
        """
        Calculate the angle of a face from vertical (Z-axis).
        """
        vertical = np.array([0, 0, 1])
        dot_product = np.abs(np.dot(face_normal, vertical))
        angle_from_vertical = np.arccos(np.clip(dot_product, 0, 1))
        return angle_from_vertical

    def _find_surface_finish_concern_faces(self) -> list[np.ndarray]:
        """
        Identify faces where steep angles might compromise surface finish quality.
        """
        face_normals = self.mesh.face_normals
        concern_faces = []

        # Surface finish is typically compromised when:
        # 1. Wall angle is very steep (>85 degrees)
        # 2. Face is part of a narrow channel or slot
        # 3. Face has high curvature in steep regions

        for face_idx in range(len(face_normals)):
            face_normal = face_normals[face_idx]

            # Check wall steepness
            wall_angle = self._calculate_wall_angle_from_vertical(face_normal)

            if wall_angle > np.radians(85):  # Very steep walls
                # Additional checks for surface finish concerns
                is_in_narrow_channel = self._is_face_in_narrow_channel(face_idx)
                has_high_curvature = self._face_has_high_curvature(face_idx)

                if is_in_narrow_channel or has_high_curvature:
                    concern_faces.append(face_idx)

        return [np.array(concern_faces)] if concern_faces else []

    def _is_face_in_narrow_channel(self, face_idx: int) -> bool:
        """
        Check if a face is part of a narrow channel or slot.
        """
        face_center = self.mesh.triangles_center[face_idx]
        face_normal = self.mesh.face_normals[face_idx]

        # Cast rays perpendicular to face normal to measure channel width
        # Create two perpendicular directions to face normal
        if abs(face_normal[2]) < 0.9:
            perp1 = np.cross(face_normal, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(face_normal, np.array([1, 0, 0]))

        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(face_normal, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        # Measure distances in perpendicular directions
        channel_widths = []

        for direction in [perp1, -perp1, perp2, -perp2]:
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=face_center.reshape(1, -1),
                ray_directions=direction.reshape(1, -1)
            )

            if len(locations) > 0:
                distances = np.linalg.norm(locations - face_center, axis=1)
                nearest_distance = np.min(distances[distances > 0.1])
                channel_widths.append(nearest_distance)

        # Consider it a narrow channel if minimum width is small
        min_width = min(channel_widths) if channel_widths else float('inf')
        return min_width < self.min_tool_diameter * 3  # Less than 3x tool diameter

    def _face_has_high_curvature(self, face_idx: int) -> bool:
        """
        Check if a face is in a region of high curvature.
        """
        # Get vertices of the face
        face_vertices_idx = self.mesh.faces[face_idx]

        # Simple curvature estimation based on adjacent faces
        adjacent_faces = []
        for vertex_idx in face_vertices_idx:
            vertex_faces = np.where(np.any(self.mesh.faces == vertex_idx, axis=1))[0]
            adjacent_faces.extend(vertex_faces)

        adjacent_faces = np.unique(adjacent_faces)
        adjacent_faces = adjacent_faces[adjacent_faces != face_idx]  # Remove self

        if len(adjacent_faces) < 2:
            return False

        # Calculate normal deviation
        face_normal = self.mesh.face_normals[face_idx]
        adjacent_normals = self.mesh.face_normals[adjacent_faces]

        # High curvature if normal deviates significantly from adjacent normals
        dot_products = np.dot(adjacent_normals, face_normal)
        mean_alignment = np.mean(dot_products)

        return mean_alignment < 0.8  # Normals deviate more than ~36 degrees

    def _cluster_steep_wall_faces(self, face_indices: np.ndarray) -> list[np.ndarray]:
        """
        Cluster nearby steep wall faces into coherent regions.
        """
        if len(face_indices) == 0:
            return []

        # Get face centers for clustering
        face_centers = self.mesh.triangles_center[face_indices]

        # Use DBSCAN with slightly larger epsilon for wall clustering
        clustering = DBSCAN(eps=self.voxel_resolution * 4, min_samples=2)
        cluster_labels = clustering.fit_predict(face_centers)

        # Group faces by cluster
        clustered_regions = []
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            if label == -1:  # Noise points
                continue

            cluster_mask = cluster_labels == label
            cluster_faces = face_indices[cluster_mask]

            # Only keep clusters with significant number of faces
            if len(cluster_faces) >= 3:  # Minimum cluster size
                clustered_regions.append(cluster_faces)

        return clustered_regions

    def _calculate_minimum_tool_diameter(self) -> float:
        """
        Calculate the minimum tool diameter required based on internal radii,
        corner features, and geometric constraints.

        This implementation analyzes:
        1. Internal corner radii and fillets
        2. Slot widths and channel dimensions
        3. Minimum feature sizes
        4. Tool access constraints
        """
        try:
            min_diameters = []

            # Method 1: Internal corner and fillet analysis
            corner_diameter = self._analyze_internal_corners()
            if corner_diameter > 0:
                min_diameters.append(corner_diameter)

            # Method 2: Slot and channel width analysis
            channel_diameter = self._analyze_channel_widths()
            if channel_diameter > 0:
                min_diameters.append(channel_diameter)

            # Method 3: Minimum feature size analysis
            feature_diameter = self._analyze_minimum_features()
            if feature_diameter > 0:
                min_diameters.append(feature_diameter)

            # Method 4: Curvature-based analysis
            curvature_diameter = self._analyze_curvature_constraints()
            if curvature_diameter > 0:
                min_diameters.append(curvature_diameter)

            # Return the most restrictive (largest) minimum diameter
            if min_diameters:
                calculated_min = max(min_diameters)
                logging.info(f"Calculated minimum tool diameter: {calculated_min:.2f}mm")
                return calculated_min
            else:
                # Return default minimum if no specific constraints found
                return self.min_tool_diameter

        except Exception as e:
            logging.error(f"Error calculating minimum tool diameter: {e}")
            return self.min_tool_diameter

    def _analyze_internal_corners(self) -> float:
        """
        Analyze internal corners and fillets to determine minimum tool diameter.
        """
        # Find vertices that form internal corners (concave regions)
        vertex_curvatures = self._calculate_vertex_curvatures()

        # Identify highly concave vertices (internal corners)
        internal_corner_threshold = -0.5
        internal_corners = np.where(vertex_curvatures < internal_corner_threshold)[0]

        if len(internal_corners) == 0:
            return 0.0

        min_corner_radii = []

        for corner_vertex in internal_corners:
            # Calculate the effective radius of curvature at this corner
            corner_radius = self._calculate_corner_radius(corner_vertex)

            if corner_radius > 0:
                # Tool diameter must be smaller than twice the corner radius
                # to fit into the corner (with some clearance)
                max_tool_diameter = corner_radius * 1.8  # 90% of theoretical max
                min_corner_radii.append(max_tool_diameter)

        return min(min_corner_radii) if min_corner_radii else 0.0

    def _calculate_corner_radius(self, vertex_idx: int) -> float:
        """
        Calculate the effective radius of curvature at a vertex.
        """
        vertex_pos = self.mesh.vertices[vertex_idx]

        # Find adjacent faces
        adjacent_faces = np.where(np.any(self.mesh.faces == vertex_idx, axis=1))[0]

        if len(adjacent_faces) < 2:
            return 0.0

        # Calculate edge vectors from this vertex
        edge_vectors = []
        for face_idx in adjacent_faces:
            face = self.mesh.faces[face_idx]
            other_vertices = face[face != vertex_idx]

            for other_vertex in other_vertices:
                edge_vector = self.mesh.vertices[other_vertex] - vertex_pos
                edge_length = np.linalg.norm(edge_vector)
                if edge_length > 0.001:  # Avoid very small edges
                    edge_vectors.append(edge_vector / edge_length)

        if len(edge_vectors) < 2:
            return 0.0

        # Find the tightest corner (smallest angle between edges)
        min_angle = np.pi
        for i in range(len(edge_vectors)):
            for j in range(i + 1, len(edge_vectors)):
                dot_product = np.clip(np.dot(edge_vectors[i], edge_vectors[j]), -1, 1)
                angle = np.arccos(dot_product)
                min_angle = min(min_angle, angle)

        # Estimate radius based on local edge lengths and corner angle
        # This is a simplified geometric approximation
        avg_edge_length = np.mean([np.linalg.norm(ev) for ev in edge_vectors])

        if min_angle > 0.1:  # Avoid division by very small angles
            # Approximate radius for inscribed circle in corner
            estimated_radius = avg_edge_length * np.sin(min_angle / 2) / (1 + np.cos(min_angle / 2))
            return max(estimated_radius, 0.1)  # Minimum 0.1mm radius

        return 0.1  # Default small radius

    def _analyze_channel_widths(self) -> float:
        """
        Analyze narrow channels and slots to determine minimum tool diameter.
        """
        narrow_channels = []
        face_centers = self.mesh.triangles_center
        face_normals = self.mesh.face_normals

        # Sample faces to check for narrow channels
        sample_indices = np.arange(0, len(face_centers), max(1, len(face_centers) // 100))

        for face_idx in sample_indices:
            face_center = face_centers[face_idx]
            face_normal = face_normals[face_idx]

            # Measure channel width at this face
            channel_width = self._measure_local_channel_width(face_center, face_normal)

            if 0 < channel_width < 20:  # Only consider realistic channel widths
                narrow_channels.append(channel_width)

        if narrow_channels:
            # Tool diameter should be smaller than the narrowest channel
            min_channel_width = min(narrow_channels)
            max_tool_diameter = min_channel_width * 0.8  # 80% of channel width for clearance
            return max_tool_diameter

        return 0.0

    def _measure_local_channel_width(self, center_point: np.ndarray, normal: np.ndarray) -> float:
        """
        Measure the width of a channel or slot at a specific point.
        """
        # Create perpendicular directions to the normal
        if abs(normal[2]) < 0.9:
            perp1 = np.cross(normal, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(normal, np.array([1, 0, 0]))

        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(normal, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        # Measure distances in both perpendicular directions
        widths = []

        for direction in [perp1, perp2]:
            # Cast rays in both directions
            distances_both_ways = []

            for ray_dir in [direction, -direction]:
                locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=center_point.reshape(1, -1),
                    ray_directions=ray_dir.reshape(1, -1)
                )

                if len(locations) > 0:
                    distances = np.linalg.norm(locations - center_point, axis=1)
                    nearest_distance = np.min(distances[distances > 0.1])
                    distances_both_ways.append(nearest_distance)

            # Total width in this direction
            if len(distances_both_ways) == 2:
                total_width = sum(distances_both_ways)
                widths.append(total_width)

        return min(widths) if widths else 0.0

    def _analyze_minimum_features(self) -> float:
        """
        Analyze minimum feature sizes that constrain tool selection.
        """
        # Find small geometric features using edge length analysis
        edge_lengths = self.mesh.edges_unique_length

        # Filter out very small edges (likely mesh artifacts)
        meaningful_edges = edge_lengths[edge_lengths > 0.1]

        if len(meaningful_edges) == 0:
            return 0.0

        # Find small features that might constrain tool size
        small_features = meaningful_edges[meaningful_edges < 10.0]  # Features smaller than 10mm

        if len(small_features) > 0:
            # Tool should be smaller than the smallest meaningful feature
            min_feature_size = np.min(small_features)
            max_tool_diameter = min_feature_size * 0.5  # Tool diameter = 50% of feature size
            return max_tool_diameter

        return 0.0

    def _analyze_curvature_constraints(self) -> float:
        """
        Analyze surface curvature to determine tool size constraints.
        """
        try:
            # Sample points on the mesh surface
            sample_points, face_indices = self.mesh.sample(1000, return_index=True)

            curvature_constraints = []

            for i, point in enumerate(sample_points):
                face_idx = face_indices[i]

                # Estimate local curvature radius
                local_curvature_radius = self._estimate_local_curvature_radius(point, face_idx)

                if 0 < local_curvature_radius < 50:  # Reasonable curvature range
                    # Tool radius should be smaller than curvature radius for good surface contact
                    max_tool_radius = local_curvature_radius * 0.8
                    max_tool_diameter = max_tool_radius * 2
                    curvature_constraints.append(max_tool_diameter)

            if curvature_constraints:
                return min(curvature_constraints)

        except Exception as e:
            logging.warning(f"Curvature analysis failed: {e}")

        return 0.0

    def _estimate_local_curvature_radius(self, point: np.ndarray, face_idx: int) -> float:
        """
        Estimate the local radius of curvature at a point on the mesh.
        """
        # Get face normal at this point
        face_normal = self.mesh.face_normals[face_idx]

        # Find nearby faces within a small radius
        face_centers = self.mesh.triangles_center
        distances = np.linalg.norm(face_centers - point, axis=1)
        nearby_faces = np.where(distances < 2.0)[0]  # Within 2mm

        if len(nearby_faces) < 3:
            return 0.0

        # Calculate normal variation in local neighborhood
        nearby_normals = self.mesh.face_normals[nearby_faces]

        # Estimate curvature from normal deviation
        normal_deviations = []
        for normal in nearby_normals:
            deviation_angle = np.arccos(np.clip(np.dot(face_normal, normal), -1, 1))
            normal_deviations.append(deviation_angle)

        max_deviation = max(normal_deviations)

        if max_deviation > 0.01:  # More than ~0.5 degrees
            # Rough curvature radius estimation
            # R ≈ arc_length / angle
            arc_length = 2.0  # Our search radius
            curvature_radius = arc_length / max_deviation
            return curvature_radius

        return 0.0  # Nearly flat surface

    def _calculate_difficulty_score(
            self,
            deep_pockets: list[DeepPocket],
            undercuts: list[np.ndarray],
            steep_walls: list[np.ndarray],
            min_tool_dia: float
    ) -> float:
        """
        Calculate a normalized difficulty score (0-1) based on various factors.
        """
        score = 0.0

        # Score based on deep pockets
        if deep_pockets:
            max_aspect_ratio = max(pocket.aspect_ratio for pocket in deep_pockets)
            pocket_score = min(max_aspect_ratio / (self.max_depth_to_diameter_ratio * 2), 0.4)
            score += pocket_score

        # Score based on undercuts (high penalty)
        if undercuts:
            score += 0.3

        # Score based on steep walls
        if steep_walls:
            score += 0.2

        # Score based on required tool diameter
        if min_tool_dia < self.min_tool_diameter:
            score += 0.1

        return min(score, 1.0)

    def _recommend_processes(
            self,
            deep_pockets: list[DeepPocket],
            undercuts: list[np.ndarray],
            steep_walls: list[np.ndarray]
    ) -> list[ManufacturingProcess]:
        """
        Recommend suitable manufacturing processes based on part features.
        """
        recommendations = []

        # Check for complex features requiring 5-axis
        needs_5_axis = (
                len(undercuts) > 0 or
                any(pocket.aspect_ratio > self.max_depth_to_diameter_ratio * 0.8 for pocket in deep_pockets) or
                len(steep_walls) > 0
        )

        if needs_5_axis:
            recommendations.append(ManufacturingProcess.CNC_5_AXIS)
        else:
            recommendations.append(ManufacturingProcess.CNC_3_AXIS)
            recommendations.append(ManufacturingProcess.MILLING)

        return recommendations



    def _add_text_summary(self, scene: trimesh.Scene, result: ManufacturabilityResult):
        """Add comprehensive text summary to scene metadata."""
        summary = {
            'manufacturability_status': 'PASS' if result.is_manufacturable else 'FAIL',
            'difficulty_score': f"{result.difficulty_score:.2f}/1.0",
            'issues_count': len(result.issues),
            'issues': result.issues,
            'deep_pockets_count': len(result.deep_pocket_regions),
            'undercuts_count': len(result.undercut_regions),
            'steep_walls_count': len(result.steep_angles),
            'min_tool_diameter': f"{result.minimum_tool_diameter:.1f}mm",
            'recommended_processes': [proc.value for proc in result.recommended_processes]
        }

        # Store summary in scene metadata
        scene.metadata['manufacturability_analysis'] = summary




    def generate_report(self) -> dict:
        """
        Generate a comprehensive manufacturability report.
        """
        result = self.analyze_manufacturability()

        report = {
            "manufacturability_summary": {
                "is_manufacturable": result.is_manufacturable,
                "difficulty_score": result.difficulty_score,
                "issues": result.issues
            },
            "deep_pockets_analysis": {
                "count": len(result.deep_pocket_regions),
                "details": []
            },
            "recommendations": {
                "processes": [proc.value for proc in result.recommended_processes],
                "minimum_tool_diameter": result.minimum_tool_diameter
            }
        }

        return report
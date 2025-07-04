
# =============================================================================
# Cell 14: Find Narrow Channels Function
# =============================================================================
def find_narrow_channels(mesh, min_width=5.0, sample_size=500):
    """
    Find faces in channels too narrow for standard tooling.
    Minimum practical channel width is ~3x tool diameter.
    """
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    narrow_faces = []

    # Sample subset of faces for performance
    actual_sample_size = min(sample_size, len(face_centers))
    sample_indices = np.random.choice(len(face_centers), actual_sample_size, replace=False)

    print(f"Checking {actual_sample_size} faces for narrow channels...")

    for idx in sample_indices:
        center = face_centers[idx]
        normal = face_normals[idx]

        # Measure channel width by casting rays perpendicular to normal
        width = measure_channel_width(mesh, center, normal)

        # If channel is very narrow (< min_width mm)
        if 0 < width < min_width:
            narrow_faces.append(idx)

    print(f"Found {len(narrow_faces)} faces in narrow channels")
    return np.array(narrow_faces)

# Test this function:
# narrow_channels = find_narrow_channels(mesh)


# =============================================================================
# Cell 15: Measure Channel Width Helper Function
# =============================================================================
def measure_channel_width(mesh, center, normal):
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
            try:
                locations, _, _ = mesh.ray.intersects_location(
                    ray_origins=center.reshape(1, -1),
                    ray_directions=direction.reshape(1, -1)
                )
                if len(locations) > 0:
                    dist = np.min(np.linalg.norm(locations - center, axis=1))
                    distances.append(dist)
            except:
                continue

        return sum(distances) if len(distances) == 2 else 0
    except:
        return 0

# Test this function with a face center and normal:
# width = measure_channel_width(mesh, mesh.triangles_center[0], mesh.face_normals[0])


# =============================================================================
# Cell 16: Find Deep Pockets (Enhanced) Function
# =============================================================================
def find_deep_pockets_enhanced(mesh, depth_threshold=20.0):
    """
    Find faces in very deep pockets where tool length becomes an issue.
    Depth-to-width ratio > 5:1 is typically problematic.
    """
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    deep_faces = []

    print(f"Checking for pockets deeper than {depth_threshold} units...")

    for i, (center, normal) in enumerate(zip(face_centers, face_normals)):
        # Estimate depth by measuring distance to part boundary
        depth = estimate_pocket_depth(mesh, center, normal)

        if depth > depth_threshold:
            deep_faces.append(i)

    print(f"Found {len(deep_faces)} faces in deep pockets")
    return np.array(deep_faces)

# Test this function:
# deep_pockets_enhanced = find_deep_pockets_enhanced(mesh)


# =============================================================================
# Cell 17: Estimate Pocket Depth Helper Function
# =============================================================================
def estimate_pocket_depth(mesh, center, normal):
    """Estimate the depth of a pocket at a specific face."""
    try:
        # Cast ray outward from face
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=center.reshape(1, -1),
            ray_directions=normal.reshape(1, -1)
        )

        if len(locations) > 0:
            depths = np.linalg.norm(locations - center, axis=1)
            valid_depths = depths[depths > 0.1]  # Ignore very close hits
            return np.min(valid_depths) if len(valid_depths) > 0 else 0

        return 0
    except:
        return 0

# Test this function:
# depth = estimate_pocket_depth(mesh, mesh.triangles_center[0], mesh.face_normals[0])


# =============================================================================
# Cell 18: Check Tool Path Clear Function
# =============================================================================
def is_tool_path_clear(mesh, face_center, tool_direction):
    """Check if a cutting tool can reach the face from given direction."""
    try:
        # Cast ray from far away toward the face
        ray_start = face_center - tool_direction * 100  # Start 100 units away

        # Check for intersections between start and face
        locations, _, _ = mesh.ray.intersects_location(
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

# Test this function:
# is_clear = is_tool_path_clear(mesh, mesh.triangles_center[0], np.array([0, 0, 1]))


# =============================================================================
# Cell 19: Calculate CNC Manufacturability Score Function
# =============================================================================
def calculate_cnc_manufacturability_score(mesh):
    """
    Analyze manufacturability for CNC milling.
    Returns score 0-100 (100 = perfect for CNC, 0 = impossible)
    """
    manufacturability_score = 100
    problem_regions = []

    print("Analyzing CNC manufacturability...")

    # 1. Check for undercuts (biggest CNC problem)
    undercuts = find_undercuts(mesh)
    if len(undercuts) > 0:
        penalty = min(50, len(undercuts) * 0.1)  # Up to 50 point penalty
        manufacturability_score -= penalty
        problem_regions.append(("Undercuts (Cannot machine)", undercuts))

    # 2. Check for enclosed internal volumes (impossible to machine)
    internal_volumes = find_internal_volumes(mesh)
    if internal_volumes > 0:
        manufacturability_score -= 40
        problem_regions.append(("Internal Volumes", []))

    # 3. Check for very narrow channels (tool access limited)
    narrow_channels = find_narrow_channels(mesh)
    if len(narrow_channels) > 0:
        penalty = min(25, len(narrow_channels) * 0.05)
        manufacturability_score -= penalty
        problem_regions.append(("Narrow Channels", narrow_channels))

    # 4. Check for very deep pockets (tool length/rigidity issues)
    deep_pockets = find_deep_pockets_enhanced(mesh)
    if len(deep_pockets) > 0:
        penalty = min(20, len(deep_pockets) * 0.03)
        manufacturability_score -= penalty
        problem_regions.append(("Deep Pockets", deep_pockets))

    # 5. Check for small features (below minimum tool size)
    if find_small_features(mesh):
        manufacturability_score -= 15
        problem_regions.append(("Small Features", []))

    manufacturability_score = max(0, manufacturability_score)
    print(f"Final CNC Manufacturability Score: {manufacturability_score}/100")

    return manufacturability_score, problem_regions

# Test this function:
# cnc_score, cnc_regions = calculate_cnc_manufacturability_score(mesh)


# =============================================================================
# Cell 20: Plot CNC Analysis Function
# =============================================================================
def plot_cnc_analysis(mesh, problem_regions, manufacturability_score):
    """Plot mesh with CNC problem regions highlighted."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    vertices = mesh.vertices
    faces = mesh.faces

    # Default color for all faces (good for CNC)
    face_colors = ['lightgreen'] * len(faces)  # Green = good for CNC

    # Color problem regions
    colors = ['red', 'orange', 'yellow', 'purple']
    for i, (region_name, face_indices) in enumerate(problem_regions):
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

    ax.set_title(f'CNC Manufacturability Analysis\nScore: {manufacturability_score}/100',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend
    legend_text = ["Green: Good for CNC"]
    if problem_regions:
        for i, (region_name, _) in enumerate(problem_regions):
            legend_text.append(f"{colors[i % len(colors)]}: {region_name}")

    ax.text2D(0.02, 0.98, '\n'.join(legend_text), transform=ax.transAxes,
              verticalalignment='top', fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.show()
    return fig

# Test this function:
# plot_cnc_analysis(mesh, cnc_regions, cnc_score)


# =============================================================================
# Cell 21: Print CNC Report Function
# =============================================================================
def print_cnc_report(stl_path, mesh, manufacturability_score, problem_regions):
    """Print a CNC-focused manufacturability report."""
    print("=" * 50)
    print("CNC MANUFACTURABILITY REPORT")
    print("=" * 50)
    print(f"STL File: {os.path.basename(stl_path)}")
    print(f"Faces: {len(mesh.faces):,}")
    print(f"Vertices: {len(mesh.vertices):,}")
    print(f"Volume: {mesh.volume:.2f} cubic units")
    print(f"Watertight: {'Yes' if mesh.is_watertight else 'No'}")

    print(f"\nCNC Manufacturability Score: {manufacturability_score}/100")

    if manufacturability_score >= 80:
        print("CNC Suitability: EXCELLENT - Ideal for CNC machining")
    elif manufacturability_score >= 60:
        print("CNC Suitability: GOOD - Suitable with standard tooling")
    elif manufacturability_score >= 40:
        print("CNC Suitability: MODERATE - May need special tooling/setup")
    elif manufacturability_score >= 20:
        print("CNC Suitability: POOR - Significant CNC challenges")
    else:
        print("CNC Suitability: VERY POOR - Not recommended for CNC")

    print(f"\nCNC Problem Areas Found: {len(problem_regions)}")
    for region_name, faces in problem_regions:
        severity = "CRITICAL" if "Undercuts" in region_name else "MODERATE"
        face_count = len(faces) if hasattr(faces, '__len__') else "N/A"
        print(f"  - {region_name}: {face_count} faces ({severity})")

    print("\nCNC Recommendations:")
    if manufacturability_score < 40:
        print("  - Consider design changes to eliminate undercuts")
        print("  - Evaluate if part can be made in multiple pieces")
        print("  - Consider alternative manufacturing (3D printing, casting)")
    elif manufacturability_score < 70:
        print("  - Plan for specialized tooling")
        print("  - Consider 5-axis CNC for better access")
        print("  - Verify tool length requirements")
    else:
        print("  - Standard 3-axis CNC should work well")
        print("  - Good candidate for CNC manufacturing")

# Test this function:
# print_cnc_report("test_file/simple/cube.stl", mesh, cnc_score, cnc_regions)


# =============================================================================
# Cell 22: Complete CNC Analysis Function
# =============================================================================
def analyze_cnc_suitability_complete(stl_path, save_image=None):
    """
    Complete CNC analysis workflow combining all functions.
    """
    # Load mesh
    mesh = load_stl(stl_path)
    if mesh is None:
        return None

    # Calculate CNC manufacturability
    manufacturability_score, problem_regions = calculate_cnc_manufacturability_score(mesh)

    # Print detailed report
    print_cnc_report(stl_path, mesh, manufacturability_score, problem_regions)

    # Create visualizations
    plot_mesh(mesh, "Original STL Model")
    plot_cnc_analysis(mesh, problem_regions, manufacturability_score)

    # Decision making
    if manufacturability_score >= 60:
        print(f"\n✓ ACCEPT: This part is suitable for CNC (Score: {manufacturability_score})")
    else:
        print(f"\n✗ REJECT: This part is not suitable for CNC (Score: {manufacturability_score})")

    return mesh, manufacturability_score, problem_regions

# Test complete CNC analysis:
# result = analyze_cnc_suitability_complete("test_file/simple/cube.stl")


# =============================================================================
# Cell 23: Test CNC Analysis with Multiple Files
# =============================================================================
# Test with different complexity files
test_files = [
    "test_file/simple/cube.stl",
    "test_file/simple/sphere.STL",
    "test_file/simple/cylinder.stl",
    "test_file/moderate/38mm-90degreeBracket.STL"
]

print("Testing CNC analysis on multiple files...")
for stl_file in test_files:
    if os.path.exists(stl_file):
        print(f"\n{'='*60}")
        print(f"Testing: {stl_file}")
        try:
            mesh, score, regions = analyze_cnc_suitability_complete(stl_file)
            print(f"Result: {score}/100")
        except Exception as e:
            print(f"Error analyzing {stl_file}: {e}")
    else:
        print(f"File not found: {stl_file}")

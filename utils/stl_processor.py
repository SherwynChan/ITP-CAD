# STL File Processing and Manufacturability Analysis System
# ===========================================================
# This system processes 3D CAD models (STL files) and analyzes their manufacturability
# for subtractive manufacturing processes like CNC machining and milling.
#
# Processing Pipeline:
# 1. Load STL file
# 2. Analyze mesh quality
# 3. Clean and repair mesh issues
# 4. Simplify mesh (reduce polygon count)
# 5. Apply transformations if needed
# 6. Extract geometric features
# 7. Analyze manufacturability
# 8. Generate visualizations and reports

import numpy as np
import trimesh
import os
from manufacturability_analyzer import ManufacturabilityAnalyzer
from mesh_visualization import EnhancedMeshVisualization

# Path to the test STL file - modify this to point to your STL file
file_path = os.path.join(os.path.dirname(__file__), 'test.stl')


# ============================================================================
# STEP 1: STL FILE LOADING
# ============================================================================

def load_stl(file_path):
    """
    Load an STL file and return a trimesh object.

    STL (STereoLithography) files contain 3D mesh data represented as triangular faces.
    This function uses the trimesh library to parse the file and create a mesh object
    that we can analyze and manipulate.

    Args:
        file_path (str): Full path to the STL file to be loaded

    Returns:
        trimesh.Trimesh: Loaded mesh object containing vertices, faces, and normals
        None: If loading fails due to file not found, corruption, or format issues
    """
    try:
        # trimesh.load() automatically detects file format and loads the mesh
        mesh = trimesh.load(file_path)

        # Print basic mesh statistics for confirmation
        print(f"✓ Successfully loaded STL file: {os.path.basename(file_path)}")
        print(f"  - Faces (triangles): {len(mesh.faces):,}")
        print(f"  - Vertices (points): {len(mesh.vertices):,}")
        print(f"  - File size: {os.path.getsize(file_path) / 1024:.1f} KB")

        return mesh

    except FileNotFoundError:
        print(f"✗ Error: STL file not found at {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading STL file: {e}")
        print("  Common causes: corrupted file, unsupported format, or insufficient memory")
        return None


# ============================================================================
# STEP 2: MESH ANALYSIS AND VALIDATION
# ============================================================================

def analyze_mesh(mesh):
    """
    Perform comprehensive analysis and validation of the loaded mesh.

    This function checks for common mesh problems that could affect manufacturing
    analysis or cause processing errors. A high-quality mesh should be watertight
    (no holes), have consistent face winding, and no duplicate geometry.

    Args:
        mesh (trimesh.Trimesh): Input mesh to analyze

    Returns:
        dict: Comprehensive analysis results including:
            - is_watertight: Whether mesh has no holes (critical for volume calculations)
            - is_winding_consistent: Whether all faces point in consistent directions
            - has_duplicated_faces: Whether there are duplicate triangular faces
            - volume: Internal volume in cubic units
            - bounds: Bounding box coordinates [min_xyz, max_xyz]
            - centroid: Geometric center point [x, y, z]
            - is_convex: Whether the shape is convex (no indentations)
    """

    # Check for duplicate faces by comparing face indices
    # This is important because duplicates can cause rendering and analysis issues
    faces_view = mesh.faces.view(np.ndarray)
    unique_faces = np.unique(faces_view, axis=0)
    has_duplicated_faces = len(unique_faces) < len(mesh.faces)

    # Compile all analysis results
    results = {
        # Mesh topology and quality checks
        "is_watertight": mesh.is_watertight,  # No holes in the mesh
        "is_winding_consistent": mesh.is_winding_consistent,  # All faces oriented consistently
        "has_duplicated_faces": has_duplicated_faces,  # Duplicate triangular faces present

        # Geometric properties
        "volume": mesh.volume,  # Internal volume (cubic units)
        "bounds": mesh.bounds,  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        "centroid": mesh.centroid,  # Geometric center [x, y, z]
        "is_convex": mesh.is_convex  # Whether shape is convex
    }

    # Display analysis results with interpretations
    print("\n" + "=" * 50)
    print("MESH ANALYSIS RESULTS")
    print("=" * 50)

    for key, value in results.items():
        # Add interpretations for key properties
        interpretation = ""
        if key == "is_watertight":
            interpretation = " (✓ Good for manufacturing)" if value else " (⚠ May cause volume calculation errors)"
        elif key == "has_duplicated_faces":
            interpretation = " (⚠ Should be cleaned)" if value else " (✓ Clean geometry)"
        elif key == "is_winding_consistent":
            interpretation = " (✓ Proper face orientation)" if value else " (⚠ May cause rendering issues)"
        elif key == "volume" and value > 0:
            interpretation = f" (≈ {value:.1f} cubic units)"

        print(f"  {key}: {value}{interpretation}")

    return results


# ============================================================================
# STEP 3: MESH CLEANING AND REPAIR
# ============================================================================

def clean_mesh(mesh, repair_watertight=True, remove_duplicates=True):
    """
    Clean and repair common mesh issues to prepare for analysis.

    Manufacturing analysis requires high-quality mesh data. This function addresses
    common problems like duplicate faces and holes that could lead to incorrect
    analysis results or processing failures.

    Args:
        mesh (trimesh.Trimesh): Input mesh to clean
        repair_watertight (bool): Whether to attempt to fill holes and make watertight
        remove_duplicates (bool): Whether to remove duplicate triangular faces

    Returns:
        trimesh.Trimesh: Cleaned mesh with repairs applied
    """

    print("\n" + "=" * 50)
    print("MESH CLEANING AND REPAIR")
    print("=" * 50)

    # Create a copy to avoid modifying the original mesh
    mesh_copy = mesh.copy()
    changes = {}  # Track what changes were made

    # REMOVE DUPLICATE FACES
    # Duplicate faces can cause issues with volume calculations and rendering
    if remove_duplicates:
        print("Checking for duplicate faces...")

        # Find unique faces using numpy's unique function
        faces_view = mesh_copy.faces.view(np.ndarray)
        unique_faces, unique_indices = np.unique(faces_view, axis=0, return_index=True)

        # Check if we found duplicates
        has_duplicates = len(unique_indices) < len(mesh_copy.faces)

        if has_duplicates:
            original_faces = len(mesh_copy.faces)
            # Keep only the unique faces
            mesh_copy.faces = mesh_copy.faces[unique_indices]
            removed_count = original_faces - len(mesh_copy.faces)
            changes["duplicate_faces_removed"] = removed_count
            print(f"  ✓ Removed {removed_count} duplicate faces")
        else:
            changes["duplicate_faces_removed"] = 0
            print("  ✓ No duplicate faces found")

    # REPAIR WATERTIGHT ISSUES
    # A watertight mesh has no holes, which is essential for accurate volume calculations
    if repair_watertight and not mesh.is_watertight:
        print("Attempting to repair watertight issues...")

        try:
            # First, fix face normal directions
            # Inconsistent normals can cause inside/outside confusion
            mesh_copy.fix_normals()
            changes["normals_fixed"] = True
            print("  ✓ Fixed face normals")

            # Attempt to fill holes
            # Note: trimesh's hole filling is basic and may not work for complex holes
            try:
                mesh_copy.fill_holes()
                is_now_watertight = mesh_copy.is_watertight
                changes["holes_filled"] = is_now_watertight

                if is_now_watertight:
                    print("  ✓ Successfully filled holes - mesh is now watertight")
                else:
                    print("  ⚠ Hole filling attempted but mesh still not watertight")
                    print("    (Complex holes may require manual CAD repair)")

            except Exception as e:
                changes["holes_filled"] = False
                changes["hole_filling_error"] = str(e)
                print(f"  ⚠ Hole filling failed: {e}")

        except Exception as e:
            changes["repair_failed"] = True
            changes["repair_error"] = str(e)
            print(f"  ✗ Mesh repair failed: {e}")

    elif mesh.is_watertight:
        print("  ✓ Mesh is already watertight - no repair needed")

    # Summary of cleaning results
    print("\nCleaning Summary:")
    for key, value in changes.items():
        if key not in ["hole_filling_error", "repair_error"]:  # Skip error messages in summary
            print(f"  {key}: {value}")

    return mesh_copy


# ============================================================================
# STEP 4: MESH SIMPLIFICATION
# ============================================================================

def simplify_mesh(mesh, target_percent=0.5):
    """
    Reduce the number of polygons in the mesh while preserving overall shape.

    Mesh simplification is useful for:
    - Reducing processing time for complex models
    - Creating lower-detail versions for faster analysis
    - Reducing memory usage for large models

    The quadric decimation algorithm preserves important geometric features
    while removing less significant details.

    Args:
        mesh (trimesh.Trimesh): Input mesh to simplify
        target_percent (float): Percentage of faces to keep (0.0-1.0)
                               0.5 means keep 50% of faces, remove 50%

    Returns:
        trimesh.Trimesh: Simplified mesh with reduced polygon count
    """

    print("\n" + "=" * 50)
    print("MESH SIMPLIFICATION")
    print("=" * 50)

    # If target is 100% or more, return original mesh
    if target_percent >= 1.0:
        print("Target percentage >= 100% - returning original mesh")
        return mesh.copy()

    original_faces = len(mesh.faces)
    original_vertices = len(mesh.vertices)

    # Calculate target_reduction (opposite of target_percent)
    # If target_percent = 0.5 (keep 50%), then target_reduction = 0.5 (reduce by 50%)
    target_reduction = 1.0 - target_percent

    print(f"Original mesh: {original_faces:,} faces, {original_vertices:,} vertices")
    print(f"Target: Keep {target_percent:.0%} of faces (reduce by {target_reduction:.0%})")

    try:
        # Use trimesh's quadric decimation algorithm
        # This method preserves important geometric features better than random decimation
        simplified = mesh.simplify_quadric_decimation(target_reduction)

        # Calculate actual reduction achieved
        final_faces = len(simplified.faces)
        final_vertices = len(simplified.vertices)
        actual_percent = final_faces / original_faces

        print(f"✓ Simplification successful!")
        print(f"  Final mesh: {final_faces:,} faces, {final_vertices:,} vertices")
        print(f"  Actual reduction: {actual_percent:.1%} of original faces kept")
        print(f"  Face reduction: {original_faces - final_faces:,} faces removed")

        # Check if simplification was too aggressive
        if final_faces < 10:
            print("  ⚠ Warning: Very few faces remaining - may have lost important details")

        return simplified

    except Exception as e:
        print(f"✗ Simplification failed: {e}")
        print("  Possible causes: mesh too simple already, corrupted geometry")
        print("  Returning original mesh unchanged")
        return mesh.copy()


# ============================================================================
# STEP 5: MESH TRANSFORMATION
# ============================================================================

def transform_mesh(mesh, scale=None, rotation=None, translation=None):
    """
    Apply geometric transformations to position and orient the mesh.

    Transformations are useful for:
    - Standardizing part orientation for consistent analysis
    - Scaling parts to different sizes
    - Positioning parts at specific locations
    - Correcting import orientation issues

    Args:
        mesh (trimesh.Trimesh): Input mesh to transform
        scale (float or np.ndarray): Scale factor(s) - single value for uniform scaling,
                                   or [x, y, z] array for non-uniform scaling
        rotation (np.ndarray): Either a 3x3 rotation matrix or [roll, pitch, yaw]
                              Euler angles in radians
        translation (np.ndarray): Translation vector [x, y, z] in mesh units

    Returns:
        trimesh.Trimesh: Transformed mesh with applied transformations
    """

    print("\n" + "=" * 50)
    print("MESH TRANSFORMATION")
    print("=" * 50)

    # Create a copy to avoid modifying the original
    mesh_copy = mesh.copy()
    original_bounds = mesh_copy.bounds

    print(f"Original bounds: {original_bounds}")

    # APPLY SCALING
    if scale is not None:
        print("Applying scaling transformation...")

        # Convert single scale value to uniform scaling array
        if isinstance(scale, (int, float)):
            scale_array = np.array([scale, scale, scale])
            print(f"  Uniform scaling: {scale}x")
        else:
            scale_array = np.array(scale)
            print(f"  Non-uniform scaling: x={scale_array[0]}, y={scale_array[1]}, z={scale_array[2]}")

        mesh_copy.apply_scale(scale_array)
        print("  ✓ Scaling applied")

    # APPLY ROTATION
    if rotation is not None:
        print("Applying rotation transformation...")

        # Check if rotation is Euler angles [roll, pitch, yaw] or rotation matrix
        if len(rotation) == 3:  # Euler angles in radians
            roll, pitch, yaw = rotation
            print(f"  Euler angles (radians): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
            print(
                f"  Euler angles (degrees): roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")

            # Convert Euler angles to rotation matrices
            # Roll (rotation around X-axis)
            rx = np.array([[1, 0, 0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])

            # Pitch (rotation around Y-axis)
            ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                           [0, 1, 0],
                           [-np.sin(pitch), 0, np.cos(pitch)]])

            # Yaw (rotation around Z-axis)
            rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0],
                           [0, 0, 1]])

            # Combine rotations: Rz * Ry * Rx
            rotation_matrix = rz @ ry @ rx
        else:
            # Assume it's already a rotation matrix
            rotation_matrix = rotation
            print("  Using provided rotation matrix")

        # Create 4x4 transformation matrix for rotation
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix

        mesh_copy.apply_transform(transform_matrix)
        print("  ✓ Rotation applied")

    # APPLY TRANSLATION
    if translation is not None:
        translation_array = np.array(translation)
        print(f"Applying translation: x={translation_array[0]}, y={translation_array[1]}, z={translation_array[2]}")

        # Create 4x4 transformation matrix for translation
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation_array

        mesh_copy.apply_transform(translation_matrix)
        print("  ✓ Translation applied")

    # Show final bounds after all transformations
    final_bounds = mesh_copy.bounds
    print(f"Final bounds: {final_bounds}")

    return mesh_copy


# ============================================================================
# STEP 6: FEATURE EXTRACTION
# ============================================================================

def extract_features(mesh):
    """
    Extract important geometric features and properties from the mesh.

    These features are useful for:
    - Manufacturing cost estimation
    - Material usage calculations
    - Design optimization
    - Comparative analysis between designs

    Args:
        mesh (trimesh.Trimesh): Input mesh to analyze

    Returns:
        dict: Dictionary containing extracted geometric features:
            - surface_area: Total surface area (affects machining time)
            - volume: Internal volume (affects material removal)
            - bounding_box_volume: Volume of bounding box (stock material size)
            - convex_hull_volume: Volume of convex hull (tightest convex wrapper)
            - compactness: How compact the shape is (0-1, higher = more compact)
            - aspect_ratio: Ratio of longest to shortest dimension
    """

    print("\n" + "=" * 50)
    print("GEOMETRIC FEATURE EXTRACTION")
    print("=" * 50)

    # Calculate bounding box dimensions
    bbox_extents = mesh.bounding_box.extents  # [width, height, depth]

    # Calculate features
    features = {
        # Surface and volume properties
        "surface_area": mesh.area,  # Total surface area
        "volume": mesh.volume,  # Internal volume
        "bounding_box_volume": np.prod(bbox_extents),  # Volume of bounding box
        "convex_hull_volume": mesh.convex_hull.volume,  # Volume of convex hull

        # Shape characteristics
        "compactness": mesh.volume / mesh.convex_hull.volume if mesh.volume > 0 else 0,  # How compact (0-1)
        "aspect_ratio": np.max(bbox_extents) / np.min(bbox_extents) if np.min(bbox_extents) > 0 else 0,
        # Length/width ratio

        # Bounding box dimensions
        "bbox_width": bbox_extents[0],
        "bbox_height": bbox_extents[1],
        "bbox_depth": bbox_extents[2],
    }

    # Display extracted features with interpretations
    print("Extracted Geometric Features:")
    print(f"  Surface Area: {features['surface_area']:.2f} units² (affects machining time)")
    print(f"  Volume: {features['volume']:.2f} units³ (material to remove)")
    print(
        f"  Bounding Box: {features['bbox_width']:.1f} × {features['bbox_height']:.1f} × {features['bbox_depth']:.1f} units")
    print(f"  Bounding Box Volume: {features['bounding_box_volume']:.2f} units³ (stock material size)")
    print(f"  Convex Hull Volume: {features['convex_hull_volume']:.2f} units³")
    print(f"  Compactness: {features['compactness']:.3f} (1.0 = perfect convex shape)")
    print(f"  Aspect Ratio: {features['aspect_ratio']:.2f} (length/width ratio)")

    # Add interpretations for key metrics
    if features['compactness'] < 0.5:
        print("    → Shape has significant concave features or holes")
    elif features['compactness'] > 0.9:
        print("    → Shape is very compact and convex")

    if features['aspect_ratio'] > 5:
        print("    → Very elongated shape - may need special workholding")
    elif features['aspect_ratio'] < 1.5:
        print("    → Compact, balanced proportions")

    return features


# ============================================================================
# STEP 7: MANUFACTURABILITY ANALYSIS
# ============================================================================

def analyze_manufacturability(mesh):
    """
    Analyze the mesh for manufacturability using subtractive manufacturing processes.

    This function evaluates whether the part can be manufactured using processes like:
    - 3-axis CNC milling
    - 5-axis CNC machining
    - Conventional milling
    - Drilling operations

    The analysis considers factors like:
    - Tool accessibility
    - Undercuts and overhangs
    - Deep pockets and narrow features
    - Surface finish requirements

    Args:
        mesh (trimesh.Trimesh): Input mesh to analyze for manufacturability

    Returns:
        ManufacturabilityResult: Detailed analysis results including:
            - is_manufacturable: Boolean indicating if part can be manufactured
            - difficulty_score: Numerical score (0.0 = easy, 1.0 = very difficult)
            - issues: List of specific manufacturing challenges found
            - recommended_processes: Suitable manufacturing processes
            - minimum_tool_diameter: Smallest tool required
    """

    print("\n" + "=" * 50)
    print("MANUFACTURABILITY ANALYSIS")
    print("=" * 50)

    # Create analyzer instance and perform analysis
    analyzer = ManufacturabilityAnalyzer(mesh)
    result = analyzer.analyze_manufacturability()

    # Display overall manufacturability assessment
    status_symbol = "✓" if result.is_manufacturable else "✗"
    status_text = "MANUFACTURABLE" if result.is_manufacturable else "NOT MANUFACTURABLE"
    print(f"Overall Assessment: {status_symbol} {status_text}")

    # Display difficulty score with interpretation
    difficulty = result.difficulty_score
    print(f"Difficulty Score: {difficulty:.2f}/1.0", end="")

    if difficulty < 0.3:
        print(" (Easy - suitable for basic machining)")
    elif difficulty < 0.6:
        print(" (Moderate - may require advanced tooling)")
    elif difficulty < 0.8:
        print(" (Difficult - requires expert machining)")
    else:
        print(" (Very Difficult - may need specialized processes)")

    # Display specific issues found
    if result.issues:
        print(f"\nManufacturing Issues Identified ({len(result.issues)}):")
        for i, issue in enumerate(result.issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ No significant manufacturing issues found")

    # Display minimum tool requirements
    print(f"\nTooling Requirements:")
    print(f"  Minimum Tool Diameter: {result.minimum_tool_diameter:.2f} mm")

    if result.minimum_tool_diameter <= 1.0:
        print("    → Requires micro-machining capabilities")
    elif result.minimum_tool_diameter <= 3.0:
        print("    → Standard small tooling sufficient")
    else:
        print("    → Can use larger, more rigid tooling")

    # Display recommended manufacturing processes
    print(f"\nRecommended Manufacturing Processes ({len(result.recommended_processes)}):")
    for i, process in enumerate(result.recommended_processes, 1):
        print(f"  {i}. {process.value}")

        # Add brief explanations for each process
        if "3-axis" in process.value.lower():
            print("     → Standard milling with X, Y, Z movement")
        elif "5-axis" in process.value.lower():
            print("     → Advanced machining with rotational axes for complex features")
        elif "milling" in process.value.lower():
            print("     → Conventional milling operations")

    # Display additional analysis details if available
    if hasattr(result, 'deep_pocket_regions') and result.deep_pocket_regions:
        print(f"\nDetailed Analysis:")
        print(f"  Deep Pockets Found: {len(result.deep_pocket_regions)}")

    if hasattr(result, 'undercut_regions') and result.undercut_regions:
        print(f"  Undercut Regions: {len(result.undercut_regions)}")

    if hasattr(result, 'steep_angles') and result.steep_angles:
        print(f"  Steep Wall Regions: {len(result.steep_angles)}")

    return result


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_mesh(mesh):
    """
    Display the mesh using trimesh's built-in 3D viewer.

    This opens an interactive 3D window where you can:
    - Rotate and zoom the model
    - Inspect surface details
    - Verify mesh quality visually

    The viewer is useful for quick visual inspection and verification
    that the mesh loaded correctly.

    Args:
        mesh (trimesh.Trimesh): Mesh to visualize in 3D viewer
    """

    print("\n" + "=" * 50)
    print("3D MESH VIEWER")
    print("=" * 50)
    print("Opening interactive 3D viewer...")
    print("Controls:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Pan view")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Close window when finished")
    print("=" * 50)

    # Open the interactive 3D viewer window
    # This will block execution until the window is closed
    mesh.show()


def visualize_integrated(mesh, manufacturability_result=None, save_path=None):
    """
    Create an integrated visualization showing both 3D model and analysis report.

    This function generates a comprehensive view with:
    - Left side: 3D rendered model with proper lighting and materials
    - Right side: Detailed manufacturability analysis report

    The visualization is perfect for:
    - Design reviews and presentations
    - Documentation and reporting
    - Sharing analysis results with stakeholders

    Args:
        mesh (trimesh.Trimesh): Mesh to visualize
        manufacturability_result: Results from manufacturability analysis
        save_path (str, optional): Path to save the visualization image

    Returns:
        matplotlib.figure.Figure: The generated figure object
    """

    print("\n" + "=" * 50)
    print("INTEGRATED VISUALIZATION")
    print("=" * 50)

    if mesh is None:
        print("✗ Error: No mesh provided for visualization")
        return None

    print("Generating integrated plot with 3D model and analysis report...")

    # Use the enhanced visualization which provides both 3D view and detailed report
    fig = EnhancedMeshVisualization.visualize_with_report(
        mesh,
        manufacturability_result,
        save_path=save_path,
        show_plot=True  # Display the plot window
    )

    if save_path:
        print(f"✓ Visualization saved to: {save_path}")

    print("✓ Integrated visualization complete")
    return fig


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution function that runs the complete STL processing pipeline.

    This function orchestrates all processing steps in the correct order:
    1. Load STL file
    2. Analyze mesh quality
    3. Clean and repair mesh
    4. Simplify if needed
    5. Apply transformations (examples)
    6. Extract geometric features
    7. Analyze manufacturability
    8. Generate visualizations

    Modify the parameters and file paths as needed for your specific use case.
    """

    print("=" * 60)
    print("STL PROCESSING AND MANUFACTURABILITY ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"Processing file: {file_path}")

    # ========================================================================
    # STEP 1: LOAD STL FILE
    # ========================================================================
    mesh = load_stl(file_path)
    if mesh is None:
        print("✗ Critical Error: Failed to load mesh. Cannot continue.")
        print("  Please check:")
        print("  - File path is correct")
        print("  - File exists and is readable")
        print("  - File is a valid STL format")
        print("  - File is not corrupted")
        return

    # ========================================================================
    # STEP 2: ANALYZE MESH QUALITY
    # ========================================================================
    analysis_results = analyze_mesh(mesh)

    # Check for critical issues that would prevent further processing
    if not analysis_results["is_watertight"]:
        print("⚠ Warning: Mesh is not watertight - volume calculations may be inaccurate")

    if analysis_results["has_duplicated_faces"]:
        print("⚠ Warning: Duplicate faces detected - cleaning recommended")

    # ========================================================================
    # STEP 3: CLEAN AND REPAIR MESH
    # ========================================================================
    cleaned_mesh = clean_mesh(mesh, repair_watertight=True, remove_duplicates=True)

    # Use cleaned mesh for further processing
    working_mesh = cleaned_mesh

    # ========================================================================
    # STEP 4: MESH SIMPLIFICATION (OPTIONAL)
    # ========================================================================
    # Simplify mesh if it has too many faces (adjust threshold as needed)
    face_count = len(working_mesh.faces)

    if face_count > 10000:  # Threshold for simplification
        print(f"\nMesh has {face_count:,} faces - applying simplification...")
        simplified_mesh = simplify_mesh(working_mesh, target_percent=0.5)
        working_mesh = simplified_mesh
    else:
        print(f"\nMesh has {face_count:,} faces - no simplification needed")

    # ========================================================================
    # STEP 5: MESH TRANSFORMATIONS (EXAMPLES)
    # ========================================================================
    # These are examples - modify or remove based on your needs

    # Example 1: Scale the mesh (e.g., convert units or resize)
    # scaled_mesh = transform_mesh(working_mesh, scale=2.0)

    # Example 2: Rotate the mesh (e.g., correct orientation)
    # rotated_mesh = transform_mesh(working_mesh, rotation=[0, np.pi/4, 0])  # 45° rotation around Y-axis

    # Example 3: Translate the mesh (e.g., position at origin)
    # translated_mesh = transform_mesh(working_mesh, translation=[10, 0, 0])

    # For this example, we'll continue with the current working_mesh
    # working_mesh = scaled_mesh  # Uncomment to use transformed version

    # ========================================================================
    # STEP 6: EXTRACT GEOMETRIC FEATURES
    # ========================================================================
    features = extract_features(working_mesh)

    # ========================================================================
    # STEP 7: MANUFACTURABILITY ANALYSIS
    # ========================================================================
    manufacturability_result = analyze_manufacturability(working_mesh)

    # ========================================================================
    # STEP 8: VISUALIZATION AND REPORTING
    # ========================================================================

    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    # Option 1: Integrated matplotlib visualization with side-by-side 3D view and report
    print("1. Creating integrated plot visualization...")
    try:
        visualize_integrated(
            working_mesh,
            manufacturability_result,
            save_path="manufacturing_analysis.png"
        )
        print("   ✓ Integrated visualization complete")
    except Exception as e:
        print(f"   ✗ Integrated visualization failed: {e}")

    # Option 2: Interactive 3D viewer for detailed inspection
    print("\n2. Opening interactive 3D viewer...")
    print("   (Close the 3D viewer window to continue)")
    try:
        visualize_mesh(working_mesh)
        print("   ✓ 3D viewer session complete")
    except Exception as e:
        print(f"   ✗ 3D viewer failed: {e}")

    # ========================================================================
    # PROCESSING COMPLETE - SUMMARY
    # ========================================================================

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE - SUMMARY")
    print("=" * 60)

    # Summary of key results
    print(f"Original mesh: {len(mesh.faces):,} faces, {len(mesh.vertices):,} vertices")
    print(f"Final mesh: {len(working_mesh.faces):,} faces, {len(working_mesh.vertices):,} vertices")
    print(f"Volume: {features['volume']:.2f} cubic units")
    print(f"Surface area: {features['surface_area']:.2f} square units")
    print(f"Bounding box: {features['bbox_width']:.1f} × {features['bbox_height']:.1f} × {features['bbox_depth']:.1f}")

    # Manufacturability summary
    status = "✓ MANUFACTURABLE" if manufacturability_result.is_manufacturable else "✗ NOT MANUFACTURABLE"
    print(f"\nManufacturability: {status}")
    print(f"Difficulty score: {manufacturability_result.difficulty_score:.2f}/1.0")
    print(f"Minimum tool diameter: {manufacturability_result.minimum_tool_diameter:.2f} mm")

    if manufacturability_result.issues:
        print(f"Issues found: {len(manufacturability_result.issues)}")
    else:
        print("No significant issues found")

    # Files generated
    print(f"\nFiles generated:")
    print(f"  - manufacturing_analysis.png (integrated visualization)")

    print("\n" + "=" * 60)
    print("Analysis complete! Check the generated files and review results above.")
    print("=" * 60)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Script entry point - runs when file is executed directly.

    To use this script:
    1. Install required dependencies: pip install trimesh numpy matplotlib
    2. Place your STL file in the same directory as this script
    3. Update the 'file_path' variable to point to your STL file
    4. Run the script: python stl_processor.py

    The script will process the STL file through all steps and generate
    visualizations and analysis reports.
    """

    # Verify that required modules can be imported
    try:
        # Test imports
        import matplotlib.pyplot as plt
        from manufacturability_analyzer import ManufacturabilityAnalyzer
        from mesh_visualization import EnhancedMeshVisualization

        print("✓ All required modules imported successfully")

        # Run the main processing pipeline
        main()

    except ImportError as e:
        print(f"✗ Import Error: {e}")
        print("\nRequired dependencies:")
        print("  - trimesh: pip install trimesh")
        print("  - numpy: pip install numpy")
        print("  - matplotlib: pip install matplotlib")
        print("  - Custom modules: manufacturability_analyzer.py, mesh_visualization.py")
        print("\nPlease install missing dependencies and ensure custom modules are available.")

    except Exception as e:
        print(f"✗ Unexpected error during execution: {e}")
        print("Please check your STL file path and ensure all dependencies are properly installed.")
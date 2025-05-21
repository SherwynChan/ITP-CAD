# STL file preprocessing
# Step 1 - Load STL file
# Step 2 - Mesh analysis
# Step 3 - Mesh cleaning and repair
# Step 4 - mesh simplification 
# Step 5 - mesh tranformation
# Step 6 - feature extraction
# Step 7 - rendering
# to auto generate requirements.txt pip install pipreqs

import numpy as np
import trimesh
import os

file_path = os.path.join(os.path.dirname(__file__), 'test.stl')
# Step 1 - Load STL file
def load_stl(file_path):
    """
    Load an STL file and return a trimesh object
    
    Args:
        file_path (str): Path to the STL file
        
    Returns:
        trimesh.Trimesh: Loaded mesh object
    """
    try:
        mesh = trimesh.load(file_path)
        print(f"Successfully loaded mesh with {len(mesh.faces)} faces and {len(mesh.vertices)} vertices")
        return mesh
    except Exception as e:
        print(f"Error loading STL file: {e}")
        return None
    
# Step 2 - Mesh analysis
def analyze_mesh(mesh):
    """
    Analyze and validate a mesh
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        
    Returns:
        dict: Dictionary with mesh analysis results
    """
    # Check for duplicate faces by comparing face indices
    faces_view = mesh.faces.view(np.ndarray)
    unique_faces = np.unique(faces_view, axis=0)
    has_duplicated_faces = len(unique_faces) < len(mesh.faces)

    results = {
        "is_watertight": mesh.is_watertight,
        "is_winding_consistent": mesh.is_winding_consistent,
        "has_duplicated_faces": has_duplicated_faces,
        "volume": mesh.volume,
        "bounds": mesh.bounds,
        "centroid": mesh.centroid,
        "is_convex": mesh.is_convex
    }
    
    print("Mesh analysis results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    return results
# 3. Mesh cleaning and repair
def clean_mesh(mesh, repair_watertight=True, remove_duplicates=True):
    """
    Clean and repair mesh issues
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        repair_watertight (bool): Whether to attempt to make the mesh watertight
        remove_duplicates (bool): Whether to remove duplicate faces
        
    Returns:
        trimesh.Trimesh: Cleaned mesh
    """
    mesh_copy = mesh.copy()
    
    # Track changes
    changes = {}
    
    # Remove duplicate faces
    if remove_duplicates:
        # Find unique faces
        faces_view = mesh_copy.faces.view(np.ndarray)
        unique_faces, unique_indices = np.unique(faces_view, axis=0, return_index=True)
        
        # Check if we found duplicates
        has_duplicates = len(unique_indices) < len(mesh_copy.faces)
        
        if has_duplicates:
            original_faces = len(mesh_copy.faces)
            # Keep only the unique faces
            mesh_copy.faces = mesh_copy.faces[unique_indices]
            changes["duplicate_faces_removed"] = original_faces - len(mesh_copy.faces)
        else:
            changes["duplicate_faces_removed"] = 0
    
    # Fill holes to make watertight if requested
    if repair_watertight and not mesh.is_watertight:
        try:
            # First, fix normals
            mesh_copy.fix_normals()
            changes["normals_fixed"] = True
            
            # Fill holes - note that this might not always work
            # as trimesh has limited hole-filling capabilities
            try:
                # Try using trimesh's built-in method
                mesh_copy.fill_holes()
                changes["holes_filled"] = mesh_copy.is_watertight
            except Exception as e:
                # If built-in method fails, we'll just note it
                changes["holes_filled"] = False
                changes["hole_filling_error"] = str(e)
        except Exception as e:
            changes["repair_failed"] = True
            changes["repair_error"] = str(e)
    
    print("Mesh cleaning results:")
    for key, value in changes.items():
        print(f"  {key}: {value}")
    
    return mesh_copy
# 4. Mesh simplification
def simplify_mesh(mesh, target_percent=0.5):
    """
    Simplify a mesh by reducing the number of polygons (faces) 
    in a 3D model while preserving its overall shape and key features 
    as much as possible
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        target_percent (float): Target percentage of faces to keep (0.0-1.0)
        
    Returns:
        trimesh.Trimesh: Simplified mesh
    """
    if target_percent >= 1.0:
        return mesh.copy()
    
    original_faces = len(mesh.faces)
     # Calculate target_reduction (opposite of target_percent)
    # If target_percent = 0.5 (keep 50% of faces), then target_reduction = 0.5 (reduce by 50%)
    target_reduction = 1.0 - target_percent
  
    try:
        # Use trimesh's simplification with target_reduction parameter
        simplified = mesh.simplify_quadric_decimation(target_reduction)
        
        print(f"Simplified mesh from {original_faces} to {len(simplified.faces)} faces " 
              f"({len(simplified.faces)/original_faces:.2%} of original)")
        
        return simplified
    except Exception as e:
        print(f"Simplification failed: {e}")
        return mesh.copy()

# 5. Mesh transformation
def transform_mesh(mesh, scale=None, rotation=None, translation=None):
    """
    Apply transformations to a mesh
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        scale (float or np.ndarray): Scale factor(s)
        rotation (np.ndarray): Rotation matrix or [roll, pitch, yaw] in radians
        translation (np.ndarray): Translation vector [x, y, z]
        
    Returns:
        trimesh.Trimesh: Transformed mesh
    """
    mesh_copy = mesh.copy()
    
    # Apply scaling
    if scale is not None:
        if isinstance(scale, (int, float)):
            scale = np.array([scale, scale, scale])
        mesh_copy.apply_scale(scale)
        print(f"Applied scaling: {scale}")
    
    # Apply rotation
    if rotation is not None:
        if len(rotation) == 3:  # [roll, pitch, yaw]
            # Convert Euler angles to rotation matrix
            rx = np.array([[1, 0, 0], 
                           [0, np.cos(rotation[0]), -np.sin(rotation[0])], 
                           [0, np.sin(rotation[0]), np.cos(rotation[0])]])
            
            ry = np.array([[np.cos(rotation[1]), 0, np.sin(rotation[1])], 
                           [0, 1, 0], 
                           [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])
            
            rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]), 0], 
                           [np.sin(rotation[2]), np.cos(rotation[2]), 0], 
                           [0, 0, 1]])
            
            rotation_matrix = rx @ ry @ rz
        else:
            rotation_matrix = rotation
            
        mesh_copy.apply_transform(
            np.vstack([
                np.hstack([rotation_matrix, np.zeros((3, 1))]),
                [0, 0, 0, 1]
            ])
        )
        print("Applied rotation")
    
    # Apply translation
    if translation is not None:
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation
        mesh_copy.apply_transform(translation_matrix)
        print(f"Applied translation: {translation}")
    
    return mesh_copy

# 6. Feature extraction
def extract_features(mesh):
    """
    Extract geometric features from a mesh
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        
    Returns:
        dict: Dictionary with extracted features
    """
    features = {
        "surface_area": mesh.area,
        "volume": mesh.volume,
        "bounding_box_volume": np.prod(mesh.bounding_box.extents),
        "convex_hull_volume": mesh.convex_hull.volume,
        "compactness": mesh.volume / mesh.convex_hull.volume if mesh.volume > 0 else 0,
        "aspect_ratio": np.ptp(mesh.bounding_box.extents) / np.min(mesh.bounding_box.extents)
    }
    
    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    return features
def visualize_mesh(mesh):
    """
    Visualize the mesh using trimesh's built-in viewer
    
    Args:
        mesh (trimesh.Trimesh): Mesh to visualize
    """
    # This will open a window with the 3D model
    mesh.show()

mesh = load_stl(file_path)
analysis_results = analyze_mesh(mesh)

# Test cleaning
cleaned_mesh = clean_mesh(mesh)

# Test simplification
simplified_mesh = simplify_mesh(cleaned_mesh, target_percent=0.5)

# Test transformation
# Scale by 2x
scaled_mesh = transform_mesh(simplified_mesh, scale=2.0)
# Rotate 45 degrees around y-axis
rotated_mesh = transform_mesh(simplified_mesh, rotation=[0, np.pi/4, 0])
# Move 10 units along x-axis
translated_mesh = transform_mesh(simplified_mesh, translation=[10, 0, 0])

# Test feature extraction
features = extract_features(simplified_mesh)

visualize_mesh(mesh)
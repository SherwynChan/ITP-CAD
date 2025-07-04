import trimesh
import os
from typing import Optional, Dict, Any

def find_internal_volumes(stl_path: str) -> Optional[Dict[str, Any]]:
    """
    Analyze an STL file to detect internal volumes and cavities.

    Args:
        stl_path (str): Path to the STL file

    Returns:
        Dict containing analysis results or None if analysis failed
    """

    # Validate input file
    if not os.path.exists(stl_path):
        print(f"❌ File not found: {stl_path}")
        return None

    try:
        # Load the mesh
        mesh = trimesh.load(stl_path)
        print(f"Loaded mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")

        # Check if watertight (required for internal volume detection)
        print(f"Is watertight: {mesh.is_watertight}")
        if not mesh.is_watertight:
            print("❌ Mesh is not watertight - cannot detect internal volumes reliably")
            print("This is common with STL files that have gaps or holes")
            return {
                'watertight': False,
                'error': 'Mesh is not watertight'
            }

        # Calculate volumes
        actual_volume = mesh.volume
        convex_volume = mesh.convex_hull.volume

        print(f"Actual volume: {actual_volume:.3f}")
        print(f"Convex hull volume: {convex_volume:.3f}")

        if actual_volume > 0 and convex_volume > 0:
            volume_ratio = actual_volume / convex_volume
            print(f"Volume ratio (actual/convex): {volume_ratio:.3f}")

            print(f"\nInterpretation:")
            if volume_ratio > 0.9:
                status = "solid"
                print(f"✓ Solid object (ratio > 0.9)")
            elif volume_ratio > 0.6:
                status = "minor_cavities"
                print(f"⚠ Minor internal spaces (ratio 0.6-0.9)")
            else:
                status = "significant_cavities"
                print(f"🕳 Significant internal volumes (ratio < 0.6)")
                print(f"   This suggests the object is hollow inside!")

            return {
                'watertight': True,
                'actual_volume': actual_volume,
                'convex_volume': convex_volume,
                'volume_ratio': volume_ratio,
                'status': status,
                'faces': len(mesh.faces),
                'vertices': len(mesh.vertices)
            }
        else:
            print("❌ Invalid volume calculations")
            return {
                'watertight': True,
                'error': 'Invalid volume calculations'
            }

    except Exception as e:
        print(f"❌ Error analyzing mesh: {e}")
        return {
            'error': str(e)
        }


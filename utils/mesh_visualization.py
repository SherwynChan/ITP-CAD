# Mesh Visualization Module
# =========================
# This module provides enhanced 3D mesh visualization capabilities with integrated
# manufacturability analysis reporting. It creates professional-quality plots
# that combine 3D model rendering with detailed text-based analysis reports.
#
# Key Features:
# - Side-by-side 3D mesh visualization and analysis report
# - Customizable 3D rendering with proper materials and lighting
# - Detailed manufacturability analysis display
# - High-quality output suitable for documentation and presentations

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


class EnhancedMeshVisualization:
    """
    Enhanced mesh visualization class providing professional-quality 3D rendering
    with integrated manufacturability analysis reporting.

    This class creates comprehensive visualizations that are perfect for:
    - Design review meetings
    - Manufacturing feasibility presentations  
    - Technical documentation
    - Quality assurance reports
    """

    @staticmethod
    def visualize_with_report(mesh, manufacturability_result=None, save_path=None, show_plot=True):
        """
        Create a comprehensive visualization combining 3D mesh rendering with detailed analysis report.

        This is the main visualization function that creates a professional two-panel layout:
        - Left panel: High-quality 3D mesh rendering with proper lighting and materials
        - Right panel: Detailed manufacturability analysis report with all key metrics

        The visualization is designed to provide both visual inspection capabilities
        and comprehensive technical information in a single, easy-to-understand format.

        Args:
            mesh (trimesh.Trimesh): The 3D mesh object to visualize
            manufacturability_result (ManufacturabilityResult, optional): Analysis results to display
            save_path (str, optional): File path to save the visualization (PNG format)
            show_plot (bool): Whether to display the plot in a window (default: True)

        Returns:
            matplotlib.figure.Figure: The complete figure object containing both panels
        """

        # Create main figure with professional layout
        # Using 16:10 aspect ratio for better screen/document compatibility
        fig = plt.figure(figsize=(16, 10))

        # Set up subplot layout: 1 row, 2 columns with different widths
        # Left subplot (3D mesh): 60% of width for detailed model viewing
        # Right subplot (text report): 40% of width for comprehensive text display
        ax_3d = fig.add_subplot(121, projection='3d')  # Left: 3D mesh visualization
        ax_text = fig.add_subplot(122)  # Right: Text-based analysis report

        # Remove axes and formatting from text panel for clean appearance
        ax_text.axis('off')

        # Render the 3D mesh with enhanced visual quality
        print("  Rendering 3D mesh...")
        EnhancedMeshVisualization._render_mesh_3d(ax_3d, mesh)

        # Add comprehensive text report if analysis results are available
        if manufacturability_result:
            print("  Adding manufacturability analysis report...")
            EnhancedMeshVisualization._add_text_report(ax_text, manufacturability_result, mesh)
        else:
            # If no analysis results, show basic mesh information
            print("  Adding basic mesh information...")
            EnhancedMeshVisualization._add_basic_mesh_info(ax_text, mesh)

        # Apply tight layout to optimize spacing and prevent overlap
        plt.tight_layout()

        # Save visualization to file if path provided
        if save_path:
            print(f"  Saving visualization to {save_path}...")
            # Use high DPI for crisp images and tight bounding box for clean edges
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"  ✓ Visualization saved successfully")

        # Display the plot window if requested
        if show_plot:
            print("  Displaying interactive plot window...")
            plt.show()

        return fig

    @staticmethod
    def _render_mesh_3d(ax, mesh):
        """
        Render the 3D mesh with enhanced visual quality and professional appearance.

        This function creates a high-quality 3D rendering with:
        - Proper face coloring and transparency for depth perception
        - Edge highlighting for better geometry definition
        - Optimal viewing angle and lighting
        - Professional axis labeling and scaling

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): 3D matplotlib axes object
            mesh (trimesh.Trimesh): The mesh object containing vertices and faces
        """

        # Extract mesh geometry data
        vertices = mesh.vertices  # 3D coordinate points
        faces = mesh.faces  # Triangle face definitions (indices into vertices)

        # Create 3D polygon collection with professional styling
        # Using light blue faces with navy edges for good contrast and visibility
        mesh_3d = Poly3DCollection(
            vertices[faces],  # Face geometry
            alpha=0.7,  # Semi-transparent for depth perception
            facecolor='lightblue',  # Light blue surface color
            edgecolor='navy',  # Dark blue edge lines
            linewidth=0.1  # Thin edge lines to avoid visual clutter
        )

        # Add the mesh to the 3D axes
        ax.add_collection3d(mesh_3d)

        # Calculate and set appropriate axis limits based on mesh bounds
        # Add small padding (5%) to prevent mesh from touching axis boundaries
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

        # Add 5% padding to each dimension for better visual presentation
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        z_padding = (z_max - z_min) * 0.05

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.set_zlim(z_min - z_padding, z_max + z_padding)

        # Set professional axis labels and title
        ax.set_xlabel('X Coordinate', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z Coordinate', fontsize=10, fontweight='bold')
        ax.set_title('3D CAD Model', fontsize=14, fontweight='bold', pad=20)

        # Set optimal viewing angle for most parts
        # Elevation: 20° provides good depth perception without too much distortion
        # Azimuth: 45° shows three faces of typical rectangular parts
        ax.view_init(elev=20, azim=45)

        # Enable grid for better spatial reference
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _add_text_report(ax, result, mesh):
        """
        Add a comprehensive manufacturability analysis report to the visualization.

        This function creates a detailed, well-formatted text report that includes:
        - Overall manufacturability assessment with clear pass/fail indication
        - Part specifications and geometric properties
        - Manufacturing constraints and limitations
        - Recommended processes and tooling requirements
        - Specific issues found and recommendations

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): Text subplot axes
            result (ManufacturabilityResult): Analysis results from manufacturability analyzer
            mesh (trimesh.Trimesh): Original mesh object for basic specifications
        """

        # Clear and configure the text axes
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')  # Remove axes for clean text display

        # Build comprehensive report content with professional formatting
        # Using consistent spacing and clear section headers
        report_text = f"""MANUFACTURABILITY ANALYSIS REPORT
{'=' * 45}

OVERALL ASSESSMENT:
Status: {'✓ MANUFACTURABLE' if result.is_manufacturable else '✗ NOT MANUFACTURABLE'}
Difficulty Score: {result.difficulty_score:.2f}/1.0"""

        # Add difficulty interpretation for better understanding
        if result.difficulty_score < 0.3:
            report_text += " (Easy)"
        elif result.difficulty_score < 0.6:
            report_text += " (Moderate)"
        elif result.difficulty_score < 0.8:
            report_text += " (Difficult)"
        else:
            report_text += " (Very Difficult)"

        # Part specifications section with key geometric properties
        report_text += f"""

PART SPECIFICATIONS:
Volume: {mesh.volume:.2f} mm³
Surface Area: {mesh.area:.2f} mm²
Bounding Box: {mesh.bounding_box.extents[0]:.1f} × {mesh.bounding_box.extents[1]:.1f} × {mesh.bounding_box.extents[2]:.1f} mm
Watertight: {'Yes' if mesh.is_watertight else 'No'}"""

        # Manufacturing constraints section with technical details
        report_text += f"""

MANUFACTURING CONSTRAINTS:
Minimum Tool Diameter: {result.minimum_tool_diameter:.2f} mm"""

        # Add specific constraint details if available
        if hasattr(result, 'deep_pocket_regions'):
            report_text += f"""
Deep Pockets Found: {len(result.deep_pocket_regions)}"""

        if hasattr(result, 'undercut_regions'):
            report_text += f"""
Undercut Regions: {len(result.undercut_regions)}"""

        if hasattr(result, 'steep_angles'):
            report_text += f"""
Steep Wall Regions: {len(result.steep_angles)}"""

        # Recommended manufacturing processes
        report_text += f"""

RECOMMENDED PROCESSES:"""

        # List each recommended process with bullet points
        for process in result.recommended_processes:
            report_text += f"""
• {process.value}"""

        # Issues section - only display if issues were found
        if result.issues:
            report_text += f"""

ISSUES IDENTIFIED:"""
            for issue in result.issues:
                report_text += f"""
• {issue}"""

        # General recommendations section with practical advice
        report_text += f"""

MANUFACTURING RECOMMENDATIONS:
• Use appropriate tooling for minimum diameter requirements
• Consider 5-axis machining for complex features
• Plan tool paths carefully for deep pockets
• Ensure adequate workholding for steep walls
• Verify surface finish requirements before machining
• Consider material properties in tool selection"""

        # Add the formatted text to the plot with professional styling
        ax.text(
            0.05, 0.95,  # Position: top-left with small margin
            report_text,  # The complete report text
            fontsize=10,  # Readable font size
            fontfamily='monospace',  # Monospace font for consistent alignment
            verticalalignment='top',  # Align to top of text box
            transform=ax.transAxes,  # Use axes coordinates (0-1 range)
            bbox=dict(  # Background box styling
                boxstyle="round,pad=0.5",  # Rounded corners with padding
                facecolor="lightgray",  # Light gray background
                alpha=0.8  # Semi-transparent background
            )
        )

    @staticmethod
    def _add_basic_mesh_info(ax, mesh):
        """
        Add basic mesh information when no manufacturability analysis is available.

        This fallback function provides essential mesh properties and statistics
        when detailed manufacturability analysis hasn't been performed.

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): Text subplot axes
            mesh (trimesh.Trimesh): Mesh object to analyze
        """

        # Clear and configure the text axes
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Create basic mesh information report
        basic_info = f"""MESH INFORMATION REPORT
{'=' * 35}

GEOMETRY PROPERTIES:
Faces (Triangles): {len(mesh.faces):,}
Vertices (Points): {len(mesh.vertices):,}
Volume: {mesh.volume:.2f} mm³
Surface Area: {mesh.area:.2f} mm²

MESH QUALITY:
Watertight: {'Yes' if mesh.is_watertight else 'No'}
Winding Consistent: {'Yes' if mesh.is_winding_consistent else 'No'}
Convex Shape: {'Yes' if mesh.is_convex else 'No'}

DIMENSIONS:
Bounding Box: {mesh.bounding_box.extents[0]:.1f} × {mesh.bounding_box.extents[1]:.1f} × {mesh.bounding_box.extents[2]:.1f} mm
Centroid: ({mesh.centroid[0]:.1f}, {mesh.centroid[1]:.1f}, {mesh.centroid[2]:.1f})

NOTE:
For detailed manufacturability analysis,
run the analyze_manufacturability() function
and provide the results to this visualization."""

        # Add the basic information to the plot
        ax.text(
            0.05, 0.95,
            basic_info,
            fontsize=10,
            fontfamily='monospace',
            verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightyellow",  # Different color to indicate basic info
                alpha=0.8
            )
        )


# ============================================================================
# ENHANCED INTEGRATION FUNCTION
# ============================================================================

def enhanced_visualize_integrated(mesh, manufacturability_result=None, save_path=None):
    """
    Enhanced visualization function that serves as the main entry point for mesh visualization.

    This function provides a clean, simple interface to the EnhancedMeshVisualization class
    while maintaining backward compatibility with existing code.

    Key features:
    - Automatic error handling and validation
    - Consistent return values
    - Professional logging and status updates
    - Optimized for both interactive use and automated processing

    Args:
        mesh (trimesh.Trimesh or None): Mesh object to visualize
        manufacturability_result (ManufacturabilityResult, optional): Analysis results
        save_path (str, optional): Path to save the visualization image

    Returns:
        matplotlib.figure.Figure or None: Generated figure object, or None if visualization failed
    """

    # Validate input mesh
    if mesh is None:
        print("✗ Error: No mesh provided for visualization")
        print("  Please ensure the STL file was loaded successfully")
        return None

    # Validate mesh has required properties
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        print("✗ Error: Invalid mesh object - missing vertices or faces")
        return None

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print("✗ Error: Empty mesh - no geometry to visualize")
        return None

    print("Creating enhanced visualization...")

    try:
        # Use the enhanced visualization class to create the comprehensive plot
        fig = EnhancedMeshVisualization.visualize_with_report(
            mesh,
            manufacturability_result,
            save_path=save_path,
            show_plot=True
        )

        print("✓ Enhanced visualization completed successfully")
        return fig

    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        print("  Common causes:")
        print("  - Corrupted mesh geometry")
        print("  - Insufficient memory for large meshes")
        print("  - Display/graphics driver issues")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_mesh_screenshot(mesh, filename="mesh_screenshot.png", view_angle=(20, 45)):
    """
    Quickly save a simple screenshot of the mesh without analysis report.

    This utility function is useful for generating quick previews or thumbnails
    without the full analysis report layout.

    Args:
        mesh (trimesh.Trimesh): Mesh to capture
        filename (str): Output filename for the screenshot
        view_angle (tuple): (elevation, azimuth) viewing angles in degrees

    Returns:
        bool: True if screenshot saved successfully, False otherwise
    """

    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Simple mesh rendering
        vertices = mesh.vertices
        faces = mesh.faces

        mesh_3d = Poly3DCollection(vertices[faces], alpha=0.7,
                                   facecolor='lightblue', edgecolor='navy')
        ax.add_collection3d(mesh_3d)

        # Set limits and view
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        # Save with high quality
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory

        print(f"✓ Screenshot saved: {filename}")
        return True

    except Exception as e:
        print(f"✗ Failed to save screenshot: {e}")
        return False


def create_mesh_comparison(mesh_list, titles=None, save_path="mesh_comparison.png"):
    """
    Create a side-by-side comparison visualization of multiple meshes.

    Useful for comparing:
    - Original vs. simplified meshes
    - Different processing stages
    - Multiple design iterations

    Args:
        mesh_list (list): List of trimesh objects to compare
        titles (list, optional): List of titles for each mesh
        save_path (str): Path to save the comparison image

    Returns:
        matplotlib.figure.Figure: Comparison figure object
    """

    if not mesh_list:
        print("✗ Error: No meshes provided for comparison")
        return None

    num_meshes = len(mesh_list)
    if titles is None:
        titles = [f"Mesh {i + 1}" for i in range(num_meshes)]

    # Create figure with subplots
    fig = plt.figure(figsize=(5 * num_meshes, 8))

    for i, mesh in enumerate(mesh_list):
        ax = fig.add_subplot(1, num_meshes, i + 1, projection='3d')

        # Render each mesh
        vertices = mesh.vertices
        faces = mesh.faces

        mesh_3d = Poly3DCollection(vertices[faces], alpha=0.7,
                                   facecolor='lightblue', edgecolor='navy')
        ax.add_collection3d(mesh_3d)

        # Set limits and labels
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Mesh comparison saved: {save_path}")

    plt.show()
    return fig
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


class EnhancedMeshVisualization:
    @staticmethod
    def visualize_with_report(mesh, manufacturability_result=None, save_path=None, show_plot=True):
        """
        Create a comprehensive visualization with 3D mesh and detailed text report
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))

        # 3D mesh subplot (left side - larger)
        ax_3d = fig.add_subplot(121, projection='3d')

        # Text report subplot (right side)
        ax_text = fig.add_subplot(122)
        ax_text.axis('off')

        # Render 3D mesh
        EnhancedMeshVisualization._render_mesh_3d(ax_3d, mesh)

        # Add text report
        if manufacturability_result:
            EnhancedMeshVisualization._add_text_report(ax_text, manufacturability_result, mesh)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        if show_plot:
            plt.show()

        return fig

    @staticmethod
    def _render_mesh_3d(ax, mesh):
        """Render 3D mesh using matplotlib"""
        # Get mesh data
        vertices = mesh.vertices
        faces = mesh.faces

        # Create 3D collection
        mesh_3d = Poly3DCollection(vertices[faces], alpha=0.7, facecolor='lightblue', edgecolor='navy', linewidth=0.1)
        ax.add_collection3d(mesh_3d)

        # Set axis limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D CAD Model', fontsize=14, fontweight='bold')

        # Set viewing angle
        ax.view_init(elev=20, azim=45)

    @staticmethod
    def _add_text_report(ax, result, mesh):
        """Add detailed text report to the plot"""
        # Clear axis
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Report content
        report_text = f"""
MANUFACTURABILITY ANALYSIS REPORT
{'=' * 45}

OVERALL ASSESSMENT:
Status: {'✓ MANUFACTURABLE' if result.is_manufacturable else '✗ NOT MANUFACTURABLE'}
Difficulty Score: {result.difficulty_score:.2f}/1.0

PART SPECIFICATIONS:
Volume: {mesh.volume:.2f} mm³
Surface Area: {mesh.area:.2f} mm²
Bounding Box: {mesh.bounding_box.extents}

MANUFACTURING CONSTRAINTS:
Minimum Tool Diameter: {result.minimum_tool_diameter:.2f} mm
Deep Pockets Found: {len(result.deep_pocket_regions)}
Undercut Regions: {len(result.undercut_regions)}
Steep Wall Regions: {len(result.steep_angles)}

RECOMMENDED PROCESSES:
"""
        for process in result.recommended_processes:
            report_text += f"• {process.value}\n"

        if result.issues:
            report_text += f"\nISSUES IDENTIFIED:\n"
            for issue in result.issues:
                report_text += f"• {issue}\n"

        report_text += f"""
RECOMMENDATIONS:
• Use appropriate tooling for minimum diameter requirements
• Consider 5-axis machining for complex features
• Plan tool paths carefully for deep pockets
• Ensure adequate workholding for steep walls
"""

        # Add text to plot
        ax.text(0.05, 0.95, report_text, fontsize=10, fontfamily='monospace',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))


# Enhanced mesh processing with better visualization
def enhanced_visualize_integrated(mesh, manufacturability_result=None, save_path=None):
    """
    Enhanced visualization function to replace the original
    """
    if mesh is None:
        print("No mesh provided for visualization.")
        return None

    # Use the enhanced visualization
    fig = EnhancedMeshVisualization.visualize_with_report(
        mesh, manufacturability_result, save_path
    )

    return fig


# Alternative: HTML Report Generation
class HTMLReportGenerator:
    @staticmethod
    def generate_html_report(mesh, manufacturability_result, output_path="manufacturability_report.html"):
        """Generate an HTML report with embedded 3D viewer"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Manufacturability Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-pass {{ color: green; font-weight: bold; }}
        .status-fail {{ color: red; font-weight: bold; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }}
        .metric {{ margin: 10px 0; }}
        .issue {{ color: #d63384; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Manufacturability Analysis Report</h1>
        <h2 class="{'status-pass' if manufacturability_result.is_manufacturable else 'status-fail'}">
            {'✓ MANUFACTURABLE' if manufacturability_result.is_manufacturable else '✗ NOT MANUFACTURABLE'}
        </h2>
    </div>

    <div class="section">
        <h3>Part Specifications</h3>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Volume</td><td>{mesh.volume:.2f} mm³</td></tr>
            <tr><td>Surface Area</td><td>{mesh.area:.2f} mm²</td></tr>
            <tr><td>Bounding Box</td><td>{mesh.bounding_box.extents}</td></tr>
            <tr><td>Is Watertight</td><td>{'Yes' if mesh.is_watertight else 'No'}</td></tr>
        </table>
    </div>

    <div class="section">
        <h3>Manufacturing Assessment</h3>
        <div class="metric">Difficulty Score: {manufacturability_result.difficulty_score:.2f}/1.0</div>
        <div class="metric">Minimum Tool Diameter: {manufacturability_result.minimum_tool_diameter:.2f} mm</div>
        <div class="metric">Deep Pockets: {len(manufacturability_result.deep_pocket_regions)}</div>
        <div class="metric">Undercut Regions: {len(manufacturability_result.undercut_regions)}</div>
        <div class="metric">Steep Wall Regions: {len(manufacturability_result.steep_angles)}</div>
    </div>

    <div class="section">
        <h3>Recommended Manufacturing Processes</h3>
        <ul>
"""

        for process in manufacturability_result.recommended_processes:
            html_content += f"<li>{process.value}</li>"

        html_content += "</ul></div>"

        if manufacturability_result.issues:
            html_content += """
    <div class="section">
        <h3>Issues Identified</h3>
        <ul>
"""
            for issue in manufacturability_result.issues:
                html_content += f'<li class="issue">{issue}</li>'

            html_content += "</ul></div>"

        html_content += """
    <div class="section">
        <h3>Recommendations</h3>
        <ul>
            <li>Use appropriate tooling for minimum diameter requirements</li>
            <li>Consider 5-axis machining for complex features</li>
            <li>Plan tool paths carefully for deep pockets</li>
            <li>Ensure adequate workholding for steep walls</li>
        </ul>
    </div>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report generated: {output_path}")
        return output_path


# Usage example for your main script:
def updated_main():
    """Updated main function with better visualization"""
    # Your existing code...
    mesh = load_stl(file_path)
    if mesh is None:
        print("Failed to load mesh. Exiting.")
        exit(1)

    # ... other processing ...

    # Analyze manufacturability
    manufacturability_result = analyze_manufacturability(mesh)

    # Enhanced visualization with proper text report
    enhanced_visualize_integrated(mesh, manufacturability_result, save_path="manufacturing_analysis.png")

    # Alternative: Generate HTML report
    HTMLReportGenerator.generate_html_report(mesh, manufacturability_result)
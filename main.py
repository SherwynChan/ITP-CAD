import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.dxf_processor import process_dxf
from environment.path_planner import CADPathPlanner
from agents.systematic_agent import SystematicPathFinder
from utils.training import discover_cutting_path, optimize_cutting_sequence

def main():
    st.title("CAD Path Planning System")
    
    # Initialize session state
    if 'pathfinder' not in st.session_state:
        st.session_state.pathfinder = None
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'path_discovered' not in st.session_state:
        st.session_state.path_discovered = False
    
    uploaded_file = st.file_uploader("Choose a DXF file", type=['dxf'])
    
    if uploaded_file:
        target_shape = process_dxf(uploaded_file)
        
        if target_shape is not None:
            st.header("Initial State and Target Shape")
            
            # Create visualization showing initial filled block and target shape
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Create and show initial filled block
            initial_state = np.ones_like(target_shape)
            ax1.imshow(initial_state, cmap='binary_r', interpolation='nearest')
            ax1.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax1.set_title('Initial State\n(White = Material, Black = Background)', pad=20)
            ax1.set_xlabel('X coordinate')
            ax1.set_ylabel('Y coordinate')
            
            # Show target shape
            ax2.imshow(target_shape, cmap='binary', interpolation='nearest')
            ax2.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax2.set_title('Target Shape\n(White = Material, Black = Background)', pad=20)
            ax2.set_xlabel('X coordinate')
            
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.write(f"Grid size: {target_shape.shape[0]}x{target_shape.shape[1]}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Path Discovery")
                if st.button("Discover Cutting Path"):
                    st.session_state.env = CADPathPlanner(target_shape=target_shape)
                    st.session_state.pathfinder = SystematicPathFinder()
                    
                    with st.spinner("Discovering valid cutting path..."):
                        discover_cutting_path(st.session_state.env, st.session_state.pathfinder)
                        st.session_state.path_discovered = True
                        st.success("Path discovery completed!")
            
            with col2:
                st.header("Path Optimization")
                num_attempts = st.slider("Number of optimization attempts", min_value=5, max_value=20, value=10)
                if st.button("Optimize Sequence", disabled=not st.session_state.path_discovered):
                    if st.session_state.pathfinder is None or st.session_state.env is None:
                        st.error("Please discover cutting path first!")
                    else:
                        with st.spinner("Optimizing cutting sequence..."):
                            results = optimize_cutting_sequence(
                                st.session_state.env, 
                                st.session_state.pathfinder, 
                                num_attempts
                            )
                            if results and results['perfect_match']:
                                st.success(f"Perfect match achieved with efficiency {results['efficiency']:.3f}!")
                            elif results:
                                st.warning(f"Best partial match achieved with {results['final_similarity']:.1%} similarity")

if __name__ == "__main__":
    main()
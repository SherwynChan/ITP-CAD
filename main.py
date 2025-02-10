import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.dxf_processor import process_dxf
from environment.cad_environment import CADEnvironment
from agents.systematic_agent import SystematicCuttingAgent

def train_model(env, agent, n_episodes=1000):
    progress_bar = st.progress(0)
    
    for episode in range(n_episodes):
        state = env.reset()
        agent.reset()
        moves_this_episode = 0
        
        while True:
            action, position = agent.get_action(state, env.target_shape)
            
            if action == 0 and position is not None:
                next_state, reward, done = env.step(action, position)
                agent.learn(state, action, reward, next_state, position)
                state = next_state
                moves_this_episode += 1
                
                if moves_this_episode % 10 == 0:
                    st.write(f"Episode {episode + 1}, Move {moves_this_episode}")
                    st.write(f"Current Similarity: {env.best_similarity:.3f}")
                    fig = env.render()
                    st.pyplot(fig)
                    plt.close(fig)
                
                if done:
                    st.success(f"Target shape achieved at episode {episode + 1}!")
                    fig = env.render()
                    st.pyplot(fig)
                    plt.close(fig)
                    return
            else:
                break
        
        progress_bar.progress((episode + 1) / n_episodes)

def main():
    st.title("CAD Shape Learning System")
    
    uploaded_file = st.file_uploader("Choose a DXF file", type=['dxf'])
    
    if uploaded_file:
        target_shape = process_dxf(uploaded_file)
        
        if target_shape is not None:
            st.header("Initial State and Target Shape")
            
            initial_state = np.ones_like(target_shape)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.imshow(initial_state, cmap='binary_r', interpolation='nearest')
            ax1.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax1.set_title('Initial State\n(White = Material, Black = Background)', pad=20)
            ax1.set_xlabel('X coordinate')
            ax1.set_ylabel('Y coordinate')
            
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
            st.write("- White = Material")
            st.write("- Black = Background/Empty Space")
            
            env = CADEnvironment(target_shape=target_shape)
            agent = SystematicCuttingAgent()
            
            st.header("Training")
            
            if st.button("Train Model"):
                with st.spinner("Training in progress..."):
                    train_model(env, agent)
                    st.success("Training completed!")

if __name__ == "__main__":
    main()
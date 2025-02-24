import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def discover_cutting_path(env, agent, n_episodes=10):
    """Train the agent for systematic cutting."""
    progress_bar = st.progress(0)
    
    print("\n=== Starting Training ===")
    print("Initial state shape:", env.current_state.shape)
    print("Target shape:", env.target_shape.shape)
    
    best_similarity = 0
    best_path = None
    best_distance = float('inf')
    
    episode_metrics = []
    
    for episode in range(n_episodes):
        state = env.reset()
        agent.reset()
        
        moves_this_episode = 0
        episode_similarity = 0
        episode_reward = 0
        path = []
        
        while True:
            action, position = agent.get_action(state, env.target_shape)
            
            if action == 0 and position is not None:
                next_state, reward, done = env.step(position)
                agent.learn(state, action, reward, next_state, position)
                
                path.append(position)
                episode_reward += reward
                moves_this_episode += 1
                
                similarity = np.sum(env.current_state == env.target_shape) / (env.size * env.size)
                episode_similarity = max(episode_similarity, similarity)
                
                if moves_this_episode % 10 == 0:
                    st.write(f"Episode {episode + 1}, Move {moves_this_episode}")
                    st.write(f"Current Similarity: {similarity:.3f}")
                    fig = env.render()
                    st.pyplot(fig)
                    plt.close(fig)
                
                if done:
                    total_distance = 0
                    for i in range(1, len(path)):
                        y1, x1 = path[i-1]
                        y2, x2 = path[i]
                        total_distance += abs(y2 - y1) + abs(x2 - x1)
                    
                    if similarity >= best_similarity and total_distance < best_distance:
                        best_similarity = similarity
                        best_path = path.copy()
                        best_distance = total_distance
                    
                    st.success(f"Target shape achieved at episode {episode + 1}!")
                    st.write(f"Path efficiency: {total_distance/len(path):.2f}")
                    fig = env.render()
                    st.pyplot(fig)
                    plt.close(fig)
                    break
                
                state = next_state
            else:
                break
        
        episode_metrics.append({
            'episode': episode + 1,
            'moves': moves_this_episode,
            'similarity': episode_similarity,
            'reward': episode_reward
        })
        
        progress_bar.progress((episode + 1) / n_episodes)
    
    plot_training_metrics(episode_metrics)
    return best_path

def plot_training_metrics(episode_metrics):
    """Plot training metrics."""
    if episode_metrics:
        metrics_df = pd.DataFrame(episode_metrics)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        ax1.plot(metrics_df['episode'], metrics_df['similarity'])
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Best Similarity')
        ax1.set_title('Training Progress - Similarity')
        ax1.grid(True)
        
        ax2.plot(metrics_df['episode'], metrics_df['reward'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Training Progress - Rewards')
        ax2.grid(True)
        
        st.pyplot(fig)
        plt.close(fig)

def optimize_cutting_sequence(env, agent, num_attempts=10):
    """Test multiple cutting sequences using pathfinding to optimize the cutting path."""
    def find_nearest_unvisited(current_pos, unvisited, prev_direction=None):
        if not unvisited:
            return None
            
        curr_y, curr_x = current_pos
        candidates = []
        
        for pos in unvisited:
            y, x = pos
            base_dist = abs(y - curr_y) + abs(x - curr_x)
            penalty = 0
            
            new_direction = None
            if y != curr_y or x != curr_x:
                dy = 1 if y > curr_y else -1 if y < curr_y else 0
                dx = 1 if x > curr_x else -1 if x < curr_x else 0
                new_direction = (dy, dx)
            
            if prev_direction and new_direction and prev_direction != new_direction:
                penalty += 2.0
            
            if y != curr_y and x != curr_x:
                penalty += 1.0
            
            if y == curr_y or x == curr_x:
                penalty -= 0.5
                
            future_neighbors = [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]
            future_options = sum(1 for n in future_neighbors if n in unvisited)
            penalty -= future_options * 0.2
            
            total_cost = base_dist + penalty
            candidates.append((total_cost, pos))
            
        return min(candidates, key=lambda x: x[0])[1]
    
    def find_edge_start_points(valid_moves):
        if not valid_moves:
            return []
        
        edge_points = []
        for pos in valid_moves:
            y, x = pos
            if y == 0 or y == env.size-1 or x == 0 or x == env.size-1:
                edge_points.append(pos)
        return edge_points or [valid_moves[0]]
    
    def optimize_path(valid_moves, start_pos):
        path = [start_pos]
        remaining = set(valid_moves) - {start_pos}
        current = start_pos
        prev_direction = None
        
        while remaining:
            next_pos = find_nearest_unvisited(current, remaining, prev_direction)
            if next_pos is None:
                break
            
            y, x = current
            ny, nx = next_pos
            if ny != y or nx != x:
                dy = 1 if ny > y else -1 if ny < y else 0
                dx = 1 if nx > x else -1 if nx < x else 0
                prev_direction = (dy, dx)
                
            path.append(next_pos)
            remaining.remove(next_pos)
            current = next_pos
            
        return path
    
    print("\n=== Starting Testing with Path Optimization ===")
    
    best_path = None
    best_distance = float('inf')
    best_steps = float('inf')
    best_similarity = 0
    best_efficiency = 0
    all_paths = []
    
    valid_moves = list(agent.successful_moves)
    if not valid_moves:
        st.error("No valid moves found from training!")
        return None
    
    all_start_points = find_edge_start_points(valid_moves)
    start_points = random.sample(all_start_points, num_attempts) if len(all_start_points) > num_attempts else all_start_points
    
    for start_pos in start_points:
        state = env.reset()
        current_path = optimize_path(valid_moves, start_pos)
        total_distance = 0
        total_reward = 0
        step_count = 0
        
        for i, position in enumerate(current_path):
            next_state, reward, done = env.step(position)
            
            if i > 0:
                prev_y, prev_x = current_path[i-1]
                curr_y, curr_x = position
                distance = abs(curr_y - prev_y) + abs(curr_x - prev_x)
                total_distance += distance
            
            total_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                similarity = np.sum(env.current_state == env.target_shape) / (env.size * env.size)
                movement_efficiency = 1.0 / (1.0 + total_distance/step_count)
                
                path_data = {
                    'path': current_path,
                    'distance': total_distance,
                    'steps': step_count,
                    'movement_efficiency': movement_efficiency,
                    'similarity': similarity,
                    'reward': total_reward,
                    'start_pos': start_pos
                }
                
                all_paths.append(path_data)
                
                if movement_efficiency > best_efficiency:
                    best_path = current_path
                    best_distance = total_distance
                    best_steps = step_count
                    best_similarity = similarity
                    best_efficiency = movement_efficiency
                break
    
    display_optimization_results(all_paths, best_path, best_steps, best_distance, best_efficiency, env)
    
    return {
        'final_state': env.current_state,
        'final_similarity': best_similarity,
        'perfect_match': np.array_equal(env.current_state, env.target_shape),
        'steps': best_steps,
        'total_distance': best_distance,
        'path': best_path,
        'efficiency': best_efficiency
    }

def display_optimization_results(all_paths, best_path, best_steps, best_distance, best_efficiency, env):
    """Display optimization results."""
    if all_paths:
        all_paths.sort(key=lambda x: (-x['movement_efficiency']))
        
        st.write("### Summary of All Paths")
        summary_data = pd.DataFrame([{
            'Start Position': f"({p['start_pos'][0]}, {p['start_pos'][1]})",
            'Steps': p['steps'],
            'Distance': p['distance'],
            'Movement Efficiency': f"{p['movement_efficiency']:.3f}",
            'Similarity': f"{p['similarity']:.3f}"
        } for p in all_paths])
        st.dataframe(summary_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        efficiencies = [p['movement_efficiency'] for p in all_paths]
        ax.bar(range(len(all_paths)), efficiencies)
        ax.set_xlabel('Path Number')
        ax.set_ylabel('Movement Efficiency')
        ax.set_title('Efficiency Comparison of Different Cutting Sequences')
        st.pyplot(fig)
        plt.close(fig)
        
        st.write("### Best Cutting Sequence Found")
        st.write(f"Starting Position: ({best_path[0][0]}, {best_path[0][1]})")
        st.write(f"Steps: {best_steps}")
        st.write(f"Total Distance: {best_distance:.2f}")
        st.write(f"Movement Efficiency: {best_efficiency:.3f}")
        
        state = env.reset()
        for position in best_path:
            next_state, reward, done = env.step(position)
        
        fig = env.render()
        st.pyplot(fig)
        plt.close(fig)
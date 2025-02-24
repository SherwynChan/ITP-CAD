import numpy as np
import matplotlib.pyplot as plt

class CADPathPlanner:
    def __init__(self, target_shape: np.ndarray, max_steps: int = 1000):
        self.size = target_shape.shape[0]
        self.target_shape = target_shape
        self.max_steps = max_steps
        
        # State variables
        self.current_state = None
        self.current_position = None
        self.cut_history = []
        self.steps_taken = 0
        self.best_similarity = 0
        self.best_state = None
        self.total_distance = 0
        
        # Initialize state
        self.current_state = np.ones((self.size, self.size))
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_state = np.ones((self.size, self.size))
        self.current_position = None
        self.cut_history = []
        self.steps_taken = 0
        self.best_similarity = 0
        self.best_state = None
        self.total_distance = 0
        return self.current_state

    def step(self, action):
        """Execute one step in the environment."""
        self.steps_taken += 1
        
        if not isinstance(action, tuple) or len(action) != 2:
            return self.current_state, -5.0, True
            
        y, x = action
        if not (0 <= y < self.size and 0 <= x < self.size):
            return self.current_state, -5.0, True
            
        if self.current_position is not None:
            prev_y, prev_x = self.current_position
            distance = abs(y - prev_y) + abs(x - prev_x)
            self.total_distance += distance
        
        if self.current_state[y, x] == 1 and self.target_shape[y, x] == 0:
            prev_similarity = np.sum(self.current_state == self.target_shape) / (self.size * self.size)
            
            self.current_state[y, x] = 0
            self.current_position = action
            self.cut_history.append(action)
            
            new_similarity = np.sum(self.current_state == self.target_shape) / (self.size * self.size)
            similarity_improvement = new_similarity - prev_similarity
            
            # Calculate rewards
            movement_efficiency = 1.0 / (1.0 + self.total_distance/self.steps_taken)
            progress_efficiency = similarity_improvement * 10
            completion_ratio = new_similarity
            
            base_reward = 1.0
            movement_bonus = movement_efficiency * 2.0
            progress_bonus = progress_efficiency * 3.0
            completion_bonus = completion_ratio * 1.0
            
            reward = base_reward + movement_bonus + progress_bonus + completion_bonus
            
            if similarity_improvement > self.best_similarity:
                self.best_similarity = new_similarity
                self.best_state = self.current_state.copy()
            
            done = np.array_equal(self.current_state, self.target_shape)
            if done:
                reward += 100
            
            return self.current_state, reward, done
        else:
            return self.current_state, -5.0, True
    
    def render(self):
        """Render the current state."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(self.current_state, cmap='binary', interpolation='nearest')
        ax1.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.set_title('Current State\n(White = Material, Black = Background)', pad=20)
        
        ax2.imshow(self.target_shape, cmap='binary', interpolation='nearest')
        ax2.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_title('Target Shape\n(White = Material, Black = Background)', pad=20)
        
        if self.cut_history:
            y_coords, x_coords = zip(*self.cut_history)
            ax1.plot(x_coords, y_coords, 'r-', alpha=0.5, linewidth=1)
            ax1.scatter(x_coords, y_coords, c=range(len(x_coords)), 
                       cmap='viridis', alpha=0.5, s=30)
        
        similarity = np.sum(self.current_state == self.target_shape) / (self.size * self.size)
        fig.suptitle(f'Similarity: {similarity:.3f} | Steps: {self.steps_taken}', y=0.95)
        
        plt.tight_layout()
        return fig
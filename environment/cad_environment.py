import numpy as np
import matplotlib.pyplot as plt

class CADEnvironment:
    def __init__(self, target_shape=None, size=50):
        self.size = size if target_shape is None else target_shape.shape[0]
        self.target_shape = target_shape if target_shape is not None else self._create_default_target()
        self.current_state = np.ones((self.size, self.size))
        self.best_similarity = 0
        self.best_state = None
        print("Initial state setup:")
        print(self.current_state)
        print("Target shape:")
        print(self.target_shape)
    
    def reset(self):
        """Reset to initial state (filled block)"""
        self.current_state = np.ones((self.size, self.size))
        return self.current_state.copy()

    def step(self, action, position):
        """Execute one step"""
        if action != 0 or position is None:
            return self.current_state.copy(), -1, False
            
        y, x = position
        old_state = self.current_state.copy()
        
        if self.target_shape[y, x] == 0 and self.current_state[y, x] == 1:
            print(f"Removing material at ({y}, {x})")
            self.current_state[y, x] = 0
            reward = 10
        else:
            print(f"Invalid move at ({y}, {x})")
            reward = -10
            self.current_state = old_state
            return self.current_state.copy(), reward, False

        print("Current state after action:")
        print(self.current_state)
        
        similarity = np.sum(self.current_state == self.target_shape) / (self.size * self.size)
        
        if similarity > self.best_similarity:
            self.best_similarity = similarity
            self.best_state = self.current_state.copy()
        
        done = np.array_equal(self.current_state, self.target_shape)
        return self.current_state.copy(), reward, done

    def render(self):
        """Render the current state"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(self.current_state, cmap='binary', interpolation='nearest')
        ax1.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.set_title('Current State\n(White = Material, Black = Background)', pad=20)
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        
        ax2.imshow(self.target_shape, cmap='binary', interpolation='nearest')
        ax2.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_title('Target Shape\n(White = Material, Black = Background)', pad=20)
        ax2.set_xlabel('X coordinate')
        
        similarity = np.sum(self.current_state == self.target_shape) / (self.size * self.size)
        fig.suptitle(f'Similarity: {similarity:.3f}', y=0.95)
        
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        return fig
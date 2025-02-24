import numpy as np

class SystematicPathFinder:
    def __init__(self):
        self.successful_moves = set()
        self.forbidden_moves = set()
        self.current_layer = 0
        self.layer_positions = None
        self.position_index = 0
        self.current_direction = None
        self.last_position = None
        
    def _optimize_layer_positions(self, positions, state_shape):
        """Optimize the cutting sequence within a layer."""
        if not positions:
            return positions
            
        optimized = []
        remaining = set(positions)
        current = positions[0]
        optimized.append(current)
        remaining.remove(current)
        
        while remaining:
            y, x = current
            closest = min(remaining, 
                        key=lambda p: abs(p[0]-y) + abs(p[1]-x))
            optimized.append(closest)
            remaining.remove(closest)
            current = closest
            
        return optimized
    
    def get_layer_positions(self, state):
        """Get optimized positions for current layer in efficient order."""
        try:
            if state.ndim == 3:
                rows, cols = state.shape[1], state.shape[2]
            else:
                rows, cols = state.shape
                
            layer = self.current_layer
            positions = []
            
            if layer >= min(rows, cols) // 2:
                return positions
            
            # Get all positions for this layer
            positions.extend([(layer, j) for j in range(layer, cols-layer)])
            positions.extend([(i, cols-layer-1) for i in range(layer, rows-layer)])
            positions.extend([(rows-layer-1, j) for j in range(cols-layer-1, layer-1, -1)])
            positions.extend([(i, layer) for i in range(rows-layer-1, layer-1, -1)])
            
            return self._optimize_layer_positions(positions, state.shape)
            
        except Exception as e:
            print(f"Error in get_layer_positions: {str(e)}")
            return []
    
    def get_action(self, state, target_shape):
        """Get next position to try cutting with path optimization."""
        try:
            if self.layer_positions is None or self.position_index >= len(self.layer_positions):
                if isinstance(state, np.ndarray) and state.ndim == 3:
                    current_state = state[0]
                else:
                    current_state = state
                    
                self.layer_positions = self.get_layer_positions(current_state)
                self.position_index = 0
                
                if not self.layer_positions:
                    self.current_layer += 1
                    self.layer_positions = self.get_layer_positions(current_state)
                    self.position_index = 0
                
                if not self.layer_positions:
                    return 1, None

            while self.position_index < len(self.layer_positions):
                position = self.layer_positions[self.position_index]
                self.position_index += 1
                
                if position in self.forbidden_moves or position in self.successful_moves:
                    continue
                
                y, x = position
                if isinstance(state, np.ndarray) and state.ndim == 3:
                    current_value = state[0, y, x]
                else:
                    current_value = state[y, x]
                
                target_value = target_shape[y, x]
                
                if np.isclose(current_value, 1.0) and np.isclose(target_value, 0.0):
                    if self.last_position is not None:
                        new_direction = (position[0] - self.last_position[0],
                                       position[1] - self.last_position[1])
                        self.current_direction = new_direction
                    self.last_position = position
                    
                    return 0, position
                else:
                    self.forbidden_moves.add(position)
            
            self.current_layer += 1
            self.layer_positions = None
            return self.get_action(state, target_shape)
            
        except Exception as e:
            print(f"Error in get_action: {str(e)}")
            return 1, None
    
    def learn(self, state, action, reward, next_state, position):
        """Update agent's knowledge based on action results."""
        if position is None:
            return
            
        if reward > 0:
            self.successful_moves.add(position)
        else:
            self.forbidden_moves.add(position)
    
    def reset(self):
        """Reset agent's state for new episode."""
        self.current_layer = 0
        self.layer_positions = None
        self.position_index = 0
        self.current_direction = None
        self.last_position = None
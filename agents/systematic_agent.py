class SystematicCuttingAgent:
    def __init__(self):
        self.successful_moves = set()
        self.forbidden_moves = set()
        self.current_layer = 0
        self.layer_positions = None
        self.position_index = 0
        
    def get_action(self, state, target_shape):
        """Get next position to try cutting"""
        if self.layer_positions is None or self.position_index >= len(self.layer_positions):
            self.layer_positions = self.get_layer_positions(state.shape)
            self.position_index = 0
            
            if not self.layer_positions:
                self.current_layer += 1
                self.layer_positions = self.get_layer_positions(state.shape)
                self.position_index = 0
            
            if not self.layer_positions:
                return 1, None
    
        while self.position_index < len(self.layer_positions):
            position = self.layer_positions[self.position_index]
            self.position_index += 1
            
            if position in self.forbidden_moves or position in self.successful_moves:
                continue
                
            y, x = position
            if state[y, x] == 1 and target_shape[y, x] == 0:
                print(f"Found cutting position at {position}")
                print(f"Current state value: {state[y, x]}")
                print(f"Target shape value: {target_shape[y, x]}")
                return 0, position
            
            self.forbidden_moves.add(position)
        
        self.current_layer += 1
        self.layer_positions = None
        return self.get_action(state, target_shape)
        
    def get_layer_positions(self, shape):
        """Get positions for current layer in clockwise order"""
        rows, cols = shape
        layer = self.current_layer
        positions = []
        
        if layer >= min(rows, cols) // 2:
            return positions
            
        # Process outer layer first, in clockwise order
        positions.extend([(layer, j) for j in range(layer, cols-layer)])  # Top
        positions.extend([(i, cols-layer-1) for i in range(layer, rows-layer)])  # Right
        positions.extend([(rows-layer-1, j) for j in range(cols-layer-1, layer-1, -1)])  # Bottom
        positions.extend([(i, layer) for i in range(rows-layer-1, layer-1, -1)])  # Left
        
        return positions
        
    def learn(self, state, action, reward, next_state, position):
        if position is None:
            return
        if reward > 0:
            self.successful_moves.add(position)
        else:
            self.forbidden_moves.add(position)
    
    def reset(self):
        self.current_layer = 0
        self.layer_positions = None
        self.position_index = 0
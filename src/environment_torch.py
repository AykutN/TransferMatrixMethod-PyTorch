import numpy as np
import os
import sys

# Ensure we can import config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import config.settings as config
from src.tmm_torch import TMMTorch

class Env():
    def __init__(self):
        # Initialize TMM Torch instead of MATLAB
        self.tmm = TMMTorch()
        
        # Initial thicknesses 
        # We need to support N layers dynamically based on config.LAYERS
        # For backward compatibility with DQN Agent (which expects d1..d6), 
        # we might need to map dynamic layers to fixed inputs or update Agent too.
        # But for now, let's assume we stick to 6 variable layers as per user request to replace current system.
        # If config.LAYERS changes size, we need to adapt self.d_values list.
        
        self.num_layers = len(config.LAYERS)
        # Initial thicknesses 
        # Support N layers dynamically based on config.LAYERS
        self.num_layers = len(config.LAYERS)
        self.d_values = []
        
        # Initialize within bounds (e.g., midpoint or random)
        for i in range(self.num_layers):
            key = f"d{i+1}"
            min_b, max_b = config.BOUNDS.get(key, (10, 100))
            # Start at random point or specific if logic exists
            # Let's start at a "safe" random point
            val = float(np.random.randint(min_b, max_b + 1))
            self.d_values.append(val)
            
        # For backward compatibility with main.py logging that accesses env.d1p etc.

        # For backward compatibility with main.py logging that accesses env.d1p etc.
        # We will use property getters or just set attributes effectively.
        self._update_attributes()
        
        self.A_value, self.B_value = self._get_location(*self.d_values)
        self.max_steps = config.MAX_STEPS
        self.action_space = self._generate_action_space()
        self.n_actions = len(self.action_space)                                     
        self.highest_A = self.A_value
        self.highest_B = self.B_value
        self.current_step = 0
        self.previous_A = self.A_value
        self.previous_B = self.B_value
        
        # Extended metrics initialized by _get_location
        # self.cri_value, self.x_value, self.y_value are now available

    def cleanup(self):
        pass # No MATLAB engine to close

    def _update_attributes(self):
        # Expose d1p, d2p... for external loggers expecting them
        for i, val in enumerate(self.d_values):
            setattr(self, f"d{i+1}p", val)

    def _generate_action_space(self):
        # Generate based on config.ACTION_CATEGORIES which lists ["d1", "d2"...]
        # Ideally this should also be dynamic based on len(LAYERS)
        # But config might be static. Let's rely on config.
        categories = config.ACTION_CATEGORIES
        actions = config.ACTIONS_PER_CATEGORY 
        action_space = []
        
        for category in categories:
            for action in actions:
                action_space.append([category, action])  

        return action_space

    def _get_location(self, *d_vals):
        # Call PyTorch TMM
        # Returns: A, B, CRI_ext, x_cr, y_cr
        AVT, Jph, cri, x, y = self.tmm.forward(*d_vals)
        
        # Convert to python floats
        if hasattr(AVT, 'item'): AVT = AVT.item()
        if hasattr(Jph, 'item'): Jph = Jph.item()
        if hasattr(cri, 'item'): cri = cri.item()
        if hasattr(x, 'item'): x = x.item()
        if hasattr(y, 'item'): y = y.item()
        
        # Store extended metrics
        self.cri_value = cri
        self.x_value = x
        self.y_value = y
        self.AVT_value = AVT
        self.Jph_value = Jph
        
        return AVT, Jph

    def reset(self):
        # Random initialization
        self.d_values = []
        for i in range(self.num_layers):
            key = f"d{i+1}"
            low, high = config.BOUNDS.get(key, (10, 100)) # Fallback bounds
            val = np.random.randint(low, high)
            self.d_values.append(float(val))
            
        self._update_attributes()
        
        self.current_step = 0
        self.A_value, self.B_value = self._get_location(*self.d_values)
        self.highest_A = self.A_value
        self.highest_B = self.B_value
        self.previous_A = self.A_value
        self.previous_B = self.B_value
        
        return self._get_state()

    def step(self, action_idx):
        self.current_step += 1
        category, action = self.action_space[action_idx] # e.g. "d1", "forward"
        self.previous_A, self.previous_B = self.A_value, self.B_value
        
        # Parse category index (d1 -> 0, d2 -> 1)
        idx = int(category.replace('d', '')) - 1
        
        bounds = config.BOUNDS
        penalty = 0  

        # Update position
        current_val = self.d_values[idx]
        
        if action == "forward":
            current_val += 1
            if current_val > bounds[category][1]:
                penalty -= 10
        elif action == "backward":
            current_val -= 1
            if current_val < bounds[category][0]:
                penalty -= 10
                
        # Clamp value
        current_val = max(bounds[category][0], min(current_val, bounds[category][1]))
        self.d_values[idx] = current_val
        self._update_attributes()

        # Recalculate physics
        self.A_value, self.B_value = self._get_location(*self.d_values)
        
        # Update tracks
        if self.A_value > self.highest_A: self.highest_A = self.A_value
        if self.B_value > self.highest_B: self.highest_B = self.B_value

        # Calculate Reward (Same logic as original)
        reward = 0
        
        if self.A_value > self.previous_A:
            reward += 5
        else:
            reward -= 5
 
        reward += penalty
        
        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done

    def _get_state(self):
        state = np.array(self.d_values, dtype=np.float32)
        return state

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
        self.d_values = [30.0] * self.num_layers # Default start
        
        # Map specific defaults if 6 layers (standard case)
        if self.num_layers == 6:
             self.d_values = [25.0, 10.0, 25.0, 150.0, 30.0, 30.0]

        # For backward compatibility with main.py logging that accesses env.d1p etc.
        # We will use property getters or just set attributes effectively.
        self._update_attributes()
        
        self.avt_value, self.jph_value = self._get_location(*self.d_values)
        self.max_steps = config.MAX_STEPS
        self.action_space = self._generate_action_space()
        self.n_actions = len(self.action_space)                                     
        self.highest_avt = self.avt_value
        self.highest_jph = self.jph_value
        self.current_step = 0
        self.previous_avt = self.avt_value
        self.previous_jph = self.jph_value

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
        # It handles float inputs and returns float (item())
        avt, jph = self.tmm.forward(*d_vals)
        if hasattr(avt, 'item'): avt = avt.item()
        if hasattr(jph, 'item'): jph = jph.item()
        return avt, jph

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
        self.avt_value, self.jph_value = self._get_location(*self.d_values)
        self.highest_avt = self.avt_value
        self.highest_jph = self.jph_value
        self.previous_avt = self.avt_value
        self.previous_jph = self.jph_value
        
        return self._get_state()

    def step(self, action_idx):
        self.current_step += 1
        category, action = self.action_space[action_idx] # e.g. "d1", "forward"
        self.previous_avt, self.previous_jph = self.avt_value, self.jph_value
        
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
        self.avt_value, self.jph_value = self._get_location(*self.d_values)
        
        # Update tracks
        if self.avt_value > self.highest_avt: self.highest_avt = self.avt_value
        if self.jph_value > self.highest_jph: self.highest_jph = self.jph_value

        # Calculate Reward (Same logic as original)
        reward = 0
        
        if self.avt_value > 25:
            reward += 70
        
        if 25 <= self.avt_value <= 27:
            reward += 150
        else:
            reward -= 50

        if self.jph_value > self.previous_jph:
            reward += 20
        else:
            reward -= 15

        if 25 <= self.avt_value <= 27 and self.jph_value > self.previous_jph:
            reward += 300
            
        if 25 <= self.previous_avt <= 27 and 25 <= self.avt_value <= 27:
            reward += 250
 
        reward += penalty
        
        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done

    def _get_state(self):
        state = np.array(self.d_values, dtype=np.float32)
        return state

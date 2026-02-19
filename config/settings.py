import os

# Base Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MATLAB_DIR = os.path.join(SRC_DIR, 'matlab')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

# Training Hyperparameters
EPISODES = 3000
MAX_STEPS = 250
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 20000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01

# Material Stack Definition (Order matters!)
# 'Vac' is added automatically at start and end by the engine, so list only the active layers.
LAYERS = ['MoO3', 'Ag', 'ZnO', 'PTB7_PCBM', 'MoO3', 'Ag', 'Au']

# Environment Settings
STATE_DIM = len(LAYERS)
ACTION_CATEGORIES = [f"d{i+1}" for i in range(STATE_DIM)]
ACTIONS_PER_CATEGORY = ["forward", "backward"]

# Bounds for optimization variables
# (min, max)
# Auto-generate default bounds if precise ones aren't mapped
# This mapping provides specific bounds for common materials if 6-layer stack is detected logic-wise,
# or we can keep the explicit dict but make it dynamic.
# For now, let's keep the explicit map for the 6 layers, and add a logical fallback generator if LAYERS changes size.

# Helper to generate default bounds if user changes layers
def get_bounds(layers):
    defaults = {
        "MoO3": (5, 70),
        "Ag": (5, 50),
        "ZnO": (10, 70),
        "PTB7_PCBM": (50, 250), # Active layer usually thicker
        # Add generic default
        "DEFAULT": (10, 100)
    }
    
    bounds = {}
    for i, mat in enumerate(layers):
        key = f"d{i+1}"
        # Some specific logic for the 6-layer stack provided originally:
        # d1(MoO3): 5-70
        # d2(Ag): 10-50 (Ag usually thin)
        # d3(ZnO): 10-70
        # d4(Active): 50-250
        # d5(MoO3): 10-50 (Top MoO3 might be thinner/thicker?)
        # d6(Ag): 5-100
        
        # If we recognize the specific original structure, we could hardcode,
        # but better to use material-based defaults.
        # Let's use the material name to pick defaults.
        
        mat_base = mat.split('_')[0] # e.g. PTB7 from PTB7_PCBM
        
        # Override for specific indices if needed (like the original code had d6 Ag up to 100, but d2 Ag up to 50)
        # We will try to respect original exact values if length is 6
        if len(layers) == 6:
             original_bounds = {
                "d1": (5, 70),
                "d2": (10, 50),
                "d3": (10, 70),
                "d4": (50, 250),
                "d5": (10, 50),
                "d6": (5, 100),
             }
             return original_bounds
             
        # Fallback for other structures
        b = defaults.get(mat, defaults.get("DEFAULT"))
        bounds[key] = b
    return bounds

BOUNDS = get_bounds(LAYERS)

# Reward System Config
REWARD = {
    "Penalty_Out_Of_Bounds": -10,
    "Penalty_Low_AVT": 70, # Logic in original code was somewhat inverted/complex, we'll keep the logic but maybe fine tune values here if needed.
    # For now, hardcoded complex logic in environment.py is safer to keep as is, but we can define simple constants.
}

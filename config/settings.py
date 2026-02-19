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

# Environment Settings
STATE_DIM = 6
ACTION_CATEGORIES = ["d1", "d2", "d3", "d4", "d5", "d6"]
ACTIONS_PER_CATEGORY = ["forward", "backward"]

# Material Stack Definition (Order matters!)
# 'Vac' is added automatically at start and end by the engine, so list only the active layers.
LAYERS = ['MoO3', 'Ag', 'ZnO', 'PTB7_PCBM', 'MoO3', 'Ag']

# Bounds for optimization variables
# (min, max)
BOUNDS = {
    "d1": (5, 70),
    "d2": (10, 50),
    "d3": (10, 70),
    "d4": (50, 250),
    "d5": (10, 50),
    "d6": (5, 100),
}

# Reward System Config
REWARD = {
    "Penalty_Out_Of_Bounds": -10,
    "Penalty_Low_AVT": 70, # Logic in original code was somewhat inverted/complex, we'll keep the logic but maybe fine tune values here if needed.
    # For now, hardcoded complex logic in environment.py is safer to keep as is, but we can define simple constants.
}

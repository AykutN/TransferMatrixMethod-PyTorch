import numpy as np
import os
import sys

# Ensure we can import config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import config.settings as config

# Conditional Import based on configuration or default preference
# We will use the PyTorch Environment by default as requested.
# If user wants MATLAB, they can manually revert or we can add a flag.
# For now, let's make this file a wrapper that points to the PyTorch Env.

from src.environment_torch import Env as TorchEnv

class Env(TorchEnv):
    """
    Wrapper class that uses PyTorch TMM implementation.
    Inherits everything from environment_torch.Env
    """
    def __init__(self):
        print("Initializing PyTorch-based Environment (Physics-Informed)...")
        super().__init__()


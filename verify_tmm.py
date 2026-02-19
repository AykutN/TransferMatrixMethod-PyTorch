import sys
import os
import torch
import numpy as np
import time

# Add project root
sys.path.append(os.getcwd())

import config.settings as config
from src.environment import Env
from src.tmm_torch import TMMTorch

def verify_equivalence():
    print("Initializing PyTorch TMM...")
    tmm_torch = TMMTorch()
    
    print("Initializing MATLAB TMM (via Env)...")
    try:
        env = Env()
    except Exception as e:
        print(f"Failed to init MATLAB env: {e}")
        return

    # Random Test Cases
    num_tests = 5
    print(f"\nRunning {num_tests} random test cases...")
    
    errors_avt = []
    errors_jph = []
    
    for i in range(num_tests):
        # Generate random inputs within bounds
        # d1..d6
        d1 = np.random.uniform(*config.BOUNDS["d1"])
        d2 = np.random.uniform(*config.BOUNDS["d2"])
        d3 = np.random.uniform(*config.BOUNDS["d3"])
        d4 = np.random.uniform(*config.BOUNDS["d4"])
        d5 = np.random.uniform(*config.BOUNDS["d5"])
        d6 = np.random.uniform(*config.BOUNDS["d6"])
        
        # Run MATLAB
        start_m = time.time()
        # Env uses these internal variables, we can hack them or use _get_location
        avt_matlab, jph_matlab = env._get_location(d1, d2, d3, d4, d5, d6)
        time_m = time.time() - start_m
        
        # Run PyTorch
        start_p = time.time()
        d_tensor = [torch.tensor(x, dtype=torch.float64) for x in [d1, d2, d3, d4, d5, d6]]
        avt_torch, jph_torch = tmm_torch.forward(*d_tensor)
        time_p = time.time() - start_p
        
        # Compare
        avt_p_val = avt_torch.item()
        jph_p_val = jph_torch.item()
        
        err_avt = abs(avt_matlab - avt_p_val)
        err_jph = abs(jph_matlab - jph_p_val)
        
        errors_avt.append(err_avt)
        errors_jph.append(err_jph)
        
        print(f"Test {i+1}:")
        print(f"  Inputs: d1={d1:.2f}, d2={d2:.2f}, ...")
        print(f"  MATLAB: AVT={avt_matlab:.6f}, Jph={jph_matlab:.6f} ({time_m*1000:.1f}ms)")
        print(f"  PyTorch: AVT={avt_p_val:.6f}, Jph={jph_p_val:.6f} ({time_p*1000:.1f}ms)")
        print(f"  Diff: AVT={err_avt:.2e}, Jph={err_jph:.2e}")
        
    print("\nSummary:")
    print(f"Mean Error AVT: {np.mean(errors_avt):.2e}")
    print(f"Mean Error Jph: {np.mean(errors_jph):.2e}")
    
    if np.mean(errors_avt) < 1e-4 and np.mean(errors_jph) < 1e-4:
        print("SUCCESS: Implementations are equivalent within tolerance.")
    else:
        print("WARNING: Differences detected larger than expected.")

    env.cleanup()

if __name__ == "__main__":
    verify_equivalence()

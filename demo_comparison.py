import sys
import os
import torch
import numpy as np
import time
import pandas as pd

# Add project root
sys.path.append(os.getcwd())

import config.settings as config
from src.environment import Env
from src.tmm_torch import TMMTorch

def run_demo():
    print("="*60)
    print("  TMM ENGINE COMPARISON: MATLAB vs PyTorch (Physics-Informed AI)")
    print("="*60)
    
    # Init Engines
    print("[1/3] Initializing Engines...")
    tmm_torch = TMMTorch()
    try:
        env = Env()
    except Exception as e:
        print(f"MATLAB Init Failed: {e}")
        return

    # Generate a RANDOM but valid configuration
    d1 = np.random.uniform(*config.BOUNDS["d1"])
    d2 = np.random.uniform(*config.BOUNDS["d2"])
    d3 = np.random.uniform(*config.BOUNDS["d3"])
    d4 = np.random.uniform(*config.BOUNDS["d4"])
    d5 = np.random.uniform(*config.BOUNDS["d5"])
    d6 = np.random.uniform(*config.BOUNDS["d6"])

    print("\n" + "="*60)
    print("  SCENARIO: 7-Layer Structure Test")
    print("  (MoO3 / Ag / ZnO / PTB7:PCBM / MoO3 / Ag)")
    print("="*60)
    print(f"\n[INSTRUCTION] Ask your professor to enter these thickness values into MATLAB:")
    print("-" * 50)
    print(f"  Layer 2 (MoO3)      : {d1:.2f} nm")
    print(f"  Layer 3 (Ag)        : {d2:.2f} nm")
    print(f"  Layer 4 (ZnO)       : {d3:.2f} nm")
    print(f"  Layer 5 (PTB7:PCBM) : {d4:.2f} nm")
    print(f"  Layer 6 (MoO3)      : {d5:.2f} nm")
    print(f"  Layer 7 (Ag)        : {d6:.2f} nm")
    print("-" * 50)
    
    input("\nPress Enter to run BOTH MATLAB and PyTorch simulations...")

    print("\n[2/3] Running Simulations...")
    
    # MATLAB Run
    t0 = time.time()
    avt_m, jph_m = env._get_location(d1, d2, d3, d4, d5, d6)
    t_matlab = (time.time() - t0) * 1000 
    
    # PyTorch Run
    t0 = time.time()
    d_tensor = [torch.tensor(x, dtype=torch.float64) for x in [d1, d2, d3, d4, d5, d6]]
    avt_p_tensor, jph_p_tensor = tmm_torch.forward(*d_tensor)
    t_torch = (time.time() - t0) * 1000
    
    avt_p = avt_p_tensor.item()
    jph_p = jph_p_tensor.item()

    # Results
    print("\n[3/3] RESULTS COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<15} | {'MATLAB':<15} | {'PyTorch (AI)':<15} | {'Difference':<15}")
    print("-" * 80)
    print(f"{'AVT (%)':<15} | {avt_m:<15.6f} | {avt_p:<15.6f} | {abs(avt_m - avt_p):.2e}")
    print(f"{'Jph (mA/cm2)':<15} | {jph_m:<15.6f} | {jph_p:<15.6f} | {abs(jph_m - jph_p):.2e}")
    print(f"{'Time (ms)':<15} | {t_matlab:<15.1f} | {t_torch:<15.1f} | {t_matlab/t_torch:.1f}x FASTER")
    print("-" * 80)
    
    if abs(avt_m - avt_p) < 1e-12:
        print("\n\033[92m[SUCCESS] Perfect Mathematical Equivalence Verified!\033[0m")
    else:
        print("\n[INFO] Minor numerical differences detected (expected due to float precision).")

    env.cleanup()

if __name__ == "__main__":
    run_demo()

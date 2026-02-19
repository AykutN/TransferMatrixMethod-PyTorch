import sys
import os
import torch
import torch.optim as optim
import numpy as np
import time

# Add project root
sys.path.append(os.getcwd())

import config.settings as config
from src.tmm_torch import TMMTorch

def optimize_structure():
    print("="*60)
    print("  AI-DRIVEN OPTIMIZATION (Physics-Informed Gradient Descent)")
    print("="*60)
    
    device = torch.device('cpu') # Use 'cuda' if available usually, but CPU is fine here.
    
    # 1. Initialize Physics Engine
    print("[1/4] Initializing Differentiable Physics Engine...")
    tmm = TMMTorch(device=device)
    
    # 2. Define Learnable Parameters (The Thicknesses)
    # We start from a random guess, but we make them PyTorch Parameters.
    # requires_grad=True is the magic key.
    print("[2/4] Initializing Design Parameters...")
    
    # Initial Guess (Random valid)
    initial_d = [
        np.random.uniform(*config.BOUNDS["d1"]),
        np.random.uniform(*config.BOUNDS["d2"]),
        np.random.uniform(*config.BOUNDS["d3"]),
        np.random.uniform(*config.BOUNDS["d4"]),
        np.random.uniform(*config.BOUNDS["d5"]),
        np.random.uniform(*config.BOUNDS["d6"])
    ]
    
    # Create tensors with gradient tracking
    # We use a single tensor for easier optimization
    d_params = torch.tensor(initial_d, dtype=torch.float64, device=device, requires_grad=True)
    
    print(f"  Starting Guess (nm): {d_params.detach().numpy().round(2)}")
    
    # 3. Define Optimizer
    # Adam is a standard AI optimizer.
    optimizer = optim.Adam([d_params], lr=2.0) # High learning rate for fast convergence in physical space
    
    # Scheduler to slow down as we get closer
    # 'verbose' argument is sometimes problematic in newer versions or requires different syntax. Removing for safety.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, factor=0.5)

    print("\n[3/4] Starting Optimization Loop...")
    print("-" * 80)
    print(f"{'Step':<5} | {'AVT':<10} | {'Jph':<10} | {'Reward':<10} | {'Thicknesses (d1..d6)'}")
    print("-" * 80)
    
    history = []
    start_time = time.time()
    
    best_reward = -float('inf')
    best_design = None
    
    steps = 300 # 300 gradient steps (vs 3000 RL episodes)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # --- A. Enforce Bounds (Soft Constraint) ---
        # Since d_params can go anywhere during gradient step, we clamp them logically
        # or use "Sigmoid" trick. Here we just clamp before forward pass for stability,
        # but technically we should constrain the optimization space.
        # Simple method: Projected Gradient Descent (clamp data)
        with torch.no_grad():
            d_params.data[0].clamp_(*config.BOUNDS["d1"])
            d_params.data[1].clamp_(*config.BOUNDS["d2"])
            d_params.data[2].clamp_(*config.BOUNDS["d3"])
            d_params.data[3].clamp_(*config.BOUNDS["d4"])
            d_params.data[4].clamp_(*config.BOUNDS["d5"])
            d_params.data[5].clamp_(*config.BOUNDS["d6"])
        
        # --- B. Forward Pass (The Physics) ---
        # Unpack
        d1, d2, d3, d4, d5, d6 = d_params[0], d_params[1], d_params[2], d_params[3], d_params[4], d_params[5]
        
        # Calculate Physics
        avt, jph = tmm.forward(d1, d2, d3, d4, d5, d6)
        
        # --- C. Define Loss Function (The Goal) ---
        # Goal: AVT >= 30.0, Maximize Jph.
        
        # 1. AVT Constraint (Hard Barrier / Penalty)
        # We want AVT >= 25
        target_avt = 25.0
        
        # If AVT < 30, penalty is proportional to square of violation (to pull it up fast)
        avt_violation = torch.relu(target_avt - avt) # 0 if avt >= 30, else (30 - avt)
        
        # Penalty Weight (Must be large enough to dominate Jph gain)
        penalty_weight = 10.0 
        penalty = penalty_weight * (avt_violation ** 2)
        
        # 2. Jph Objective (Maximize)
        # Jph is typically 0-25. 
        jph_score = jph
        
        # Combined Reward
        # Reward = Jph - Penalty
        reward = jph_score - penalty
        
        loss = -reward 
        
        # --- D. Backpropagation (The AI Magic) ---
        loss.backward()
        
        # --- E. Step ---
        optimizer.step()
        
        # Logging
        current_reward = reward.item()
        history.append(current_reward)
        scheduler.step(current_reward)
        
        if current_reward > best_reward:
            best_reward = current_reward
            best_design = d_params.detach().clone()
            
        if step % 10 == 0:
            d_numpy = d_params.detach().cpu().numpy().round(1)
            print(f"{step:<5} | {avt.item():<10.2f} | {jph.item():<10.2f} | {current_reward:<10.2f} | {d_numpy}")
            
    total_time = time.time() - start_time
    
    print("-" * 80)
    print("\n[4/4] OPTIMIZATION COMPLETE")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Algorithm: Gradient Descent via Automatic Differentiation")
    
    print("\n" + "="*60)
    print("  BEST DESIGN FOUND")
    print("="*60)
    best_d = best_design.cpu().numpy()
    
    # Calculate final stats for best
    with torch.no_grad():
        b_avt, b_jph = tmm.forward(*best_design)
        
    print(f"  AVT : {b_avt.item():.2f} %")
    print(f"  Jph : {b_jph.item():.2f} mA/cm2")
    print("-" * 40)
    print("  Optimal Thicknesses:")
    print(f"  d1 (MoO3)      : {best_d[0]:.2f} nm")
    print(f"  d2 (Ag)        : {best_d[1]:.2f} nm")
    print(f"  d3 (ZnO)       : {best_d[2]:.2f} nm")
    print(f"  d4 (PTB7:PCBM) : {best_d[3]:.2f} nm")
    print(f"  d5 (MoO3)      : {best_d[4]:.2f} nm")
    print(f"  d6 (Ag)        : {best_d[5]:.2f} nm")
    print("="*60)

if __name__ == "__main__":
    optimize_structure()

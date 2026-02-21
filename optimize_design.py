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
    
    device = torch.device('cpu')
    
    # 1. Initialize Physics Engine
    print("[1/4] Initializing Differentiable Physics Engine...")
    tmm = TMMTorch(device=device)
    
    # 2. Define Learnable Parameters (Dynamic layer support)
    print("[2/4] Initializing Design Parameters...")
    
    num_layers = len(config.LAYERS)
    bounds_list = [config.BOUNDS[f"d{i+1}"] for i in range(num_layers)]
    
    # Initial Guess (Random valid)
    initial_d = [np.random.uniform(lo, hi) for lo, hi in bounds_list]
    
    # Create tensors with gradient tracking
    d_params = torch.tensor(initial_d, dtype=torch.float64, device=device, requires_grad=True)
    
    print(f"  Layers: {config.LAYERS}")
    print(f"  Starting Guess (nm): {d_params.detach().numpy().round(2)}")
    
    # 3. Define Optimizer
    optimizer = optim.Adam([d_params], lr=2.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, factor=0.5)

    d_header = ', '.join([f'd{i+1}' for i in range(num_layers)])
    print(f"\n[3/4] Starting Optimization Loop...")
    print("-" * 80)
    print(f"{'Step':<5} | {'AVT':<10} | {'Jph':<10} | {'Reward':<10} | Thicknesses ({d_header})")
    print("-" * 80)
    
    history = []
    start_time = time.time()
    
    best_reward = -float('inf')
    best_design = None
    
    steps = 300
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Projected Gradient Descent: clamp to bounds
        with torch.no_grad():
            for i, (lo, hi) in enumerate(bounds_list):
                d_params.data[i].clamp_(lo, hi)
        
        # Forward pass through differentiable TMM
        avt, jph, cri, x_cr, y_cr = tmm.forward(d_params)
        
        # Loss: Maximize Jph subject to AVT >= 25%
        target_avt = 25.0
        avt_violation = torch.relu(target_avt - avt)
        penalty_weight = 10.0 
        penalty = penalty_weight * (avt_violation ** 2)
        
        reward = jph - penalty
        loss = -reward 
        
        loss.backward()
        optimizer.step()
        
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
    
    with torch.no_grad():
        b_avt, b_jph, b_cri, b_x, b_y = tmm.forward(*best_design)
        
    print(f"  AVT : {b_avt.item():.2f} %")
    print(f"  Jph : {b_jph.item():.2f} mA/cm²")
    print(f"  CRI : {b_cri.item():.1f}")
    print(f"  x   : {b_x.item():.4f}")
    print(f"  y   : {b_y.item():.4f}")
    print("-" * 40)
    print("  Optimal Thicknesses:")
    for i in range(num_layers):
        print(f"  d{i+1} ({config.LAYERS[i]:>10s}) : {best_d[i]:.2f} nm")
    print("="*60)

if __name__ == "__main__":
    optimize_structure()

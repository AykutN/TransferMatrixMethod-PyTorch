#!/usr/bin/env python3
"""
Multi-Objective Gradient-Based Pareto Optimization for Thin-Film Optoelectronic Devices.

Uses Weighted Chebyshev Scalarization with the differentiable TMM engine to generate
Pareto-optimal trade-off surfaces for A vs B, subject to CRI and chromaticity constraints.

This replaces traditional evolutionary algorithms (e.g., NSGA-II) with gradient-based methods,
achieving 10–100x faster Pareto front generation.

Author: Y. Aykut
"""

import sys
import os
import argparse
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config.settings as config
from src.tmm_torch import TMMTorch


# ───────────────────────────────────────────────────────
#  Non-Dominated Sorting (Pareto Filtering)
# ───────────────────────────────────────────────────────

def is_pareto_optimal(costs):
    """
    Find Pareto-optimal (non-dominated) points.
    
    Args:
        costs: (N, M) array where we MINIMIZE all objectives.
               For maximization objectives, negate them before calling.
    
    Returns:
        Boolean mask of length N, True for Pareto-optimal points.
    """
    n = costs.shape[0]
    is_optimal = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_optimal[i]:
            continue
        # A point j dominates i if j is <= in all objectives and < in at least one
        for j in range(n):
            if i == j or not is_optimal[j]:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_optimal[i] = False
                break
    
    return is_optimal


# ───────────────────────────────────────────────────────
#  Single-Weight Optimization (Chebyshev Scalarization)
# ───────────────────────────────────────────────────────

def optimize_single_weight(
    tmm, w_a, w_b, utopia_a, utopia_b,
    num_restarts=5, steps=200, lr=2.0,
    a_min=None, a_max=None, cri_min=None,
    penalty_weight=20.0, device='cpu'
):
    """
    Solve a single weighted Chebyshev subproblem:
        min  max{ w_a * |A - utopia_a|, w_b * |B - utopia_b| }
        s.t. A ∈ [a_min, a_max], CRI ≥ cri_min (soft constraints)
    
    Uses multi-restart to avoid local minima.
    
    Returns:
        best result dict with thicknesses + objectives
    """
    num_layers = len(config.LAYERS)
    bounds_list = [config.BOUNDS[f"d{i+1}"] for i in range(num_layers)]
    
    best_result = None
    best_chebyshev = float('inf')
    
    for restart in range(num_restarts):
        # Random initial guess within bounds
        init_d = [np.random.uniform(lo, hi) for lo, hi in bounds_list]
        d_params = torch.tensor(init_d, dtype=torch.float64, device=device, requires_grad=True)
        
        optimizer = optim.Adam([d_params], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=30, factor=0.5, min_lr=0.01
        )
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Enforce bounds via projection
            with torch.no_grad():
                for i, (lo, hi) in enumerate(bounds_list):
                    d_params.data[i].clamp_(lo, hi)
            
            # Forward pass through differentiable TMM
            a, b, cri, x_cr, y_cr = tmm.forward(d_params)
            
            # ── Chebyshev Scalarization ──
            # We want to MAXIMIZE both A and B, so the "distance" from utopia
            # is (utopia - actual). We minimize the max weighted distance.
            dist_a = (utopia_a - a)  # positive means below utopia
            dist_b = (utopia_b - b)
            
            chebyshev = torch.max(
                w_a * dist_a,
                w_b * dist_b
            )
            
            # ── Constraint Penalties ──
            penalty = torch.tensor(0.0, dtype=torch.float64, device=device)
            
            if a_min is not None:
                violation = torch.relu(a_min - a)
                penalty = penalty + penalty_weight * violation ** 2
            
            if a_max is not None:
                violation = torch.relu(a - a_max)
                penalty = penalty + penalty_weight * violation ** 2
            
            if cri_min is not None:
                violation = torch.relu(cri_min - cri)
                penalty = penalty + penalty_weight * 0.5 * violation ** 2
            
            loss = chebyshev + penalty
            
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
        
        # Final evaluation with bounds enforced
        with torch.no_grad():
            for i, (lo, hi) in enumerate(bounds_list):
                d_params.data[i].clamp_(lo, hi)
            a_f, b_f, cri_f, x_f, y_f = tmm.forward(d_params)
        
        final_cheby = max(
            w_a * (utopia_a - a_f.item()),
            w_b * (utopia_b - b_f.item())
        )
        
        if final_cheby < best_chebyshev:
            best_chebyshev = final_cheby
            thicknesses = d_params.detach().cpu().numpy().copy()
            best_result = {
                'thicknesses': thicknesses,
                'A': a_f.item(),
                'B': b_f.item(),
                'CRI': cri_f.item(),
                'x': x_f.item(),
                'y': y_f.item(),
                'A': a_f.item() if hasattr(a_f, 'item') else a_f,
                'B': b_f.item() if hasattr(b_f, 'item') else b_f,
                'w_a': w_a,
                'w_b': w_b,
                'chebyshev': float(final_cheby),
            }
    
    return best_result


# ───────────────────────────────────────────────────────
#  Utopia Point Estimation
# ───────────────────────────────────────────────────────

def estimate_utopia(tmm, num_restarts=10, steps=300, lr=2.0, device='cpu'):
    """
    Estimate the utopia point by individually maximizing A and B.
    The utopia point is the (max_A, max_B) and is generally infeasible,
    but serves as the reference for Chebyshev scalarization.
    """
    num_layers = len(config.LAYERS)
    bounds_list = [config.BOUNDS[f"d{i+1}"] for i in range(num_layers)]
    
    best_a = -float('inf')
    best_b = -float('inf')
    
    for objective in ['a', 'b']:
        for _ in range(num_restarts):
            init_d = [np.random.uniform(lo, hi) for lo, hi in bounds_list]
            d_params = torch.tensor(init_d, dtype=torch.float64, device=device, requires_grad=True)
            optimizer = optim.Adam([d_params], lr=lr)
            
            for step in range(steps):
                optimizer.zero_grad()
                with torch.no_grad():
                    for i, (lo, hi) in enumerate(bounds_list):
                        d_params.data[i].clamp_(lo, hi)
                
                a, b, cri, x_cr, y_cr = tmm.forward(d_params)
                
                if objective == 'a':
                    loss = -a
                else:
                    loss = -b
                
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                for i, (lo, hi) in enumerate(bounds_list):
                    d_params.data[i].clamp_(lo, hi)
                a_f, b_f, _, _, _ = tmm.forward(d_params)
            
            if objective == 'a':
                best_a = max(best_a, a_f.item())
            else:
                best_b = max(best_b, b_f.item())
    
    return best_a, best_b


# ───────────────────────────────────────────────────────
#  Main Pareto Optimization Pipeline
# ───────────────────────────────────────────────────────

def run_pareto_optimization(
    num_weights=100,
    num_restarts=5,
    steps=200,
    lr=2.0,
    a_min=None,
    a_max=None,
    cri_min=None,
    penalty_weight=20.0,
    output_dir=None,
    device='cpu'
):
    """
    Generate the Pareto front for A vs B using gradient-based
    Weighted Chebyshev Scalarization.
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, 'pareto')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  MULTI-OBJECTIVE PARETO OPTIMIZATION")
    print("  Gradient-Based Weighted Chebyshev Scalarization")
    print("=" * 70)
    print(f"  Layers       : {config.LAYERS}")
    print(f"  Weight vectors: {num_weights}")
    print(f"  Restarts/weight: {num_restarts}")
    print(f"  Steps/restart : {steps}")
    print(f"  A constraint: [{a_min}, {a_max}]")
    print(f"  CRI constraint: ≥ {cri_min}")
    print("-" * 70)
    
    # 1. Initialize TMM Engine
    print("\n[1/4] Initializing Differentiable TMM Engine...")
    tmm = TMMTorch(device=device)
    
    # 2. Estimate Utopia Point
    print("[2/4] Estimating Utopia Point (individual optima)...")
    t0 = time.time()
    utopia_a, utopia_b = estimate_utopia(
        tmm, num_restarts=max(3, num_restarts), steps=steps, lr=lr, device=device
    )
    # Add a small margin to ensure the utopia is strictly better
    utopia_a *= 1.05
    utopia_b *= 1.05
    print(f"  Utopia Point: A* = {utopia_a:.2f}%, B* = {utopia_b:.2f} mA/cm²")
    print(f"  (Estimated in {time.time() - t0:.1f}s)")
    
    # 3. Generate Weight Vectors and Solve Subproblems
    print(f"\n[3/4] Solving {num_weights} Chebyshev subproblems...")
    
    # Weight vectors: w_a from 0.01 to 0.99 (avoid pure extremes)
    weights_a = np.linspace(0.01, 0.99, num_weights)
    weights_b = 1.0 - weights_a
    
    all_results = []
    t0 = time.time()
    
    for idx, (w_a, w_j) in enumerate(zip(weights_a, weights_b)):
        result = optimize_single_weight(
            tmm, w_a, w_j, utopia_a, utopia_b,
            num_restarts=num_restarts, steps=steps, lr=lr,
            a_min=a_min, a_max=a_max, cri_min=cri_min,
            penalty_weight=penalty_weight, device=device
        )
        all_results.append(result)
        
        if (idx + 1) % max(1, num_weights // 10) == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (num_weights - idx - 1)
            print(f"  [{idx+1}/{num_weights}] A={result['A']:.2f}%, "
                  f"B={result['B']:.2f} mA/cm², CRI={result['CRI']:.1f} "
                  f"| Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
    
    total_time = time.time() - t0
    
    # 4. Non-Dominated Filtering
    print("\n[4/4] Filtering non-dominated solutions...")
    
    # Build DataFrame from all results
    num_layers = len(config.LAYERS)
    rows = []
    for r in all_results:
        row = {}
        for i in range(num_layers):
            row[f'd{i+1}'] = r['thicknesses'][i]
        row['A'] = r['A']
        row['B'] = r['B']
        row['CRI'] = r['CRI']
        row['x'] = r['x']
        row['y'] = r['y']
        row['A'] = r['A']
        row['B'] = r['B']
        row['w_a'] = r['w_a']
        row['w_b'] = r['w_b']
        rows.append(row)
    
    df_all = pd.DataFrame(rows)
    
    # Non-dominated sort (minimize negative A, negative B → maximize both)
    costs = np.column_stack([-df_all['A'].values, -df_all['B'].values])
    pareto_mask = is_pareto_optimal(costs)
    
    df_pareto = df_all[pareto_mask].copy()
    df_pareto = df_pareto.sort_values('A').reset_index(drop=True)
    
    # Remove duplicate designs (within tolerance)
    df_pareto = df_pareto.round({'A': 2, 'B': 2, 'CRI': 1}).drop_duplicates(
        subset=['A', 'B'], keep='first'
    ).reset_index(drop=True)
    
    # Save outputs
    all_csv = os.path.join(output_dir, 'pareto_all_solutions.csv')
    pareto_csv = os.path.join(output_dir, 'pareto_front.csv')
    
    df_all.to_csv(all_csv, index=False, float_format='%.4f')
    df_pareto.to_csv(pareto_csv, index=False, float_format='%.4f')
    
    # Print Summary
    print("\n" + "=" * 70)
    print("  PARETO OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"  Total time     : {total_time:.1f} seconds")
    print(f"  Total solutions: {len(df_all)}")
    print(f"  Pareto-optimal : {len(df_pareto)}")
    print(f"  A range      : [{df_pareto['A'].min():.2f}%, {df_pareto['A'].max():.2f}%]")
    print(f"  B range      : [{df_pareto['B'].min():.2f}, {df_pareto['B'].max():.2f}] mA/cm²")
    print(f"  CRI range      : [{df_pareto['CRI'].min():.1f}, {df_pareto['CRI'].max():.1f}]")
    print("-" * 70)
    print(f"  All solutions → {all_csv}")
    print(f"  Pareto front  → {pareto_csv}")
    print("=" * 70)
    
    # Dominance quality check
    a_vals = df_pareto['A'].values
    b_vals = df_pareto['B'].values
    dominated = False
    for i in range(len(a_vals)):
        for j in range(len(a_vals)):
            if i != j and a_vals[j] >= a_vals[i] and b_vals[j] >= b_vals[i]:
                if a_vals[j] > a_vals[i] or b_vals[j] > b_vals[i]:
                    dominated = True
                    break
        if dominated:
            break
    
    if not dominated:
        print("\n✅ DOMINANCE CHECK PASSED: No dominated solutions in Pareto front.")
    else:
        print("\n⚠️  WARNING: Some solutions may still be dominated (increase restarts).")
    
    # Print top-5 designs
    print("\n  TOP PARETO DESIGNS (by B):")
    print("-" * 70)
    top5 = df_pareto.nlargest(5, 'B')
    d_cols = [f'd{i+1}' for i in range(num_layers)]
    for idx, row in top5.iterrows():
        d_str = ", ".join([f"{row[c]:.1f}" for c in d_cols])
        print(f"  A={row['A']:5.2f}% | B={row['B']:5.2f} | CRI={row['CRI']:5.1f} | d=[{d_str}]")
    
    return df_all, df_pareto


# ───────────────────────────────────────────────────────
#  CLI Entry Point
# ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Objective Pareto Optimization for Thin-Film Optoelectronics'
    )
    parser.add_argument('--num_weights', type=int, default=100,
                        help='Number of weight vectors for Chebyshev decomposition (default: 100)')
    parser.add_argument('--num_restarts', type=int, default=5,
                        help='Random restarts per weight vector (default: 5)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Gradient descent steps per restart (default: 200)')
    parser.add_argument('--lr', type=float, default=2.0,
                        help='Adam learning rate (default: 2.0)')
    parser.add_argument('--a_min', type=float, default=None,
                        help='Minimum A constraint (%%), e.g. 25')
    parser.add_argument('--a_max', type=float, default=None,
                        help='Maximum A constraint (%%)')
    parser.add_argument('--cri_min', type=float, default=None,
                        help='Minimum CRI constraint, e.g. 80')
    parser.add_argument('--penalty_weight', type=float, default=20.0,
                        help='Penalty weight for constraint violations (default: 20.0)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/pareto/)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Compute device (default: cpu)')
    
    args = parser.parse_args()
    
    run_pareto_optimization(
        num_weights=args.num_weights,
        num_restarts=args.num_restarts,
        steps=args.steps,
        lr=args.lr,
        a_min=args.a_min,
        a_max=args.a_max,
        cri_min=args.cri_min,
        penalty_weight=args.penalty_weight,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == '__main__':
    main()

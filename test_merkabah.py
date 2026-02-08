#!/usr/bin/env python3
"""
Special Geometries Test — Centered structures with unified core
"""

import numpy as np
from shell_scaling_test import vortex_merkabah, centered_icosahedron, measure_recovery

POOL_SIZES = [100, 300, 1000, 3000, 10000]
TRIALS = 5

# =============================================================================
# CENTERED ICOSAHEDRON (13V)
# =============================================================================

print("="*60)
print("CENTERED ICOSAHEDRON — 13V (12 outer + 1 core)")
print("="*60)

v, adj = centered_icosahedron()
print(f"\n{v} vertices, center degree = {len(adj[0])}")
print("-" * 40)

for pool in POOL_SIZES:
    rates = []
    steps_list = []
    
    for _ in range(TRIALS):
        rate, steps = measure_recovery(adj, pool)
        rates.append(rate)
        steps_list.append(steps)
    
    avg_rate = np.mean(rates)
    avg_steps = np.mean(steps_list)
    
    print(f"  Pool {pool:>6}: rate={avg_rate:.4f}  →90%={avg_steps:.1f}")

# =============================================================================
# VORTEX MERKABAH (bilateral ×3)
# =============================================================================

print("\n" + "="*60)
print("VORTEX MERKABAH — Bilateral ×3 Encoding")
print("="*60)

for n_layers in [1, 2, 3, 4, 5, 6]:
    v, adj = vortex_merkabah(n_layers)
    
    print(f"\n{n_layers}-layer merkabah ({v} vertices)")
    print("-" * 40)
    
    for pool in POOL_SIZES:
        rates = []
        steps_list = []
        
        for _ in range(TRIALS):
            rate, steps = measure_recovery(adj, pool)
            rates.append(rate)
            steps_list.append(steps)
        
        avg_rate = np.mean(rates)
        avg_steps = np.mean(steps_list)
        
        print(f"  Pool {pool:>6}: rate={avg_rate:.4f}  →90%={avg_steps:.1f}")

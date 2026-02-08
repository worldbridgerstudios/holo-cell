#!/usr/bin/env python3
"""
Crystal Layer Test — Finding optimal pool boundaries for each layer count K

Structure (the One + 3 + 12 pattern):
  K=2: 1 + 12 = 13V (center + icosahedron)
  K=3: 1 + 3 + 12 = 16V (center + middle triad + icosahedron)  
  K=4: 1 + 3 + 12 + 36 = 52V (+ outer shell at 12×3)
  K=5: 1 + 3 + 12 + 36 + 108 = 160V (+ next shell at 36×3)

Exponential pool sizes to find crossover points.
"""

import numpy as np
import json

# =============================================================================
# CRYSTAL GEOMETRY GENERATION
# =============================================================================

def icosahedron_vertices():
    """Return 12 icosahedron vertices (unit sphere)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    # Normalize to unit sphere
    norm = np.sqrt(1 + phi**2)
    return [(x/norm, y/norm, z/norm) for x, y, z in verts]


def triad_vertices(radius=0.5):
    """Return 3 vertices forming equilateral triangle in xy-plane."""
    return [
        (radius, 0, 0),
        (-radius/2, radius * np.sqrt(3)/2, 0),
        (-radius/2, -radius * np.sqrt(3)/2, 0)
    ]


def scaled_icosahedron(radius):
    """Return 12 icosahedron vertices at given radius."""
    return [(x*radius, y*radius, z*radius) for x, y, z in icosahedron_vertices()]


def crystal_geometry(k: int):
    """
    Generate K-layer crystal geometry.
    
    K=2: 1 + 12 = 13V
    K=3: 1 + 3 + 12 = 16V
    K=4: 1 + 3 + 12 + 36 = 52V
    K=5: 1 + 3 + 12 + 36 + 108 = 160V
    
    Pattern: center(1) + triad(3) + icosa(12) + 12×3 + 36×3 + ...
    """
    if k < 2:
        raise ValueError("K must be >= 2")
    
    # Build layer structure
    layers = [(0, 0, 0)]  # Layer 0: center (1 vertex)
    
    if k >= 3:
        # Layer 1: triad (3 vertices) at radius 0.3
        layers.extend(triad_vertices(0.3))
    
    # Layer 2: icosahedron (12 vertices) at radius 0.6
    layers.extend(scaled_icosahedron(0.6))
    
    if k >= 4:
        # Layer 3: 36 vertices (3 × 12) at radius 0.85
        # Use Fibonacci sphere for even distribution
        layers.extend(fibonacci_sphere(36, 0.85))
    
    if k >= 5:
        # Layer 4: 108 vertices (3 × 36) at radius 1.0
        layers.extend(fibonacci_sphere(108, 1.0))
    
    if k >= 6:
        # Layer 5: 324 vertices (3 × 108) at radius 1.15
        layers.extend(fibonacci_sphere(324, 1.15))
    
    v = len(layers)
    adj = {i: [] for i in range(v)}
    
    # Connect based on distance threshold (adaptive per layer)
    def dist(a, b):
        return np.sqrt(sum((layers[a][i] - layers[b][i])**2 for i in range(3)))
    
    # Center connects to all
    for i in range(1, v):
        adj[0].append(i)
        adj[i].append(0)
    
    # Other connections: each vertex to nearest neighbors
    # Target degree ~6 for good connectivity
    for i in range(1, v):
        dists = [(dist(i, j), j) for j in range(1, v) if j != i]
        dists.sort()
        for _, j in dists[:6]:
            if j not in adj[i]:
                adj[i].append(j)
                adj[j].append(i)
    
    # Dedupe
    for i in range(v):
        adj[i] = list(set(adj[i]))
    
    return v, adj


def fibonacci_sphere(n, radius):
    """Generate n points evenly distributed on sphere of given radius."""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2 if n > 1 else 0
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append((x * radius, y * radius, z * radius))
    
    return points


# =============================================================================
# NETWORK SIMULATION
# =============================================================================

def build_network(frozen_adj, pool_size, contact_ratio=0.1):
    n_frozen = len(frozen_adj)
    n_total = n_frozen + pool_size
    coherence = np.ones(n_total)
    full_adj = {i: list(frozen_adj[i]) for i in range(n_frozen)}
    
    for i in range(n_frozen, n_total):
        full_adj[i] = []
    
    n_contacts = max(1, int(pool_size * contact_ratio))
    for i in range(n_frozen, n_total):
        contacts = np.random.choice(n_frozen, size=min(n_contacts, n_frozen), replace=False)
        for c in contacts:
            full_adj[i].append(c)
            full_adj[c].append(i)
    
    fluid_nodes = list(range(n_frozen, n_total))
    for _ in range(pool_size * 2):
        if len(fluid_nodes) >= 2:
            i, j = np.random.choice(fluid_nodes, size=2, replace=False)
            if j not in full_adj[i]:
                full_adj[i].append(j)
                full_adj[j].append(i)
    
    return coherence, full_adj


def measure_recovery(frozen_adj, pool_size, corruption=0.5, max_steps=200, target=0.9):
    n_frozen = len(frozen_adj)
    coherence, full_adj = build_network(frozen_adj, pool_size)
    
    frozen_mask = np.zeros(len(coherence), dtype=bool)
    frozen_mask[:n_frozen] = True
    
    fluid_idx = np.arange(n_frozen, len(coherence))
    n_corrupt = int(len(fluid_idx) * corruption)
    corrupt_idx = np.random.choice(fluid_idx, size=n_corrupt, replace=False)
    coherence[corrupt_idx] = np.random.uniform(0, 0.5, size=n_corrupt)
    
    initial = np.mean(coherence)
    
    for step in range(max_steps):
        if np.mean(coherence) >= target:
            return (np.mean(coherence) - initial) / (step + 1), step + 1
        new_c = coherence.copy()
        for i in range(len(coherence)):
            if not frozen_mask[i] and full_adj[i]:
                new_c[i] = 0.7 * coherence[i] + 0.3 * np.mean([coherence[j] for j in full_adj[i]])
        coherence = new_c
    
    return (np.mean(coherence) - initial) / max_steps, max_steps


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':
    # Exponential pool sizes
    POOLS = [50, 100, 200, 500, 1000, 2000, 5000]
    TRIALS = 5
    K_RANGE = [2, 3, 4, 5]
    
    results = {}
    
    print("="*70)
    print("CRYSTAL LAYER TEST — Finding optimal pool boundaries")
    print("="*70)
    
    # Expected vertex counts
    v_expected = {2: 13, 3: 16, 4: 52, 5: 160}
    
    for k in K_RANGE:
        v, adj = crystal_geometry(k)
        print(f"\nK={k}: {v}V (center degree: {len(adj[0])})")
        print("-" * 50)
        
        results[k] = {'vertices': v, 'pools': {}}
        
        for pool in POOLS:
            # Skip if seed > pool (meaningless test)
            if v > pool * 0.5:
                print(f"  Pool {pool:>6}: SKIP (seed too large)")
                continue
                
            rates = []
            steps_list = []
            
            for _ in range(TRIALS):
                rate, steps = measure_recovery(adj, pool)
                rates.append(rate)
                steps_list.append(steps)
            
            avg_rate = np.mean(rates)
            avg_steps = np.mean(steps_list)
            
            results[k]['pools'][pool] = {
                'rate': avg_rate,
                'steps': avg_steps
            }
            
            print(f"  Pool {pool:>6}: rate={avg_rate:.4f}  →90%={avg_steps:.1f}")
    
    # Find crossover points
    print("\n" + "="*70)
    print("CROSSOVER ANALYSIS — When does K+1 beat K?")
    print("="*70)
    
    for pool in POOLS:
        best_k = None
        best_rate = -1
        
        for k in K_RANGE:
            if pool in results[k]['pools']:
                rate = results[k]['pools'][pool]['rate']
                if rate > best_rate:
                    best_rate = rate
                    best_k = k
        
        if best_k:
            print(f"Pool {pool:>6}: optimal K={best_k} ({results[best_k]['vertices']}V), rate={best_rate:.4f}")
    
    # Save results
    with open('/Users/nick/Projects/holocell/crystal_layer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to crystal_layer_results.json")

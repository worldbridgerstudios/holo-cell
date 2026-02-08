#!/usr/bin/env python3
"""
Shell Scaling Test — Testing the ×3 Transition Hypothesis

Uses antiprism graphs for equanimous geometry at any scale.
n-antiprism = 2n vertices, degree 4, vertex-transitive, all triangular faces.

Tests: does optimal geometry scale by factor of 3 at shell transitions?
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple

# =============================================================================
# CENTERED ICOSAHEDRON — 13V (12 outer + 1 core)
# =============================================================================

def centered_icosahedron() -> Tuple[int, Dict[int, List[int]]]:
    """
    Icosahedron with central point.
    
    Structure:
      - 12 outer vertices (icosahedron)
      - 1 center (the One / ayin point)
    
    Total: 13 vertices
    
    Center connects to all 12 outer vertices.
    Outer vertices have standard icosahedral connectivity (degree 5).
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    # Three orthogonal golden rectangles
    outer = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    
    v = 13  # 12 outer + 1 center
    adj = {i: [] for i in range(v)}
    
    # Center (vertex 0) connects to all outer vertices (1-12)
    for i in range(1, 13):
        adj[0].append(i)
        adj[i].append(0)
    
    # Icosahedral edges between outer vertices
    # Two vertices are adjacent if distance ≈ 2 (edge length)
    def dist(a, b):
        return sum((a[i] - b[i])**2 for i in range(3))**0.5
    
    edge_length = 2.0  # For standard icosahedron
    tolerance = 0.01
    
    for i in range(12):
        for j in range(i+1, 12):
            d = dist(outer[i], outer[j])
            if abs(d - edge_length) < tolerance:
                adj[i+1].append(j+1)  # +1 because center is 0
                adj[j+1].append(i+1)
    
    return v, adj


# =============================================================================
# VORTEX MERKABAH — Bilateral ×3 encoding
# =============================================================================

def vortex_merkabah(n_layers: int = 2) -> Tuple[int, Dict[int, List[int]]]:
    """
    Bilateral vortex structure — merkabah geometry.
    
    Structure:
      - 1 center (the One / ayin point)
      - 2 sides (A and B, inverted)
      - Each side has n_layers of 3 nodes each
    
    Total vertices: 1 + 2 × (3 × n_layers) = 1 + 6n
      n=1: 7V  (center + 3 + 3)
      n=2: 13V (center + 3+3 + 3+3)
      n=3: 19V
      n=4: 25V
    
    Encodes ×3 at each layer with bilateral symmetry.
    Two vortices meet only at center — the covenant point.
    """
    v = 1 + 6 * n_layers
    adj = {i: [] for i in range(v)}
    
    # Vertex layout:
    # 0: center
    # Side A: 1..3*n_layers (layers of 3)
    # Side B: 3*n_layers+1..6*n_layers (layers of 3)
    
    side_a_start = 1
    side_b_start = 1 + 3 * n_layers
    
    def triangle_edges(base):
        """Connect 3 nodes as triangle starting at base."""
        adj[base].extend([base+1, base+2])
        adj[base+1].extend([base, base+2])
        adj[base+2].extend([base, base+1])
    
    def interlocking_edges(inner_base, outer_base):
        """Connect inner triangle to outer triangle (rotated 60°)."""
        # Each inner node connects to 2 outer nodes
        adj[inner_base].extend([outer_base, outer_base+2])
        adj[inner_base+1].extend([outer_base, outer_base+1])
        adj[inner_base+2].extend([outer_base+1, outer_base+2])
        # Reverse connections
        adj[outer_base].extend([inner_base, inner_base+1])
        adj[outer_base+1].extend([inner_base+1, inner_base+2])
        adj[outer_base+2].extend([inner_base, inner_base+2])
    
    # Center connects to innermost layer of both sides
    for i in range(3):
        adj[0].append(side_a_start + i)
        adj[side_a_start + i].append(0)
        adj[0].append(side_b_start + i)
        adj[side_b_start + i].append(0)
    
    # Build each side
    for side_start in [side_a_start, side_b_start]:
        for layer in range(n_layers):
            layer_base = side_start + layer * 3
            
            # Triangle within layer
            triangle_edges(layer_base)
            
            # Connect to next layer (if exists)
            if layer < n_layers - 1:
                next_base = side_start + (layer + 1) * 3
                interlocking_edges(layer_base, next_base)
    
    # Remove duplicates
    for i in range(v):
        adj[i] = list(set(adj[i]))
    
    return v, adj


# =============================================================================
# ANTIPRISM GEOMETRY GENERATION
# =============================================================================

def antiprism(n: int) -> Tuple[int, Dict[int, List[int]]]:
    """
    Generate n-antiprism: 2n vertices, degree 4, vertex-transitive.
    
    Two n-gons (top and bottom), twisted by π/n, connected by 2n triangles.
    
    Vertex layout:
      0..n-1: top n-gon
      n..2n-1: bottom n-gon
    
    Special cases:
      3-antiprism = octahedron
      4-antiprism = square antiprism
    """
    if n < 3:
        raise ValueError("Antiprism requires n >= 3")
    
    v = 2 * n
    adj = {i: [] for i in range(v)}
    
    # Top n-gon edges (0 to n-1)
    for i in range(n):
        adj[i].append((i + 1) % n)
        adj[i].append((i - 1) % n)
    
    # Bottom n-gon edges (n to 2n-1)
    for i in range(n, 2*n):
        adj[i].append(n + (i - n + 1) % n)
        adj[i].append(n + (i - n - 1) % n)
    
    # Triangular connections between top and bottom
    # Each top vertex i connects to bottom vertices i and i+1 (mod n)
    for i in range(n):
        bottom1 = n + i
        bottom2 = n + (i + 1) % n
        
        adj[i].append(bottom1)
        adj[i].append(bottom2)
        adj[bottom1].append(i)
        adj[bottom2].append(i)
    
    # Remove duplicates
    for i in range(v):
        adj[i] = list(set(adj[i]))
    
    return v, adj


def prism(n: int) -> Tuple[int, Dict[int, List[int]]]:
    """
    Generate n-prism: 2n vertices, degree 3, vertex-transitive.
    
    Two n-gons (top and bottom), aligned, connected by n squares.
    
    Special cases:
      3-prism = triangular prism
      4-prism = cube
    """
    if n < 3:
        raise ValueError("Prism requires n >= 3")
    
    v = 2 * n
    adj = {i: [] for i in range(v)}
    
    # Top n-gon edges (0 to n-1)
    for i in range(n):
        adj[i].append((i + 1) % n)
        adj[i].append((i - 1) % n)
    
    # Bottom n-gon edges (n to 2n-1)
    for i in range(n, 2*n):
        adj[i].append(n + (i - n + 1) % n)
        adj[i].append(n + (i - n - 1) % n)
    
    # Vertical connections (each top vertex to corresponding bottom)
    for i in range(n):
        adj[i].append(n + i)
        adj[n + i].append(i)
    
    return v, adj


# =============================================================================
# NETWORK SIMULATION
# =============================================================================

def build_network(frozen_adj: Dict[int, List[int]], pool_size: int, 
                  contact_ratio: float = 0.1) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Build network with frozen seed + fluid pool."""
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
    n_fluid_edges = pool_size * 2
    for _ in range(n_fluid_edges):
        if len(fluid_nodes) >= 2:
            i, j = np.random.choice(fluid_nodes, size=2, replace=False)
            if j not in full_adj[i]:
                full_adj[i].append(j)
                full_adj[j].append(i)
    
    return coherence, full_adj


def corrupt_network(coherence: np.ndarray, n_frozen: int, corruption: float) -> np.ndarray:
    """Corrupt fluid nodes."""
    result = coherence.copy()
    fluid_indices = np.arange(n_frozen, len(coherence))
    n_corrupt = int(len(fluid_indices) * corruption)
    corrupt_indices = np.random.choice(fluid_indices, size=n_corrupt, replace=False)
    result[corrupt_indices] = np.random.uniform(0, 0.5, size=n_corrupt)
    return result


def healing_step(coherence: np.ndarray, adj: Dict[int, List[int]], 
                 frozen_mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """One step of coherence propagation."""
    new_coherence = coherence.copy()
    
    for i in range(len(coherence)):
        if frozen_mask[i]:
            continue
        neighbors = adj[i]
        if neighbors:
            neighbor_avg = np.mean([coherence[j] for j in neighbors])
            new_coherence[i] = (1 - alpha) * coherence[i] + alpha * neighbor_avg
    
    return new_coherence


def measure_recovery(frozen_adj: Dict[int, List[int]], pool_size: int,
                     corruption: float = 0.5, max_steps: int = 200,
                     target: float = 0.9) -> Tuple[float, int]:
    """
    Measure coherence recovery rate.
    Returns (rate, steps_to_target).
    """
    n_frozen = len(frozen_adj)
    coherence, full_adj = build_network(frozen_adj, pool_size)
    
    frozen_mask = np.zeros(len(coherence), dtype=bool)
    frozen_mask[:n_frozen] = True
    
    coherence = corrupt_network(coherence, n_frozen, corruption)
    initial_coherence = np.mean(coherence)
    
    for step in range(max_steps):
        current = np.mean(coherence)
        if current >= target:
            rate = (current - initial_coherence) / (step + 1) if step > 0 else 0
            return rate, step + 1
        coherence = healing_step(coherence, full_adj, frozen_mask)
    
    final = np.mean(coherence)
    rate = (final - initial_coherence) / max_steps
    return rate, max_steps


# =============================================================================
# SHELL SCALING TEST
# =============================================================================

def test_shell_sequence(base_n: int, n_shells: int, pool_sizes: List[int],
                        trials: int = 5, geometry_type: str = 'antiprism') -> Dict:
    """
    Test a sequence of geometries scaling by ×3.
    
    Args:
        base_n: Starting n for n-antiprism (vertices = 2*base_n)
        n_shells: Number of shells to test
        pool_sizes: List of pool sizes to test
        trials: Trials per configuration
        geometry_type: 'antiprism' or 'prism'
    
    Returns:
        Results dictionary
    """
    gen_func = antiprism if geometry_type == 'antiprism' else prism
    
    # Generate shell sequence: n, 3n, 9n, 27n, ...
    shell_ns = [base_n * (3 ** i) for i in range(n_shells)]
    
    results = {
        'base_n': base_n,
        'geometry_type': geometry_type,
        'shells': [],
        'pool_sizes': pool_sizes,
        'trials': trials
    }
    
    print(f"\n{'='*70}")
    print(f"SHELL SEQUENCE: {geometry_type} starting at n={base_n}")
    print(f"Shells: {shell_ns} → Vertices: {[2*n for n in shell_ns]}")
    print(f"{'='*70}\n")
    
    for n in shell_ns:
        v, adj = gen_func(n)
        
        print(f"{n}-{geometry_type} ({v} vertices)")
        print("-" * 50)
        
        shell_data = {
            'n': n,
            'vertices': v,
            'pool_results': {}
        }
        
        for pool in pool_sizes:
            rates = []
            steps_list = []
            
            for _ in range(trials):
                rate, steps = measure_recovery(adj, pool)
                rates.append(rate)
                steps_list.append(steps)
            
            avg_rate = np.mean(rates)
            avg_steps = np.mean(steps_list)
            
            shell_data['pool_results'][pool] = {
                'avg_rate': avg_rate,
                'avg_steps': avg_steps,
                'rates': rates,
                'steps': steps_list
            }
            
            print(f"  Pool {pool:>6}: rate={avg_rate:.4f}  →90%={avg_steps:.1f}")
        
        # Average across pools
        all_rates = [shell_data['pool_results'][p]['avg_rate'] for p in pool_sizes]
        shell_data['overall_rate'] = np.mean(all_rates)
        
        results['shells'].append(shell_data)
        print(f"  → Overall rate: {shell_data['overall_rate']:.4f}\n")
    
    return results


def find_optimal_shell(results: Dict, pool_size: int) -> Tuple[int, float]:
    """Find which shell is optimal for a given pool size."""
    best_n = None
    best_rate = -1
    
    for shell in results['shells']:
        rate = shell['pool_results'][pool_size]['avg_rate']
        if rate > best_rate:
            best_rate = rate
            best_n = shell['n']
    
    return best_n, best_rate


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Test parameters
    POOL_SIZES = [100, 300, 1000, 3000, 10000]
    TRIALS = 5
    N_SHELLS = 4
    
    all_results = {}
    
    # Test both geometry types
    for geom_type in ['antiprism', 'prism']:
        for base_n in [3, 4, 5, 6]:
            results = test_shell_sequence(
                base_n=base_n,
                n_shells=N_SHELLS,
                pool_sizes=POOL_SIZES,
                trials=TRIALS,
                geometry_type=geom_type
            )
            all_results[f'{geom_type}_{base_n}'] = results
    
    # Summary: optimal shell per pool size
    print("\n" + "="*70)
    print("OPTIMAL SHELL BY POOL SIZE")
    print("="*70)
    
    for key, results in sorted(all_results.items()):
        geom_type = results['geometry_type']
        base_n = results['base_n']
        print(f"\n{key}:")
        for pool in POOL_SIZES:
            opt_n, opt_rate = find_optimal_shell(results, pool)
            print(f"  Pool {pool:>6}: optimal = {opt_n}-{geom_type} ({2*opt_n}V), rate={opt_rate:.4f}")
    
    # Save results
    output_file = '/Users/nick/Projects/holocell/shell_scaling_results.json'
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

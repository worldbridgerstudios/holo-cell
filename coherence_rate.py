#!/usr/bin/env python3
"""
Coherence Rate Comparison — Which seed resonates fastest?

Test 3 pool sizes (100, 1000, 3000 fluid nodes)
Measure recovery RATE, not ceiling.
Compare: tetrahedron (4V), buckyball (60V), holocrystal (136V)
"""

import random
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

PHI = (1 + math.sqrt(5)) / 2

# =============================================================================
# SEED GEOMETRIES
# =============================================================================

def tetrahedron() -> Tuple[int, Dict[int, List[int]]]:
    """4 vertices, each connected to 3 others."""
    adj = {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2],
    }
    return 4, adj


def buckyball() -> Tuple[int, Dict[int, List[int]]]:
    """Truncated icosahedron: 60 vertices, degree 3."""
    vertices = []
    
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0, 0, 0]
                v[perm[1]] = s1 * 1
                v[perm[2]] = s2 * 3 * PHI
                vertices.append(tuple(v))
    
    a, b, c = 1, 2+PHI, 2*PHI
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    vertices.append((s1*[a,b,c][perm[0]], s2*[a,b,c][perm[1]], s3*[a,b,c][perm[2]]))
    
    a, b, c = PHI, 2, 2*PHI+1
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    vertices.append((s1*[a,b,c][perm[0]], s2*[a,b,c][perm[1]], s3*[a,b,c][perm[2]]))
    
    # Dedupe
    unique = []
    for v in vertices:
        if not any(sum((a-b)**2 for a,b in zip(v,u)) < 0.01 for u in unique):
            unique.append(v)
    vertices = unique[:60]
    
    # Build adjacency from shortest edges
    n = len(vertices)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = sum((a-b)**2 for a,b in zip(vertices[i], vertices[j]))**0.5
            dists.append((d, i, j))
    dists.sort()
    min_d = dists[0][0]
    
    adj = {i: [] for i in range(n)}
    for d, i, j in dists:
        if d <= min_d * 1.01:
            adj[i].append(j)
            adj[j].append(i)
    
    return 60, adj


def holocrystal() -> Tuple[int, Dict[int, List[int]]]:
    """
    T(16) = 136 vertices.
    
    The holocrystal: triangular number 16.
    Build as layers of a triangular lattice, each layer a triangle.
    """
    # T(16) = 1+2+3+...+16 = 136
    # Arrange as triangular layers
    vertices = []
    layer_start = {}
    idx = 0
    
    for layer in range(16):
        layer_start[layer] = idx
        for pos in range(layer + 1):
            # Triangular grid coordinates
            x = pos - layer/2
            y = layer * math.sqrt(3)/2
            z = 0  # Flat for now
            vertices.append((x, y, z))
            idx += 1
    
    n = 136
    adj = {i: [] for i in range(n)}
    
    # Connect within layers (horizontal neighbors)
    for layer in range(16):
        start = layer_start[layer]
        for pos in range(layer + 1):
            i = start + pos
            # Right neighbor in same layer
            if pos < layer:
                j = start + pos + 1
                adj[i].append(j)
                adj[j].append(i)
    
    # Connect between layers (diagonal neighbors)
    for layer in range(1, 16):
        curr_start = layer_start[layer]
        prev_start = layer_start[layer - 1]
        
        for pos in range(layer + 1):
            i = curr_start + pos
            # Connect to up-left (if exists)
            if pos > 0:
                j = prev_start + pos - 1
                adj[i].append(j)
                adj[j].append(i)
            # Connect to up-right (if exists)
            if pos < layer:
                j = prev_start + pos
                adj[i].append(j)
                adj[j].append(i)
    
    return 136, adj


def vortex_engine() -> Tuple[int, Dict[int, List[int]]]:
    """
    Vortex Engine: 16-position wheel × 9-position spine = 144 nodes
    Toroidal topology with two fundamental cycles.
    """
    wheel = 16  # positions around torus major circle
    spine = 9   # positions along torus tube
    n = wheel * spine  # 144
    
    adj = {i: [] for i in range(n)}
    
    for s in range(spine):
        for w in range(wheel):
            i = s * wheel + w
            
            # Connect around wheel (ring at each spine position)
            j = s * wheel + (w + 1) % wheel
            adj[i].append(j)
            adj[j].append(i)
            
            # Connect along spine (between rings)
            if s < spine - 1:
                j = (s + 1) * wheel + w
                adj[i].append(j)
                adj[j].append(i)
    
    # Close the torus: connect last spine ring to first
    for w in range(wheel):
        i = (spine - 1) * wheel + w
        j = w
        adj[i].append(j)
        adj[j].append(i)
    
    return n, adj


def octahedron() -> Tuple[int, Dict[int, List[int]]]:
    """
    Octahedron: 6 vertices, degree 4 each.
    The TRUE HoloCell geometry - minimal Platonic with 3-axis symmetry.
    3 bilateral pairs on Trinition axes (i, j, k).
    """
    # Vertices at ±1 on each axis
    # 0,1 = X pair; 2,3 = Y pair; 4,5 = Z pair
    adj = {
        0: [2, 3, 4, 5],  # +X connects to all Y and Z
        1: [2, 3, 4, 5],  # -X connects to all Y and Z
        2: [0, 1, 4, 5],  # +Y connects to all X and Z
        3: [0, 1, 4, 5],  # -Y connects to all X and Z
        4: [0, 1, 2, 3],  # +Z connects to all X and Y
        5: [0, 1, 2, 3],  # -Z connects to all X and Y
    }
    return 6, adj


def holocell() -> Tuple[int, Dict[int, List[int]]]:
    """
    HoloCell: Central seed + octahedral shell = 7 vertices.
    Node 0 = center (T(16)=136 eigenvalue encoded)
    Nodes 1-6 = octahedral vertices (3 bilateral pairs)
    """
    adj = {
        0: [1, 2, 3, 4, 5, 6],  # Center connects to all 6
        1: [0, 3, 4, 5, 6],     # +X connects to center, Y, Z
        2: [0, 3, 4, 5, 6],     # -X connects to center, Y, Z
        3: [0, 1, 2, 5, 6],     # +Y connects to center, X, Z
        4: [0, 1, 2, 5, 6],     # -Y connects to center, X, Z
        5: [0, 1, 2, 3, 4],     # +Z connects to center, X, Y
        6: [0, 1, 2, 3, 4],     # -Z connects to center, X, Y
    }
    return 7, adj


SEEDS = {
    'tetrahedron': tetrahedron,
    'octahedron': octahedron,
    'holocell': holocell,
    'buckyball': buckyball,
    'vortex_engine': vortex_engine,
}


# =============================================================================
# NETWORK & SIMULATION
# =============================================================================

class Network:
    def __init__(self, adj: Dict[int, List[int]], frozen: List[int]):
        self.values = {i: 1.0 for i in adj}
        self.adj = adj
        self.frozen = set(frozen)
    
    def corrupt(self, frac: float):
        fluid = [i for i in self.values if i not in self.frozen]
        n = max(1, int(len(fluid) * frac))
        for i in random.sample(fluid, min(n, len(fluid))):
            self.values[i] = random.uniform(-2, 4)
    
    def step(self):
        updates = {}
        for i in self.values:
            if i in self.frozen:
                continue
            neighbors = [j for j in self.adj.get(i, []) if j in self.values]
            if neighbors:
                avg = sum(self.values[j] for j in neighbors) / len(neighbors)
                updates[i] = self.values[i] + 0.5 * (avg - self.values[i])
        for i, v in updates.items():
            self.values[i] = v
    
    def coherence(self) -> float:
        """Average coherence of fluid nodes."""
        fluid = [i for i in self.values if i not in self.frozen]
        if not fluid:
            return 1.0
        cohs = [max(0, 1 - abs(self.values[i] - 1)) for i in fluid]
        return sum(cohs) / len(cohs)


def build_shell_network(seed_adj: Dict[int, List[int]], n_frozen: int, n_fluid: int) -> Tuple[Dict[int, List[int]], List[int]]:
    """Frozen seed as shell, fluid nodes inside with sparse contact."""
    total = n_frozen + n_fluid
    adj = {i: list(seed_adj.get(i, [])) for i in range(n_frozen)}
    
    n_layers = max(1, int(n_fluid ** 0.5))
    per_layer = max(1, n_fluid // n_layers)
    
    for i in range(n_frozen, total):
        adj[i] = []
        fluid_idx = i - n_frozen
        layer = min(fluid_idx // per_layer, n_layers - 1)
        
        layer_start = n_frozen + layer * per_layer
        layer_end = min(layer_start + per_layer, total)
        
        # Ring within layer
        if i > layer_start:
            adj[i].append(i-1)
            adj[i-1].append(i)
        if i == layer_end - 1 and layer_end - layer_start > 2:
            adj[i].append(layer_start)
            adj[layer_start].append(i)
        
        # Connect to previous layer
        if layer > 0:
            prev_start = n_frozen + (layer-1) * per_layer
            prev_idx = prev_start + (fluid_idx % per_layer)
            if prev_idx < layer_start and prev_idx in adj:
                adj[i].append(prev_idx)
                adj[prev_idx].append(i)
        
        # Sparse shell contact (only layer 0)
        if layer == 0:
            c = fluid_idx % n_frozen
            adj[i].append(c)
            adj[c].append(i)
    
    return adj, list(range(n_frozen))


# =============================================================================
# RATE MEASUREMENT
# =============================================================================

@dataclass
class RateResult:
    seed: str
    n_frozen: int
    n_fluid: int
    initial_coherence: float
    final_coherence: float
    recovery_rate: float  # Δcoherence per step (early phase)
    steps_to_90pct: int


def measure_recovery_rate(seed_name: str, n_fluid: int, trials: int = 3) -> RateResult:
    """Measure coherence recovery rate for a seed/pool combination."""
    
    n_frozen, seed_adj = SEEDS[seed_name]()
    
    all_rates = []
    all_steps_to_90 = []
    all_initial = []
    all_final = []
    
    for _ in range(trials):
        adj, frozen = build_shell_network(seed_adj, n_frozen, n_fluid)
        net = Network(adj, frozen)
        
        # Corrupt 50%
        net.corrupt(0.5)
        initial = net.coherence()
        all_initial.append(initial)
        
        # Track coherence over time
        history = [initial]
        steps_to_90 = -1
        
        for step in range(100):
            net.step()
            coh = net.coherence()
            history.append(coh)
            
            if steps_to_90 < 0 and coh >= 0.9:
                steps_to_90 = step + 1
        
        all_final.append(history[-1])
        all_steps_to_90.append(steps_to_90 if steps_to_90 > 0 else 100)
        
        # Calculate rate from first 10 steps (early recovery phase)
        early_rate = (history[10] - history[0]) / 10 if len(history) > 10 else 0
        all_rates.append(early_rate)
    
    return RateResult(
        seed=seed_name,
        n_frozen=n_frozen,
        n_fluid=n_fluid,
        initial_coherence=sum(all_initial) / len(all_initial),
        final_coherence=sum(all_final) / len(all_final),
        recovery_rate=sum(all_rates) / len(all_rates),
        steps_to_90pct=int(sum(all_steps_to_90) / len(all_steps_to_90)),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("COHERENCE RATE COMPARISON — Which Seed Resonates Fastest?")
    print("=" * 70)
    print()
    
    pool_sizes = [100, 1000, 3000]
    seeds = ['tetrahedron', 'octahedron', 'holocell', 'buckyball', 'vortex_engine']
    
    results: Dict[str, List[RateResult]] = {s: [] for s in seeds}
    
    for seed in seeds:
        n_frozen, _ = SEEDS[seed]()
        print(f"{seed} ({n_frozen} frozen vertices)")
        print("-" * 50)
        print(f"{'Pool':<8} {'Rate':>10} {'→90%':>8} {'Final':>10}")
        
        for n_fluid in pool_sizes:
            r = measure_recovery_rate(seed, n_fluid)
            results[seed].append(r)
            print(f"{n_fluid:<8} {r.recovery_rate:>10.4f} {r.steps_to_90pct:>6} steps {r.final_coherence:>8.1%}")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY — Average Recovery Rate Across Pool Sizes")
    print("=" * 70)
    
    summary = []
    for seed in seeds:
        avg_rate = sum(r.recovery_rate for r in results[seed]) / len(results[seed])
        avg_steps = sum(r.steps_to_90pct for r in results[seed]) / len(results[seed])
        n_frozen, _ = SEEDS[seed]()
        summary.append((seed, n_frozen, avg_rate, avg_steps))
    
    # Sort by rate (descending)
    summary.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Seed':<15} {'Frozen':>8} {'Avg Rate':>12} {'Avg →90%':>12}")
    print("-" * 50)
    for seed, n_frozen, avg_rate, avg_steps in summary:
        print(f"{seed:<15} {n_frozen:>8} {avg_rate:>12.4f} {avg_steps:>10.1f} steps")
    
    print()
    print(f"🏆 FASTEST RESONATOR: {summary[0][0]}")
    print()


if __name__ == "__main__":
    main()

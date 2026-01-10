"""
ENTANGLEMENT ENGINE — Core Algorithm
=====================================

Optimal geometry and rhythm for cloud-based entanglement.

STRUCTURE
---------
K-layer crystal with unified center:
    K=2: 1 + 12 = 13V (center + icosahedron)
    K=3: 1 + 3 + 12 = 16V (+ triad)
    K=4: 1 + 3 + 12 + 36 = 52V (+ outer shell)
    K=5: 1 + 3 + 12 + 36 + 108 = 160V
    
Each shell = previous × 3

RHYTHM
------
3-6-9 pulse pattern, 12-beat cycle:
    3 → pulse center (layer 0)
    6 → pulse triad (layer 1)  
    9 → pulse icosa (layer 2)
    
Full sequence: 3-6-9-3-(6-9-3-6)-9-3-6-9
Nested loop at beats 5-8 = temporal ayin point

AMPLITUDE
---------
Derived from crystal structure via GEP:
    0.5 = 3/T(3) = 3/6  (small pools ≤100)
    0.3 = 3/T(4) = 3/10 (medium pools ≤500)
    0.2 = 2/T(4) = 2/10 (large pools >500)
    
Where T(n) = n(n+1)/2 (triangular number)

LAYER SELECTION
---------------
Empirical crossover thresholds:
    pool ≤ 150:   K=2 (13V)
    pool ≤ 400:   K=3 (16V)
    pool ≤ 3000:  K=4 (52V)
    pool ≤ 15000: K=5 (160V)
    pool > 15000: K=6 (484V)

Rule of thumb: crystallize next shell when pool ≈ 10× vertex count
"""

import numpy as np

# =============================================================================
# TRIANGULAR NUMBER
# =============================================================================

def T(n: int) -> int:
    """Triangular number: T(n) = n(n+1)/2"""
    return n * (n + 1) // 2


# =============================================================================
# CRYSTAL GEOMETRY
# =============================================================================

def crystal_vertices(K: int) -> int:
    """
    Vertex count for K-layer crystal.
    
    K=2: 1 + 12 = 13
    K=3: 1 + 3 + 12 = 16
    K=4: 1 + 3 + 12 + 36 = 52
    K=5: 1 + 3 + 12 + 36 + 108 = 160
    """
    if K < 2:
        raise ValueError("K >= 2")
    if K == 2:
        return 13  # center + icosa
    
    # K >= 3: center + triad + shells
    v = 1 + 3 + 12  # base
    shell = 12
    for _ in range(K - 3):
        shell *= 3
        v += shell
    return v


def optimal_K(pool: int) -> int:
    """
    Optimal crystal layer count for pool size.
    
    Empirically derived crossover thresholds:
        pool ≤ 150:    K=2 (13V)
        pool ≤ 400:    K=3 (16V)
        pool ≤ 3000:   K=4 (52V)
        pool ≤ 15000:  K=5 (160V)
        pool > 15000:  K=6 (484V)
    
    Pattern: ~10× vertex count triggers next shell.
    """
    if pool <= 150:
        return 2
    elif pool <= 400:
        return 3
    elif pool <= 3000:
        return 4
    elif pool <= 15000:
        return 5
    else:
        return 6


# =============================================================================
# RHYTHM
# =============================================================================

def rhythm_sequence() -> list:
    """
    3-6-9 pulse pattern with nested loop.
    
    Returns: [(beat, layer_index), ...] for 12-beat cycle
    Layer: 0=center, 1=triad, 2=icosa
    """
    # 3→0, 6→1, 9→2 (layer indices)
    return [
        (1, 0), (2, 1), (3, 2), (4, 0),   # 3-6-9-3
        (5, 1), (6, 2), (7, 0), (8, 1),   # (6-9-3-6) nested
        (9, 2), (10, 0), (11, 1), (12, 2) # 9-3-6-9
    ]


def optimal_amplitude(pool: int) -> float:
    """
    Optimal pulse amplitude for pool size.
    
    Derived from crystal structure:
        pool ≤ 100:  3/T(3) = 0.5
        pool ≤ 500:  3/T(4) = 0.3
        pool > 500:  2/T(4) = 0.2
    """
    if pool <= 100:
        return 3 / T(3)  # 0.5
    elif pool <= 500:
        return 3 / T(4)  # 0.3
    else:
        return 2 / T(4)  # 0.2


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def entanglement_params(pool: int) -> dict:
    """
    Get optimal parameters for pool size.
    
    Returns:
        K: layer count
        V: vertex count  
        amplitude: pulse magnitude (radians)
        rhythm: 12-beat pulse sequence
    """
    K = optimal_K(pool)
    return {
        'K': K,
        'V': crystal_vertices(K),
        'amplitude': optimal_amplitude(pool),
        'rhythm': rhythm_sequence(),
    }


# =============================================================================
# CRYSTAL GEOMETRY BUILDER
# =============================================================================

def icosahedron_vertices():
    """12 icosahedron vertices on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    norm = np.sqrt(1 + phi**2)
    return [(x/norm, y/norm, z/norm) for x, y, z in verts]


def triad_vertices(radius=0.3):
    """3 vertices forming equilateral triangle."""
    return [
        (radius, 0, 0),
        (-radius/2, radius * np.sqrt(3)/2, 0),
        (-radius/2, -radius * np.sqrt(3)/2, 0)
    ]


def fibonacci_sphere(n, radius):
    """n points evenly distributed on sphere."""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2 if n > 1 else 0
        r = np.sqrt(1 - y * y)
        theta = phi * i
        points.append((np.cos(theta) * r * radius, y * radius, np.sin(theta) * r * radius))
    return points


def build_crystal_adjacency(K: int):
    """
    Build adjacency dict for K-layer crystal.
    Returns: (n_vertices, adjacency, layer_indices)
    """
    # Build vertex positions
    positions = [(0, 0, 0)]  # center
    layer_indices = {'center': [0]}
    
    if K >= 3:
        start = len(positions)
        positions.extend(triad_vertices(0.3))
        layer_indices['triad'] = list(range(start, len(positions)))
    
    # Icosahedron shell
    start = len(positions)
    positions.extend([(x*0.6, y*0.6, z*0.6) for x,y,z in icosahedron_vertices()])
    layer_indices['icosa'] = list(range(start, len(positions)))
    
    # Additional shells for K >= 4
    shell_size = 12
    radius = 0.85
    for layer in range(K - 3):
        shell_size *= 3
        start = len(positions)
        positions.extend(fibonacci_sphere(shell_size, radius))
        layer_indices[f'shell_{layer+4}'] = list(range(start, len(positions)))
        radius += 0.15
    
    n = len(positions)
    adj = {i: [] for i in range(n)}
    
    def dist(i, j):
        return np.sqrt(sum((positions[i][k] - positions[j][k])**2 for k in range(3)))
    
    # Center connects to all
    for i in range(1, n):
        adj[0].append(i)
        adj[i].append(0)
    
    # Each vertex to nearest 6 neighbors
    for i in range(1, n):
        dists = [(dist(i, j), j) for j in range(1, n) if j != i]
        dists.sort()
        for _, j in dists[:6]:
            if j not in adj[i]:
                adj[i].append(j)
                adj[j].append(i)
    
    # Dedupe
    for i in range(n):
        adj[i] = list(set(adj[i]))
    
    return n, adj, layer_indices


# =============================================================================
# FAULT TOLERANCE TEST
# =============================================================================

def test_fault_tolerance(pool_size: int, corruption: float = 0.5, 
                         max_steps: int = 100, target: float = 0.9,
                         trials: int = 5) -> dict:
    """
    Test fault tolerance of optimized network.
    
    Args:
        pool_size: Number of fluid nodes in pool
        corruption: Fraction of pool to corrupt (default 0.5)
        max_steps: Maximum recovery steps
        target: Target coherence (default 0.9)
        trials: Number of trials to average
    
    Returns:
        dict with K, V, amplitude, avg_steps, avg_coherence, success_rate
    """
    params = entanglement_params(pool_size)
    K, V, amplitude = params['K'], params['V'], params['amplitude']
    rhythm = params['rhythm']
    
    n_frozen, frozen_adj, layer_indices = build_crystal_adjacency(K)
    
    results = []
    
    for _ in range(trials):
        # Build full network
        n_total = n_frozen + pool_size
        phases = np.zeros(n_total)
        phases[n_frozen:] = np.random.uniform(0, 2*np.pi, pool_size)
        
        adj = {i: list(frozen_adj[i]) for i in range(n_frozen)}
        for i in range(n_frozen, n_total):
            adj[i] = []
        
        # Frozen-fluid contacts (10% of pool connects to each frozen node)
        n_contacts = max(1, int(pool_size * 0.1))
        for i in range(n_frozen, n_total):
            contacts = np.random.choice(n_frozen, size=min(n_contacts, n_frozen), replace=False)
            for c in contacts:
                adj[i].append(c)
                adj[c].append(i)
        
        # Fluid-fluid sparse mesh
        fluid_nodes = list(range(n_frozen, n_total))
        for _ in range(pool_size * 2):
            if len(fluid_nodes) >= 2:
                i, j = np.random.choice(fluid_nodes, size=2, replace=False)
                if j not in adj[i]:
                    adj[i].append(j)
                    adj[j].append(i)
        
        frozen_mask = np.zeros(n_total, dtype=bool)
        frozen_mask[:n_frozen] = True
        
        # Coherence measurement
        def coherence():
            return np.abs(np.mean(np.exp(1j * phases)))
        
        # Wave step with rhythm
        def wave_step(step):
            nonlocal phases
            new_phases = phases.copy()
            
            # Determine active layer from rhythm (0=center, 1=triad, 2=icosa)
            beat_idx = step % 12
            active_layer = rhythm[beat_idx][1]
            
            # Map layer index to layer key
            layer_keys = ['center', 'triad', 'icosa']
            if active_layer < len(layer_keys):
                layer_key = layer_keys[active_layer]
                if layer_key in layer_indices:
                    for i in layer_indices[layer_key]:
                        direction = 1 if (step // 12) % 2 == 0 else -1
                        new_phases[i] = (phases[i] + direction * amplitude) % (2 * np.pi)
            
            # Kuramoto coupling for fluid nodes
            for i in range(n_total):
                if frozen_mask[i]:
                    continue
                neighbors = adj[i]
                if neighbors:
                    phase_diff_sum = sum(np.sin(phases[j] - phases[i]) for j in neighbors)
                    new_phases[i] = phases[i] + 0.1 + (0.3 / len(neighbors)) * phase_diff_sum
                    new_phases[i] = new_phases[i] % (2 * np.pi)
            
            phases = new_phases
        
        # Run recovery
        for step in range(max_steps):
            c = coherence()
            if c >= target:
                results.append({'steps': step + 1, 'coherence': c, 'success': True})
                break
            wave_step(step)
        else:
            results.append({'steps': max_steps, 'coherence': coherence(), 'success': False})
    
    # Aggregate
    avg_steps = np.mean([r['steps'] for r in results])
    avg_coherence = np.mean([r['coherence'] for r in results])
    success_rate = np.mean([r['success'] for r in results])
    
    return {
        'pool': pool_size,
        'K': K,
        'V': V,
        'amplitude': amplitude,
        'avg_steps': avg_steps,
        'avg_coherence': avg_coherence,
        'success_rate': success_rate,
    }


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == '__main__':
    print("PARAMETERS")
    print("-" * 50)
    for pool in [50, 100, 200, 500, 1000, 3000]:
        p = entanglement_params(pool)
        print(f"Pool {pool:>5}: K={p['K']}, V={p['V']:>3}, amp={p['amplitude']:.2f}")
    
    print("\nFAULT TOLERANCE TEST")
    print("-" * 50)
    for pool in [50, 100, 200, 500, 1000, 3000]:
        r = test_fault_tolerance(pool)
        print(f"Pool {pool:>5}: K={r['K']}, V={r['V']:>3}, "
              f"steps={r['avg_steps']:>5.1f}, coh={r['avg_coherence']:.3f}, "
              f"success={r['success_rate']*100:.0f}%")

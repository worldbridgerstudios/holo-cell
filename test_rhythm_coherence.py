#!/usr/bin/env python3
"""
Rhythm Coherence Test — Wave-based propagation with 3-6-9 pulse pattern

Each node oscillates with phase and amplitude.
Frozen seed pulses according to rhythm pattern.
Fluid pool couples to seed and each other.
Coherence = phase alignment (Kuramoto order parameter).

Find optimal amplitude for fastest synchronization.
"""

import numpy as np
import json

# =============================================================================
# CRYSTAL GEOMETRY (from test_crystal_layers.py)
# =============================================================================

def icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    norm = np.sqrt(1 + phi**2)
    return [(x/norm, y/norm, z/norm) for x, y, z in verts]


def triad_vertices(radius=0.5):
    return [
        (radius, 0, 0),
        (-radius/2, radius * np.sqrt(3)/2, 0),
        (-radius/2, -radius * np.sqrt(3)/2, 0)
    ]


def scaled_icosahedron(radius):
    return [(x*radius, y*radius, z*radius) for x, y, z in icosahedron_vertices()]


def crystal_geometry_k3():
    """
    K=3 crystal: 1 + 3 + 12 = 16V
    Returns: (vertices, adjacency, layer_masks)
    
    Layer 0: center (index 0)
    Layer 1: triad (indices 1-3)
    Layer 2: icosahedron (indices 4-15)
    """
    layers = [(0, 0, 0)]  # center
    layers.extend(triad_vertices(0.3))  # triad
    layers.extend(scaled_icosahedron(0.6))  # icosahedron
    
    v = len(layers)
    adj = {i: [] for i in range(v)}
    
    def dist(a, b):
        return np.sqrt(sum((layers[a][i] - layers[b][i])**2 for i in range(3)))
    
    # Center connects to all
    for i in range(1, v):
        adj[0].append(i)
        adj[i].append(0)
    
    # Each vertex to nearest 6 neighbors
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
    
    # Layer masks
    layer_masks = {
        'center': [0],
        'triad': [1, 2, 3],
        'icosa': list(range(4, 16))
    }
    
    return v, adj, layer_masks


# =============================================================================
# WAVE DYNAMICS
# =============================================================================

def kuramoto_order_parameter(phases):
    """
    Measure phase coherence.
    r = |1/N * sum(e^(i*phase))|
    r = 1: perfect sync, r = 0: random
    """
    return np.abs(np.mean(np.exp(1j * phases)))


def rhythm_pattern():
    """
    The 3-6-9 pattern with nested loop.
    Returns list of (beat, layer) where layer is 'center', 'triad', or 'icosa'
    
    Pattern: 3-6-9-3-(6-9-3-6)-9-3-6-9
    Mapped: center(3), triad(6), icosa(9)
    """
    return [
        (1, 'center'),
        (2, 'triad'),
        (3, 'icosa'),
        (4, 'center'),
        # nested loop
        (5, 'triad'),
        (6, 'icosa'),
        (7, 'center'),
        (8, 'triad'),
        # resume
        (9, 'icosa'),
        (10, 'center'),
        (11, 'triad'),
        (12, 'icosa'),
    ]


def build_wave_network(frozen_adj, pool_size, contact_ratio=0.1):
    """Build network, return phases and adjacency."""
    n_frozen = len(frozen_adj)
    n_total = n_frozen + pool_size
    
    # Initialize phases: frozen at 0, fluid random
    phases = np.zeros(n_total)
    phases[n_frozen:] = np.random.uniform(0, 2*np.pi, pool_size)
    
    # Build full adjacency
    full_adj = {i: list(frozen_adj[i]) for i in range(n_frozen)}
    for i in range(n_frozen, n_total):
        full_adj[i] = []
    
    # Frozen-fluid contacts
    n_contacts = max(1, int(pool_size * contact_ratio))
    for i in range(n_frozen, n_total):
        contacts = np.random.choice(n_frozen, size=min(n_contacts, n_frozen), replace=False)
        for c in contacts:
            full_adj[i].append(c)
            full_adj[c].append(i)
    
    # Fluid-fluid sparse mesh
    fluid_nodes = list(range(n_frozen, n_total))
    for _ in range(pool_size * 2):
        if len(fluid_nodes) >= 2:
            i, j = np.random.choice(fluid_nodes, size=2, replace=False)
            if j not in full_adj[i]:
                full_adj[i].append(j)
                full_adj[j].append(i)
    
    return phases, full_adj


def wave_step(phases, adj, frozen_mask, layer_masks, beat, amplitude, 
              coupling=0.3, natural_freq=0.1):
    """
    One step of wave dynamics.
    
    - Frozen nodes in the active layer get pulsed (phase kicked by amplitude)
    - All non-frozen nodes couple to neighbors (Kuramoto)
    - Natural frequency advances all phases
    """
    new_phases = phases.copy()
    n_frozen = np.sum(frozen_mask)
    
    # Determine which layer gets pulsed this beat
    rhythm = rhythm_pattern()
    beat_mod = (beat % 12) + 1  # 1-indexed, cycling
    active_layer = None
    for b, layer in rhythm:
        if b == beat_mod:
            active_layer = layer
            break
    
    # Pulse frozen nodes in active layer
    if active_layer and active_layer in layer_masks:
        for i in layer_masks[active_layer]:
            if frozen_mask[i]:
                # Pulse: kick phase by amplitude (alternating direction for interference)
                direction = 1 if (beat // 12) % 2 == 0 else -1
                new_phases[i] = (phases[i] + direction * amplitude) % (2 * np.pi)
    
    # Kuramoto coupling for non-frozen nodes
    for i in range(len(phases)):
        if frozen_mask[i]:
            continue
        
        neighbors = adj[i]
        if neighbors:
            # Phase coupling: pull toward neighbor phases
            phase_diff_sum = sum(np.sin(phases[j] - phases[i]) for j in neighbors)
            new_phases[i] = phases[i] + natural_freq + (coupling / len(neighbors)) * phase_diff_sum
            new_phases[i] = new_phases[i] % (2 * np.pi)
    
    return new_phases


def measure_sync_time(frozen_adj, layer_masks, pool_size, amplitude,
                      target_coherence=0.9, max_steps=200):
    """
    Measure steps to reach target coherence with given amplitude.
    """
    n_frozen = len(frozen_adj)
    phases, full_adj = build_wave_network(frozen_adj, pool_size)
    
    frozen_mask = np.zeros(len(phases), dtype=bool)
    frozen_mask[:n_frozen] = True
    
    for step in range(max_steps):
        coherence = kuramoto_order_parameter(phases)
        if coherence >= target_coherence:
            return coherence, step + 1
        
        phases = wave_step(phases, full_adj, frozen_mask, layer_masks, 
                          step, amplitude)
    
    final_coherence = kuramoto_order_parameter(phases)
    return final_coherence, max_steps


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':
    # Test parameters
    POOL_SIZES = [100, 500, 1000]
    T16 = 136  # Triangular number of 16V crystal
    T16_rad = T16 * (np.pi / 180)  # 136° in radians ≈ 2.374
    T16_norm = T16 / 360 * (2 * np.pi)  # 136/360 of full cycle
    
    AMPLITUDES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 
                  T16_rad,  # 2.374 — T(16) as degrees→radians
                  T16_norm, # 2.374 — T(16)/360 of 2π (same value)
                  np.pi/2, np.pi]
    TRIALS = 5
    
    v, adj, layer_masks = crystal_geometry_k3()
    
    print("="*70)
    print("RHYTHM COHERENCE TEST — K=3 Crystal (16V) with 3-6-9 Pulse Pattern")
    print("="*70)
    print(f"Structure: {v}V (center=1, triad=3, icosa=12)")
    print(f"Rhythm: 12-beat cycle (3-6-9 with nested loop)")
    print()
    
    results = {}
    
    for pool in POOL_SIZES:
        print(f"\nPool {pool}")
        print("-" * 50)
        
        results[pool] = {}
        
        for amp in AMPLITUDES:
            coherences = []
            steps_list = []
            
            for _ in range(TRIALS):
                coh, steps = measure_sync_time(adj, layer_masks, pool, amp)
                coherences.append(coh)
                steps_list.append(steps)
            
            avg_coh = np.mean(coherences)
            avg_steps = np.mean(steps_list)
            
            results[pool][amp] = {
                'coherence': avg_coh,
                'steps': avg_steps
            }
            
            amp_label = f"{amp:.2f}" if amp < 3 else f"π/{np.pi/amp:.1f}"
            status = "✓" if avg_coh >= 0.9 else "○"
            print(f"  Amplitude {amp:>5.2f}: coherence={avg_coh:.3f}  steps={avg_steps:>5.1f}  {status}")
    
    # Find optimal amplitude per pool
    print("\n" + "="*70)
    print("OPTIMAL AMPLITUDE BY POOL SIZE")
    print("="*70)
    
    for pool in POOL_SIZES:
        best_amp = None
        best_steps = float('inf')
        
        for amp, data in results[pool].items():
            if data['coherence'] >= 0.9 and data['steps'] < best_steps:
                best_steps = data['steps']
                best_amp = amp
        
        if best_amp:
            print(f"Pool {pool:>5}: optimal amplitude = {best_amp:.2f}, steps = {best_steps:.1f}")
        else:
            print(f"Pool {pool:>5}: did not reach 90% coherence")
    
    # Save results
    with open('/Users/nick/Projects/holocell/rhythm_coherence_results.json', 'w') as f:
        # Convert numpy types
        clean = {}
        for pool, amps in results.items():
            clean[pool] = {}
            for amp, data in amps.items():
                clean[pool][float(amp)] = {
                    'coherence': float(data['coherence']),
                    'steps': float(data['steps'])
                }
        json.dump(clean, f, indent=2)
    
    print("\nResults saved to rhythm_coherence_results.json")

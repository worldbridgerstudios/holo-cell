#!/usr/bin/env python3
"""
3-Layer Centered Icosahedron Test

Structure:
  Layer 0: 1 center (the One)
  Layer 1: 3 nodes (triangle) — all connect to center
  Layer 2: 12 outer (icosahedron) — connect to center AND nearest middle nodes

Total: 16V
"""

import numpy as np

def centered_icosahedron_3layer():
    phi = (1 + np.sqrt(5)) / 2
    
    # Outer icosahedron vertices (12)
    outer = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    
    # Middle layer: 3 nodes forming triangle at radius ~1
    r = 1.0
    middle = [
        (r, 0, 0),
        (-r/2, r*np.sqrt(3)/2, 0),
        (-r/2, -r*np.sqrt(3)/2, 0)
    ]
    
    # Vertex layout: 0=center, 1-3=middle, 4-15=outer
    v = 16
    adj = {i: [] for i in range(v)}
    
    # Center connects to ALL (middle + outer)
    for i in range(1, 16):
        adj[0].append(i)
        adj[i].append(0)
    
    # Middle triangle (1-2-3)
    adj[1].extend([2, 3])
    adj[2].extend([1, 3])
    adj[3].extend([1, 2])
    
    # Outer icosahedral edges
    def dist(a, b):
        return sum((a[i] - b[i])**2 for i in range(3))**0.5
    
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dist(outer[i], outer[j]) - 2.0) < 0.01:
                adj[i+4].append(j+4)
                adj[j+4].append(i+4)
    
    # Middle-to-outer: each middle node connects to 4 nearest outer
    for mi, mp in enumerate(middle):
        dists = [(dist(mp, outer[oi]), oi) for oi in range(12)]
        dists.sort()
        for _, oi in dists[:4]:
            adj[mi+1].append(oi+4)
            adj[oi+4].append(mi+1)
    
    # Dedupe
    for i in range(v):
        adj[i] = list(set(adj[i]))
    
    return v, adj


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


if __name__ == '__main__':
    POOLS = [100, 300, 1000, 3000, 10000]
    TRIALS = 5
    
    v, adj = centered_icosahedron_3layer()
    
    print(f'3-LAYER CENTERED ICOSAHEDRON — {v}V')
    print(f'Center degree: {len(adj[0])}')
    print(f'Middle degree: {len(adj[1])}')
    print(f'Outer degree: {len(adj[4])}')
    print('-' * 40)
    
    for pool in POOLS:
        rates, steps = zip(*[measure_recovery(adj, pool) for _ in range(TRIALS)])
        print(f'  Pool {pool:>6}: rate={np.mean(rates):.4f}  →90%={np.mean(steps):.1f}')

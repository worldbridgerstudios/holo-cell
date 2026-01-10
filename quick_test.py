#!/usr/bin/env python3
"""Quick HEART vs SHELL comparison for top candidates."""

import random
import sys

# Inline the essential code to avoid import issues
PHI = 1.618033988749895

def truncated_icosahedron():
    """Buckyball: 60 vertices, degree 3."""
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
                    base = [a, b, c]
                    v = (s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]])
                    vertices.append(v)
    
    a, b, c = PHI, 2, 2*PHI+1
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    base = [a, b, c]
                    v = (s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]])
                    vertices.append(v)
    
    # Dedupe
    unique = []
    for v in vertices:
        if not any(sum((a-b)**2 for a,b in zip(v,u)) < 0.01 for u in unique):
            unique.append(v)
    return unique[:60]

def build_adjacency(vertices, tol=0.01):
    """Build adjacency from shortest edges."""
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
        if d <= min_d * (1 + tol):
            adj[i].append(j)
            adj[j].append(i)
    return adj

class Network:
    def __init__(self, adj, frozen):
        self.values = {i: 1.0 for i in adj}
        self.adj = adj
        self.frozen = set(frozen)
    
    def corrupt(self, frac):
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
    
    def coherence(self):
        fluid = [i for i in self.values if i not in self.frozen]
        if not fluid:
            return 1.0, 1.0
        cohs = [max(0, 1 - abs(self.values[i] - 1)) for i in fluid]
        return sum(cohs)/len(cohs), min(cohs)

def build_shell_network(frozen_adj, n_frozen, n_fluid, contact="sparse"):
    """Frozen shell, fluid inside."""
    total = n_frozen + n_fluid
    adj = {i: list(frozen_adj.get(i, [])) for i in range(n_frozen)}
    
    # Fluid nodes in layers
    n_layers = max(1, int(n_fluid ** 0.5))
    per_layer = n_fluid // n_layers
    
    for i in range(n_frozen, total):
        adj[i] = []
        fluid_idx = i - n_frozen
        layer = fluid_idx // per_layer if per_layer > 0 else 0
        
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
            if prev_idx < layer_start:
                adj[i].append(prev_idx)
                adj[prev_idx].append(i)
        
        # Shell contact
        if contact == "full" or (contact == "sparse" and layer == 0):
            c = random.choice(list(range(n_frozen)))
            adj[i].append(c)
            adj[c].append(i)
        elif contact == "minimal" and fluid_idx < 3:
            c = fluid_idx % n_frozen
            adj[i].append(c)
            adj[c].append(i)
    
    return adj, list(range(n_frozen))

def build_heart_network(frozen_adj, n_frozen, n_fluid, contact="sparse"):
    """Frozen core, fluid outside."""
    total = n_frozen + n_fluid
    adj = {i: list(frozen_adj.get(i, [])) for i in range(n_frozen)}
    
    n_layers = max(1, int(n_fluid ** 0.5))
    per_layer = n_fluid // n_layers
    
    for i in range(n_frozen, total):
        adj[i] = []
        fluid_idx = i - n_frozen
        layer = fluid_idx // per_layer if per_layer > 0 else 0
        
        layer_start = n_frozen + layer * per_layer
        layer_end = min(layer_start + per_layer, total)
        
        # Ring
        if i > layer_start:
            adj[i].append(i-1)
            adj[i-1].append(i)
        if i == layer_end - 1 and layer_end - layer_start > 2:
            adj[i].append(layer_start)
            adj[layer_start].append(i)
        
        # Radial to inner layer
        if layer > 0:
            prev_start = n_frozen + (layer-1) * per_layer
            prev_idx = prev_start + (fluid_idx % per_layer)
            if prev_idx < layer_start:
                adj[i].append(prev_idx)
                adj[prev_idx].append(i)
        
        # Core contact (layer 0 = innermost)
        if contact == "full" and layer == 0:
            for c in random.sample(range(n_frozen), min(3, n_frozen)):
                adj[i].append(c)
                adj[c].append(i)
        elif contact == "sparse" and layer == 0:
            c = random.choice(list(range(n_frozen)))
            adj[i].append(c)
            adj[c].append(i)
        elif contact == "minimal" and fluid_idx < 3:
            c = fluid_idx % n_frozen
            adj[i].append(c)
            adj[c].append(i)
    
    return adj, list(range(n_frozen))

def test_config(mode, frozen_adj, n_frozen, n_fluid, contact, trials=5):
    """Test one configuration."""
    successes = 0
    total_coh = 0
    
    for _ in range(trials):
        if mode == "shell":
            adj, frozen = build_shell_network(frozen_adj, n_frozen, n_fluid, contact)
        else:
            adj, frozen = build_heart_network(frozen_adj, n_frozen, n_fluid, contact)
        
        net = Network(adj, frozen)
        net.corrupt(0.5)
        
        for _ in range(100):
            net.step()
        
        avg, _ = net.coherence()
        total_coh += avg
        if avg >= 0.8:
            successes += 1
    
    return successes / trials, total_coh / trials

def main():
    print("=" * 80)
    print("HEART vs SHELL — Buckyball (60V) Characterization")
    print("=" * 80)
    
    # Build buckyball
    verts = truncated_icosahedron()
    adj = build_adjacency(verts)
    n_frozen = len(verts)
    
    print(f"\nFrozen geometry: {n_frozen} vertices")
    print("Testing ratios: 10:1, 50:1, 100:1, 200:1, 500:1")
    print("Contact modes: full, sparse, minimal")
    print()
    
    ratios = [10, 50, 100, 200, 500]
    
    for contact in ["full", "sparse", "minimal"]:
        print(f"\n--- Contact: {contact.upper()} ---")
        print(f"{'Ratio':<10} {'SHELL':<20} {'HEART':<20}")
        print("-" * 50)
        
        for ratio in ratios:
            n_fluid = n_frozen * ratio
            
            shell_rate, shell_coh = test_config("shell", adj, n_frozen, n_fluid, contact)
            heart_rate, heart_coh = test_config("heart", adj, n_frozen, n_fluid, contact)
            
            shell_str = f"{shell_rate*100:.0f}% ({shell_coh:.1%})"
            heart_str = f"{heart_rate*100:.0f}% ({heart_coh:.1%})"
            
            print(f"{ratio}:1{'':<6} {shell_str:<20} {heart_str:<20}")
            sys.stdout.flush()
    
    print("\n" + "=" * 80)
    print("FINDING: Maximum stable ratio per configuration")
    print("=" * 80)

if __name__ == "__main__":
    main()

"""
HoloCell — Extreme Sparsity Characterization

Find the actual breaking point: how many fluid nodes can be stabilized
by a given frozen geometry?

EXPERIMENT:
- For each polyhedron, scale fluid nodes until recovery fails
- Test both HEART (frozen core) and SHELL (frozen boundary) modes
- Measure: max_ratio, resilience_curve, throughput

OUTPUT: Characterization table ranking all 36 configurations
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from holocell.modes.polyhedra import (
    POLYHEDRA, POLYHEDRA_INFO, Mode, Node, CoherenceNetwork,
    _distance, _normalize
)


# =============================================================================
# IMPROVED NETWORK BUILDERS
# =============================================================================

def build_shell_network_scaled(
    name: str, 
    fluid_count: int,
    contact_mode: str = "full",  # "full", "sparse", "minimal"
) -> CoherenceNetwork:
    """
    Build SHELL mode: frozen polyhedron as boundary, fluid nodes inside.
    
    Fluid nodes form concentric layers, each connected to outer layer.
    Shell nodes connect to outermost fluid layer.
    
    contact_mode:
    - "full": Every fluid node touches shell
    - "sparse": Only outermost fluid layer touches shell  
    - "minimal": Single contact point to shell
    """
    vertices, adj = POLYHEDRA[name]()
    n_frozen = len(vertices)
    frozen_indices = list(range(n_frozen))
    total = n_frozen + fluid_count
    
    # Build full adjacency
    full_adj = {i: list(adj.get(i, [])) for i in range(n_frozen)}
    
    # Organize fluid nodes in layers (roughly sqrt layers)
    n_layers = max(1, int(fluid_count ** 0.5))
    nodes_per_layer = fluid_count // n_layers
    
    for i in range(n_frozen, total):
        full_adj[i] = []
        fluid_idx = i - n_frozen
        layer = fluid_idx // nodes_per_layer if nodes_per_layer > 0 else 0
        pos_in_layer = fluid_idx % nodes_per_layer if nodes_per_layer > 0 else fluid_idx
        
        # Connect within layer (ring)
        same_layer_start = n_frozen + layer * nodes_per_layer
        same_layer_end = min(same_layer_start + nodes_per_layer, total)
        
        if i > same_layer_start:
            full_adj[i].append(i - 1)
            full_adj[i - 1].append(i)
        
        # Close the ring
        if i == same_layer_end - 1 and same_layer_end - same_layer_start > 2:
            full_adj[i].append(same_layer_start)
            full_adj[same_layer_start].append(i)
        
        # Connect to previous layer
        if layer > 0:
            prev_layer_start = n_frozen + (layer - 1) * nodes_per_layer
            # Connect to corresponding position in previous layer
            prev_idx = prev_layer_start + (pos_in_layer % nodes_per_layer)
            if prev_idx < same_layer_start:
                full_adj[i].append(prev_idx)
                full_adj[prev_idx].append(i)
        
        # Connect to shell (frozen boundary)
        if contact_mode == "full":
            # Every fluid node connects to shell
            shell_contacts = random.sample(frozen_indices, min(2, n_frozen))
            for c in shell_contacts:
                full_adj[i].append(c)
                full_adj[c].append(i)
        elif contact_mode == "sparse":
            # Only outermost layer (layer 0) connects to shell
            if layer == 0:
                shell_contacts = random.sample(frozen_indices, min(2, n_frozen))
                for c in shell_contacts:
                    full_adj[i].append(c)
                    full_adj[c].append(i)
        elif contact_mode == "minimal":
            # Only first few fluid nodes connect to shell
            if fluid_idx < 3:
                c = frozen_indices[fluid_idx % n_frozen]
                full_adj[i].append(c)
                full_adj[c].append(i)
    
    return CoherenceNetwork(full_adj, frozen_indices)


def build_heart_network_scaled(
    name: str,
    fluid_count: int,
    contact_mode: str = "full",
) -> CoherenceNetwork:
    """
    Build HEART mode: frozen polyhedron at center, fluid nodes outside.
    
    Fluid nodes form concentric shells, each connected to inner shell.
    Innermost fluid shell connects to frozen core.
    """
    vertices, adj = POLYHEDRA[name]()
    n_frozen = len(vertices)
    frozen_indices = list(range(n_frozen))
    total = n_frozen + fluid_count
    
    full_adj = {i: list(adj.get(i, [])) for i in range(n_frozen)}
    
    n_layers = max(1, int(fluid_count ** 0.5))
    nodes_per_layer = fluid_count // n_layers
    
    for i in range(n_frozen, total):
        full_adj[i] = []
        fluid_idx = i - n_frozen
        layer = fluid_idx // nodes_per_layer if nodes_per_layer > 0 else 0
        pos_in_layer = fluid_idx % nodes_per_layer if nodes_per_layer > 0 else fluid_idx
        
        same_layer_start = n_frozen + layer * nodes_per_layer
        same_layer_end = min(same_layer_start + nodes_per_layer, total)
        
        # Ring within layer
        if i > same_layer_start:
            full_adj[i].append(i - 1)
            full_adj[i - 1].append(i)
        if i == same_layer_end - 1 and same_layer_end - same_layer_start > 2:
            full_adj[i].append(same_layer_start)
            full_adj[same_layer_start].append(i)
        
        # Connect to previous layer (closer to core)
        if layer > 0:
            prev_layer_start = n_frozen + (layer - 1) * nodes_per_layer
            prev_idx = prev_layer_start + (pos_in_layer % nodes_per_layer)
            if prev_idx < same_layer_start:
                full_adj[i].append(prev_idx)
                full_adj[prev_idx].append(i)
        
        # Connect to frozen core
        if contact_mode == "full":
            if layer == 0:  # Innermost layer connects to core
                core_contacts = random.sample(frozen_indices, min(3, n_frozen))
                for c in core_contacts:
                    full_adj[i].append(c)
                    full_adj[c].append(i)
        elif contact_mode == "sparse":
            if layer == 0 and pos_in_layer % 3 == 0:
                c = frozen_indices[pos_in_layer % n_frozen]
                full_adj[i].append(c)
                full_adj[c].append(i)
        elif contact_mode == "minimal":
            if fluid_idx < 3:
                c = frozen_indices[fluid_idx % n_frozen]
                full_adj[i].append(c)
                full_adj[c].append(i)
    
    return CoherenceNetwork(full_adj, frozen_indices)


# =============================================================================
# SPARSITY TEST
# =============================================================================

@dataclass
class SparsityResult:
    name: str
    mode: str
    frozen: int
    fluid: int
    ratio: float  # fluid:frozen
    corruption: float
    recovered: bool
    avg_coherence: float
    min_coherence: float
    recovery_steps: int


def test_sparsity_limit(
    name: str,
    mode: Mode,
    contact_mode: str = "full",
    corruption: float = 0.5,
    trials: int = 5,
    max_steps: int = 100,
    coherence_threshold: float = 0.8,
) -> Tuple[int, List[SparsityResult]]:
    """
    Find maximum fluid nodes that can be stabilized.
    
    Binary search for the breaking point.
    """
    info = POLYHEDRA_INFO[name]
    n_frozen = info[0]
    
    results = []
    
    # Extended ratios to find actual breaking point
    test_ratios = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    max_working_ratio = 0
    
    for ratio in test_ratios:
        fluid_count = n_frozen * ratio
        
        trial_results = []
        for _ in range(trials):
            if mode == Mode.SHELL:
                net = build_shell_network_scaled(name, fluid_count, contact_mode)
            else:
                net = build_heart_network_scaled(name, fluid_count, contact_mode)
            
            net.corrupt(corruption)
            
            for step in range(max_steps):
                net.step()
                avg_coh, min_coh = net.coherence()
                if min_coh > 0.95:
                    break
            
            avg_coh, min_coh = net.coherence()
            trial_results.append((avg_coh, min_coh, step + 1, avg_coh >= coherence_threshold))
        
        avg_coherence = sum(r[0] for r in trial_results) / len(trial_results)
        min_coherence = sum(r[1] for r in trial_results) / len(trial_results)
        avg_steps = sum(r[2] for r in trial_results) / len(trial_results)
        recovery_rate = sum(1 for r in trial_results if r[3]) / len(trial_results)
        
        recovered = recovery_rate >= 0.8  # 80% of trials must recover
        
        result = SparsityResult(
            name=name,
            mode=mode.value,
            frozen=n_frozen,
            fluid=fluid_count,
            ratio=ratio,
            corruption=corruption,
            recovered=recovered,
            avg_coherence=avg_coherence,
            min_coherence=min_coherence,
            recovery_steps=int(avg_steps),
        )
        results.append(result)
        
        if recovered:
            max_working_ratio = ratio
        else:
            break  # Stop at first failure
    
    return max_working_ratio, results


# =============================================================================
# FULL CHARACTERIZATION
# =============================================================================

@dataclass
class PolyhedronCharacterization:
    name: str
    vertices: int
    degree: int
    family: str
    
    # SHELL mode results
    shell_max_ratio: int
    shell_max_fluid: int
    shell_50pct_coherence: float
    
    # HEART mode results
    heart_max_ratio: int
    heart_max_fluid: int
    heart_50pct_coherence: float
    
    # Best configuration
    best_mode: str
    best_ratio: int
    efficiency_score: float  # ratio * coherence


def characterize_all_polyhedra(
    contact_mode: str = "full",
    verbose: bool = True,
) -> List[PolyhedronCharacterization]:
    """
    Full characterization of all 18 polyhedra in both modes.
    """
    results = []
    
    if verbose:
        print("\n" + "=" * 90)
        print("OPTIMAL COHERENCE NETWORK — EXTREME SPARSITY CHARACTERIZATION")
        print("=" * 90)
        print(f"Testing 18 polyhedra × 2 modes, contact_mode={contact_mode}")
        print("=" * 90)
    
    for name in POLYHEDRA:
        info = POLYHEDRA_INFO[name]
        n_verts, degree, family = info
        
        if verbose:
            print(f"\n{name} ({n_verts}V, deg-{degree}, {family})")
        
        # Test SHELL mode
        shell_ratio, shell_results = test_sparsity_limit(name, Mode.SHELL, contact_mode)
        shell_50_coh = next((r.avg_coherence for r in shell_results if r.ratio == 50), 0.0)
        
        if verbose:
            print(f"  SHELL: max_ratio={shell_ratio}:1 ({n_verts * shell_ratio} fluid)")
        
        # Test HEART mode
        heart_ratio, heart_results = test_sparsity_limit(name, Mode.HEART, contact_mode)
        heart_50_coh = next((r.avg_coherence for r in heart_results if r.ratio == 50), 0.0)
        
        if verbose:
            print(f"  HEART: max_ratio={heart_ratio}:1 ({n_verts * heart_ratio} fluid)")
        
        # Determine best mode
        if shell_ratio > heart_ratio:
            best_mode = "SHELL"
            best_ratio = shell_ratio
        elif heart_ratio > shell_ratio:
            best_mode = "HEART"
            best_ratio = heart_ratio
        else:
            # Tie-breaker: use coherence
            if shell_50_coh >= heart_50_coh:
                best_mode = "SHELL"
                best_ratio = shell_ratio
            else:
                best_mode = "HEART"
                best_ratio = heart_ratio
        
        char = PolyhedronCharacterization(
            name=name,
            vertices=n_verts,
            degree=degree,
            family=family,
            shell_max_ratio=shell_ratio,
            shell_max_fluid=n_verts * shell_ratio,
            shell_50pct_coherence=shell_50_coh,
            heart_max_ratio=heart_ratio,
            heart_max_fluid=n_verts * heart_ratio,
            heart_50pct_coherence=heart_50_coh,
            best_mode=best_mode,
            best_ratio=best_ratio,
            efficiency_score=best_ratio * max(shell_50_coh, heart_50_coh),
        )
        results.append(char)
    
    return results


def print_characterization_table(results: List[PolyhedronCharacterization]):
    """Print the final characterization table."""
    print("\n" + "=" * 120)
    print("FINAL CHARACTERIZATION TABLE — OPTIMAL COHERENCE NETWORKS")
    print("=" * 120)
    print(f"{'Polyhedron':<28} {'V':>4} {'D':>2} {'SHELL':>8} {'HEART':>8} {'Best':>6} {'Ratio':>6} {'Max Fluid':>10} {'Score':>8}")
    print("-" * 120)
    
    # Sort by efficiency score (descending)
    sorted_results = sorted(results, key=lambda r: r.efficiency_score, reverse=True)
    
    for r in sorted_results:
        print(f"{r.name:<28} {r.vertices:>4} {r.degree:>2} "
              f"{r.shell_max_ratio:>6}:1 {r.heart_max_ratio:>6}:1 "
              f"{r.best_mode:>6} {r.best_ratio:>5}:1 "
              f"{r.vertices * r.best_ratio:>10} {r.efficiency_score:>8.1f}")
    
    print("=" * 120)
    print("\nTOP 5 CONFIGURATIONS:")
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r.name} ({r.best_mode}): {r.vertices} frozen → {r.vertices * r.best_ratio} fluid")


def run_extreme_sparsity_test():
    """Main entry point for extreme sparsity testing."""
    print("\n" + "═" * 90)
    print(" HOLOCELL — EXTREME SPARSITY EXPERIMENT")
    print("═" * 90)
    print("\nObjective: Find maximum fluid:frozen ratio for each polyhedron")
    print("Corruption: 50%, Recovery threshold: 80% coherence")
    print("═" * 90)
    
    start_time = time.time()
    results = characterize_all_polyhedra(contact_mode="full", verbose=True)
    elapsed = time.time() - start_time
    
    print_characterization_table(results)
    
    print(f"\nCompleted in {elapsed:.1f} seconds")
    
    return results


if __name__ == "__main__":
    run_extreme_sparsity_test()

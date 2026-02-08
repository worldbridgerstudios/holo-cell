"""
HoloCell — Contact Mode Comparison

Compare topology effects under different contact modes:
- full: Every fluid node touches shell
- sparse: Only outermost layer touches shell  
- minimal: Single contact points to shell

This reveals whether the polyhedron topology matters or just having ANY frozen nodes.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from holocell.modes.extreme_sparsity import (
    build_shell_network_scaled,
    build_heart_network_scaled,
    Mode, SparsityResult, POLYHEDRA, POLYHEDRA_INFO
)


def test_contact_modes(
    name: str,
    fluid_count: int,
    corruption: float = 0.5,
    trials: int = 5,
    max_steps: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Test one polyhedron under all contact modes."""
    results = {}
    
    for mode in [Mode.SHELL, Mode.HEART]:
        results[mode.value] = {}
        
        for contact in ["full", "sparse", "minimal"]:
            trial_results = []
            
            for _ in range(trials):
                if mode == Mode.SHELL:
                    net = build_shell_network_scaled(name, fluid_count, contact)
                else:
                    net = build_heart_network_scaled(name, fluid_count, contact)
                
                net.corrupt(corruption)
                
                for step in range(max_steps):
                    net.step()
                    avg_coh, min_coh = net.coherence()
                    if min_coh > 0.95:
                        break
                
                avg_coh, min_coh = net.coherence()
                trial_results.append(avg_coh)
            
            results[mode.value][contact] = sum(trial_results) / len(trial_results)
    
    return results


def run_contact_comparison():
    """Compare contact modes across top polyhedra."""
    
    # Top candidates from initial test
    candidates = [
        "truncated_icosidodecahedron",  # 120V
        "truncated_icosahedron",          # 60V (buckyball)
        "rhombicosidodecahedron",         # 60V
        "icosahedron",                    # 12V (baseline)
        "tetrahedron",                    # 4V (minimal)
    ]
    
    print("\n" + "=" * 100)
    print("CONTACT MODE COMPARISON — Topology vs Contact Effects")
    print("=" * 100)
    print("Testing whether polyhedron topology matters beyond simple frozen contact")
    print("Ratio: 100:1 fluid:frozen, Corruption: 50%")
    print("=" * 100)
    
    for name in candidates:
        info = POLYHEDRA_INFO[name]
        n_verts = info[0]
        fluid_count = n_verts * 100  # 100:1 ratio
        
        print(f"\n{name} ({n_verts}V → {fluid_count} fluid)")
        print("-" * 80)
        
        results = test_contact_modes(name, fluid_count)
        
        print(f"{'Mode':<10} {'Full':<12} {'Sparse':<12} {'Minimal':<12}")
        for mode in ["shell", "heart"]:
            full = results[mode]["full"]
            sparse = results[mode]["sparse"]
            minimal = results[mode]["minimal"]
            print(f"{mode.upper():<10} {full:>10.1%} {sparse:>12.1%} {minimal:>12.1%}")


def find_breaking_point_comparison():
    """Find actual breaking point for each contact mode."""
    
    print("\n" + "=" * 100)
    print("BREAKING POINT FINDER — Maximum stable ratio per contact mode")
    print("=" * 100)
    
    candidates = [
        ("truncated_icosahedron", 60),    # buckyball
        ("icosahedron", 12),               # baseline
    ]
    
    test_ratios = [100, 250, 500, 1000, 2000, 5000, 10000]
    
    for name, n_verts in candidates:
        print(f"\n{name} ({n_verts}V)")
        print("-" * 80)
        print(f"{'Contact':<10} ", end="")
        for r in test_ratios:
            print(f"{r}:1".center(10), end="")
        print()
        
        for contact in ["full", "sparse", "minimal"]:
            print(f"{contact:<10} ", end="")
            
            for ratio in test_ratios:
                fluid_count = n_verts * ratio
                
                recoveries = 0
                trials = 3
                
                for _ in range(trials):
                    net = build_shell_network_scaled(name, fluid_count, contact)
                    net.corrupt(0.5)
                    
                    for _ in range(100):
                        net.step()
                    
                    avg_coh, _ = net.coherence()
                    if avg_coh >= 0.8:
                        recoveries += 1
                
                if recoveries >= 2:  # 2/3 must recover
                    print("   ✓     ", end="")
                else:
                    print("   ✗     ", end="")
            print()


if __name__ == "__main__":
    print("\n" + "═" * 100)
    print(" HOLOCELL — CONTACT MODE ANALYSIS")
    print("═" * 100)
    
    start = time.time()
    
    run_contact_comparison()
    find_breaking_point_comparison()
    
    print(f"\nCompleted in {time.time() - start:.1f}s")

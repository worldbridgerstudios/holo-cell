"""
HoloCell — Homogeneous Network Stability Test

Test fault tolerance of N-sized networks where all nodes have the same value.
Default: all nodes = T(16) = 136

Uses the existing evolve_coherent machinery.
"""

import random
import math
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

from .coherent_zoom import evolve_coherent
from .targets import ARCHITECTURE


@dataclass
class HomogeneousResult:
    """Result from homogeneous network test."""
    n: int
    base_value: float
    corrupted_count: int
    corruption_fraction: float
    discovered_integers: List[int]
    architectural_overlap: int
    total_error: float
    converged: bool
    
    def __repr__(self):
        status = "✓" if self.converged else "✗"
        return (
            f"{status} N={self.n}, corrupted={self.corrupted_count}/{self.n} "
            f"({self.corruption_fraction:.0%}), overlap={self.architectural_overlap}/11"
        )


def _integer_overlap(discovered: List[int]) -> int:
    """Count how many discovered integers match architectural set."""
    count = 0
    for d in discovered:
        if any(abs(d - a) <= 1 for a in ARCHITECTURE):
            count += 1
    return count


def test_homogeneous_network(
    n: int = 7,
    value: float = 136.0,
    corruption_fraction: float = 0.5,
    generations: int = 500,
    verbose: bool = True,
    random_seed: Optional[int] = None,
) -> HomogeneousResult:
    """
    Test fault tolerance of homogeneous N-node network.
    
    Creates N "constants" all with the same value, corrupts some,
    runs coherent evolution, checks if architectural integers emerge.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create targets: N nodes all with same value
    targets = {}
    corrupted_indices = set(random.sample(range(n), int(n * corruption_fraction)))
    corrupted_count = len(corrupted_indices)
    
    for i in range(n):
        if i in corrupted_indices:
            # Random corruption
            magnitude = abs(value) + 0.001
            log_min = math.log(magnitude / 100)
            log_max = math.log(magnitude * 100)
            targets[f"node_{i}"] = math.exp(log_min + random.random() * (log_max - log_min))
        else:
            targets[f"node_{i}"] = value
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HOMOGENEOUS NETWORK TEST")
        print(f"{'='*60}")
        print(f"N = {n}, base value = {value}")
        print(f"Corrupted: {corrupted_count}/{n} ({corruption_fraction:.0%})")
        print(f"\nNode values:")
        for i in range(n):
            marker = " (corrupted)" if i in corrupted_indices else ""
            print(f"  [{i}] {targets[f'node_{i}']:.4f}{marker}")
    
    # Run coherent evolution
    if verbose:
        print(f"\nRunning coherent evolution ({generations} generations)...")
    
    result = evolve_coherent(
        integer_set_size=9,
        head_length=8,
        pop_size=200,
        generations=generations,
        targets=targets,
        verbose=False,
    )
    
    overlap = _integer_overlap(result.discovered_integers)
    
    if verbose:
        print(f"\n{'─'*60}")
        print(f"RESULT")
        print(f"{'─'*60}")
        print(f"Discovered integers: {sorted(result.discovered_integers)}")
        print(f"Architectural overlap: {overlap}/11")
        print(f"Total error: {result.total_error:.6f}")
        print(f"Converged: {'✓' if result.converged else '✗'}")
        print(f"{'='*60}")
    
    return HomogeneousResult(
        n=n,
        base_value=value,
        corrupted_count=corrupted_count,
        corruption_fraction=corruption_fraction,
        discovered_integers=result.discovered_integers,
        architectural_overlap=overlap,
        total_error=result.total_error,
        converged=result.converged,
    )


def sweep_homogeneous(
    n: int = 7,
    value: float = 136.0,
    trials_per_level: int = 3,
    generations: int = 500,
    verbose: bool = True,
) -> Dict[int, List[HomogeneousResult]]:
    """
    Sweep corruption levels for fixed N.
    """
    results = {}
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"# HOMOGENEOUS NETWORK SWEEP: N={n}, value={value}")
        print(f"{'#'*60}")
    
    for corrupted in range(n):
        results[corrupted] = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CORRUPTION LEVEL: {corrupted}/{n}")
            print(f"{'='*60}")
        
        for trial in range(trials_per_level):
            if verbose:
                print(f"\n--- Trial {trial+1}/{trials_per_level} ---")
            
            result = test_homogeneous_network(
                n=n,
                value=value,
                corruption_fraction=corrupted/n if n > 0 else 0,
                generations=generations,
                verbose=verbose,
            )
            results[corrupted].append(result)
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"{'Corrupted':<12} {'Avg Overlap':<15} {'Converged'}")
        print("─" * 40)
        
        for level in sorted(results.keys()):
            trials = results[level]
            avg_overlap = sum(t.architectural_overlap for t in trials) / len(trials)
            converge_rate = sum(1 for t in trials if t.converged) / len(trials)
            bar = "█" * int(avg_overlap) + "░" * (11 - int(avg_overlap))
            print(f"{level}/{n:<10} [{bar}] {avg_overlap:.1f}    {converge_rate:.0%}")
        
        print(f"{'='*60}")
    
    return results

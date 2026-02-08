"""
HoloCell — Mode 6: Weave (Incremental Corruption & Restoration)

Instead of batch corruption, weave between coherent and corrupted states:

    BATCH (Mode 5):              WEAVE (Mode 6):
    
    corrupt 7 → evolve 1000      corrupt 1 → evolve 50 → measure
               → measure         corrupt 1 → evolve 50 → measure
                                 ...
    corrupt 7 → evolve 1000      restore 1 → evolve 50 → measure
               → measure         restore 1 → evolve 50 → measure
                                 ...

This reveals:
- Healing dynamics (per-step trajectory)
- Hysteresis (does path matter?)
- Phase transitions (sharp or gradual?)
- Selection effects (which nodes matter most?)

The weave tests three selection strategies:
- RANDOM: arbitrary order
- WORST_FIRST: corrupt highest-error nodes first, restore lowest-error first
- BEST_FIRST: corrupt lowest-error nodes first, restore highest-error first

If the manifold is a true basin of attraction, restore trajectory mirrors
degrade trajectory. If there's hysteresis, the system "remembers" damage.
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .targets import EXTENDED_TARGETS, ARCHITECTURE
from .coherent_zoom import evolve_coherent


class SelectionStrategy(Enum):
    """Strategy for selecting which node to corrupt/restore next."""
    RANDOM = "random"
    WORST_FIRST = "worst_first"   # Corrupt worst-fit first, restore best-fit first
    BEST_FIRST = "best_first"     # Corrupt best-fit first, restore worst-fit first


@dataclass
class WeaveStep:
    """A single step in the weave trajectory."""
    step_number: int
    action: str                      # "corrupt" or "restore"
    node_name: str                   # Which constant was affected
    corrupted_nodes: List[str]       # All currently corrupted nodes
    discovered_integers: List[int]
    overlap: int                     # With architectural set
    total_error: float
    node_errors: Dict[str, float]    # Per-node errors at this step


@dataclass
class WeaveResult:
    """Complete result from a weave experiment."""
    strategy: SelectionStrategy
    max_corruption: int
    trajectory: List[WeaveStep]
    degrade_steps: List[WeaveStep]   # Just the corruption phase
    restore_steps: List[WeaveStep]   # Just the restoration phase
    
    # Key metrics
    threshold_step: int              # Step where overlap first drops below 60%
    hysteresis_score: float          # 0 = perfect symmetry, 1 = total asymmetry
    min_overlap: int                 # Lowest overlap during experiment
    recovery_complete: bool          # Did we return to baseline?
    
    elapsed_seconds: float

    def __repr__(self):
        return (
            f"WeaveResult({self.strategy.value})\n"
            f"  Threshold at step: {self.threshold_step}\n"
            f"  Min overlap: {self.min_overlap}/9\n"
            f"  Hysteresis: {self.hysteresis_score:.2%}\n"
            f"  Full recovery: {self.recovery_complete}"
        )


def _exponential_sample(min_val: float, max_val: float) -> float:
    """Sample from log-uniform distribution."""
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    return math.exp(log_min + random.random() * (log_max - log_min))


def _generate_corrupted_value(original: float) -> float:
    """Generate a corrupted value far from original."""
    magnitude = abs(original) + 0.001
    return _exponential_sample(magnitude / 100, magnitude * 100)


def _integer_overlap(discovered: List[int]) -> int:
    """Count how many discovered integers match architectural set."""
    count = 0
    for d in discovered:
        if any(abs(d - a) <= 1 for a in ARCHITECTURE):
            count += 1
    return count


def _select_node_to_corrupt(
    available: List[str],
    node_errors: Dict[str, float],
    strategy: SelectionStrategy,
) -> str:
    """Select which node to corrupt next based on strategy."""
    if strategy == SelectionStrategy.RANDOM:
        return random.choice(available)
    elif strategy == SelectionStrategy.WORST_FIRST:
        # Corrupt the node with highest error first
        return max(available, key=lambda n: node_errors.get(n, 0))
    elif strategy == SelectionStrategy.BEST_FIRST:
        # Corrupt the node with lowest error first
        return min(available, key=lambda n: node_errors.get(n, float('inf')))
    else:
        return random.choice(available)


def _select_node_to_restore(
    corrupted: List[str],
    node_errors: Dict[str, float],
    strategy: SelectionStrategy,
) -> str:
    """Select which node to restore next based on strategy."""
    if strategy == SelectionStrategy.RANDOM:
        return random.choice(corrupted)
    elif strategy == SelectionStrategy.WORST_FIRST:
        # Restore the node with lowest error first (easiest to fix)
        return min(corrupted, key=lambda n: node_errors.get(n, float('inf')))
    elif strategy == SelectionStrategy.BEST_FIRST:
        # Restore the node with highest error first (hardest to fix)
        return max(corrupted, key=lambda n: node_errors.get(n, 0))
    else:
        return random.choice(corrupted)


def _compute_hysteresis(degrade: List[WeaveStep], restore: List[WeaveStep]) -> float:
    """
    Compute hysteresis score between degrade and restore trajectories.
    
    0.0 = perfect mirror symmetry (no hysteresis)
    1.0 = complete asymmetry (maximum hysteresis)
    
    Compares overlap values at matching corruption levels.
    """
    if not degrade or not restore:
        return 0.0
    
    # Get overlap at each corruption level during degrade
    degrade_overlaps = [step.overlap for step in degrade]
    
    # Get overlap at each corruption level during restore (reversed)
    restore_overlaps = [step.overlap for step in reversed(restore)]
    
    # Ensure same length
    min_len = min(len(degrade_overlaps), len(restore_overlaps))
    if min_len == 0:
        return 0.0
    
    # Compute mean absolute difference
    total_diff = sum(
        abs(d - r) 
        for d, r in zip(degrade_overlaps[:min_len], restore_overlaps[:min_len])
    )
    max_possible_diff = 9 * min_len  # Maximum if all were 0 vs 9
    
    return total_diff / max_possible_diff if max_possible_diff > 0 else 0.0


def weave(
    max_corruption: int = 8,
    strategy: SelectionStrategy = SelectionStrategy.RANDOM,
    generations_per_step: int = 100,
    integer_set_size: int = 9,
    verbose: bool = True,
    random_seed: Optional[int] = None,
) -> WeaveResult:
    """
    Mode 6: Weave — Incremental corruption and restoration.

    Incrementally corrupt nodes one at a time, then restore them,
    measuring coherence at each step. This reveals the dynamics
    of how the manifold degrades and heals.

    The experiment has two phases:
    1. DEGRADE: Corrupt nodes one at a time until max_corruption
    2. RESTORE: Restore nodes one at a time until fully coherent

    Args:
        max_corruption: Maximum nodes to corrupt before reversing
        strategy: Node selection strategy (RANDOM, WORST_FIRST, BEST_FIRST)
        generations_per_step: Evolution generations per step (fewer = faster)
        integer_set_size: Size of co-evolved integer set
        verbose: Print progress
        random_seed: For reproducibility

    Returns:
        WeaveResult with full trajectory and analysis
    """
    if random_seed is not None:
        random.seed(random_seed)

    constant_names = list(EXTENDED_TARGETS.keys())
    true_values = {name: t.value for name, t in EXTENDED_TARGETS.items()}

    if verbose:
        print("\n" + "=" * 70)
        print(f"MODE 6: WEAVE ({strategy.value})")
        print("=" * 70)
        print(f"\nMax corruption: {max_corruption} nodes")
        print(f"Generations per step: {generations_per_step}")
        print(f"Strategy: {strategy.value}\n")

    start = time.time()
    
    # State tracking
    corrupted_nodes: List[str] = []
    corrupted_values: Dict[str, float] = {}  # Store corrupted values for restoration
    current_targets = true_values.copy()
    
    trajectory: List[WeaveStep] = []
    degrade_steps: List[WeaveStep] = []
    restore_steps: List[WeaveStep] = []
    
    step_number = 0
    node_errors: Dict[str, float] = {name: 0.0 for name in constant_names}

    # === BASELINE ===
    if verbose:
        print("─" * 50)
        print("BASELINE (0 corruptions)")
        print("─" * 50)
    
    result = evolve_coherent(
        integer_set_size=integer_set_size,
        head_length=8,
        pop_size=200,
        generations=generations_per_step * 2,  # More for baseline
        targets=current_targets,
        verbose=False,
    )
    
    baseline_overlap = _integer_overlap(result.discovered_integers)
    baseline_integers = result.discovered_integers.copy()
    
    if verbose:
        ints_str = ",".join(str(i) for i in result.discovered_integers)
        print(f"  Overlap: {baseline_overlap}/9, Integers: [{ints_str}]")

    # === DEGRADE PHASE ===
    if verbose:
        print("\n" + "─" * 50)
        print("DEGRADE PHASE")
        print("─" * 50)

    available = constant_names.copy()
    
    for i in range(max_corruption):
        step_number += 1
        
        # Select node to corrupt
        node = _select_node_to_corrupt(available, node_errors, strategy)
        available.remove(node)
        
        # Corrupt it
        corrupted_value = _generate_corrupted_value(true_values[node])
        corrupted_values[node] = corrupted_value
        corrupted_nodes.append(node)
        current_targets[node] = corrupted_value
        
        # Evolve (shorter, incremental)
        result = evolve_coherent(
            integer_set_size=integer_set_size,
            head_length=8,
            pop_size=200,
            generations=generations_per_step,
            targets=current_targets,
            verbose=False,
        )
        
        overlap = _integer_overlap(result.discovered_integers)
        
        # Update node errors (simplified: use total error distributed)
        for name in constant_names:
            node_errors[name] = result.total_error / len(constant_names)
        
        step = WeaveStep(
            step_number=step_number,
            action="corrupt",
            node_name=node,
            corrupted_nodes=corrupted_nodes.copy(),
            discovered_integers=result.discovered_integers,
            overlap=overlap,
            total_error=result.total_error,
            node_errors=node_errors.copy(),
        )
        trajectory.append(step)
        degrade_steps.append(step)
        
        if verbose:
            ints_str = ",".join(str(i) for i in result.discovered_integers)
            bar = "█" * overlap + "░" * (9 - overlap)
            print(f"  Step {step_number}: corrupt '{node}' → [{bar}] {overlap}/9")

    # === RESTORE PHASE ===
    if verbose:
        print("\n" + "─" * 50)
        print("RESTORE PHASE")
        print("─" * 50)

    for i in range(max_corruption):
        step_number += 1
        
        # Select node to restore
        node = _select_node_to_restore(corrupted_nodes, node_errors, strategy)
        corrupted_nodes.remove(node)
        
        # Restore it
        current_targets[node] = true_values[node]
        
        # Evolve (shorter, incremental)
        result = evolve_coherent(
            integer_set_size=integer_set_size,
            head_length=8,
            pop_size=200,
            generations=generations_per_step,
            targets=current_targets,
            verbose=False,
        )
        
        overlap = _integer_overlap(result.discovered_integers)
        
        # Update node errors
        for name in constant_names:
            node_errors[name] = result.total_error / len(constant_names)
        
        step = WeaveStep(
            step_number=step_number,
            action="restore",
            node_name=node,
            corrupted_nodes=corrupted_nodes.copy(),
            discovered_integers=result.discovered_integers,
            overlap=overlap,
            total_error=result.total_error,
            node_errors=node_errors.copy(),
        )
        trajectory.append(step)
        restore_steps.append(step)
        
        if verbose:
            ints_str = ",".join(str(i) for i in result.discovered_integers)
            bar = "█" * overlap + "░" * (9 - overlap)
            print(f"  Step {step_number}: restore '{node}' → [{bar}] {overlap}/9")

    elapsed = time.time() - start

    # === ANALYSIS ===
    
    # Find threshold (first step where overlap < 60% of baseline)
    threshold_step = len(trajectory)
    threshold_overlap = baseline_overlap * 0.6
    for step in trajectory:
        if step.overlap < threshold_overlap:
            threshold_step = step.step_number
            break
    
    # Compute hysteresis
    hysteresis = _compute_hysteresis(degrade_steps, restore_steps)
    
    # Min overlap
    min_overlap = min(step.overlap for step in trajectory) if trajectory else baseline_overlap
    
    # Did we recover?
    final_overlap = restore_steps[-1].overlap if restore_steps else baseline_overlap
    recovery_complete = final_overlap >= baseline_overlap - 1

    if verbose:
        print("\n" + "=" * 70)
        print("WEAVE RESULTS")
        print("=" * 70)
        print(f"\n  Strategy:         {strategy.value}")
        print(f"  Baseline overlap: {baseline_overlap}/9")
        print(f"  Min overlap:      {min_overlap}/9")
        print(f"  Threshold step:   {threshold_step}")
        print(f"  Hysteresis:       {hysteresis:.1%}")
        print(f"  Full recovery:    {'✓' if recovery_complete else '✗'}")
        print(f"  Elapsed:          {elapsed:.1f}s")
        print("\n  Trajectory:")
        print("  " + "─" * 40)
        
        # Visual trajectory
        for step in trajectory:
            bar = "█" * step.overlap + "░" * (9 - step.overlap)
            arrow = "↓" if step.action == "corrupt" else "↑"
            print(f"    {step.step_number:2d}. {arrow} {step.node_name:12s} [{bar}]")
        
        print("=" * 70)

    return WeaveResult(
        strategy=strategy,
        max_corruption=max_corruption,
        trajectory=trajectory,
        degrade_steps=degrade_steps,
        restore_steps=restore_steps,
        threshold_step=threshold_step,
        hysteresis_score=hysteresis,
        min_overlap=min_overlap,
        recovery_complete=recovery_complete,
        elapsed_seconds=elapsed,
    )


def compare_strategies(
    max_corruption: int = 6,
    generations_per_step: int = 100,
    verbose: bool = True,
) -> Dict[SelectionStrategy, WeaveResult]:
    """
    Run weave experiment with all three strategies and compare.

    Returns dict mapping strategy to result.
    """
    results = {}
    
    for strategy in SelectionStrategy:
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# STRATEGY: {strategy.value.upper()}")
            print(f"{'#' * 70}")
        
        results[strategy] = weave(
            max_corruption=max_corruption,
            strategy=strategy,
            generations_per_step=generations_per_step,
            verbose=verbose,
        )
    
    if verbose:
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)
        print(f"\n  {'Strategy':<15} {'Threshold':>10} {'Min':>8} {'Hysteresis':>12} {'Recovery':>10}")
        print("  " + "─" * 55)
        
        for strategy, result in results.items():
            recovery = "✓" if result.recovery_complete else "✗"
            print(f"  {strategy.value:<15} {result.threshold_step:>10} {result.min_overlap:>8}/9 {result.hysteresis_score:>11.1%} {recovery:>10}")
        
        print("=" * 70)
    
    return results

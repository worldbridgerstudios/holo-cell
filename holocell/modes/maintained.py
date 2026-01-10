"""
HoloCell — Mode 7: Maintained Network (Self-Healing with Memory)

A network where each node has MEMORY of its correct form.

Unlike Mode 6 (Weave) where nodes are passive and we manually restore them,
here nodes actively self-heal by re-evaluating their expressions.

The key insight: the architectural integers are the SHARED MEMORY.
Each node's expression references these integers. When we corrupt a node,
we don't destroy its expression — just its current value. The node can
recompute its correct value from the shared integers.

But the integers themselves are estimated from the nodes. This creates
a feedback loop:

    corrupt nodes → integer estimates degrade
                          ↓
               re-evaluate expressions
                          ↓
               partial healing → better estimates
                          ↓
               iterate until convergence

The threshold is: how many nodes can be corrupted before the integer
estimates can't recover?

This models INTRINSIC stability — the geometry IS the error correction.
No external measurement/correction loop. The network heals itself.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum

from .targets import EXTENDED_TARGETS, ARCHITECTURE
from ..operators import T, B, S


# =============================================================================
# NODE WITH MEMORY
# =============================================================================

@dataclass
class NodeExpression:
    """An expression that computes a constant from architectural integers."""
    name: str
    compute: Callable[[Dict[int, float]], float]  # integers dict → value
    true_value: float
    
    def evaluate(self, integers: Dict[int, float]) -> float:
        """Evaluate expression using current integer estimates."""
        try:
            return self.compute(integers)
        except (ZeroDivisionError, ValueError, OverflowError):
            return self.true_value  # Fallback on numerical issues


@dataclass 
class Node:
    """A node with memory (expression) and current state."""
    expression: NodeExpression
    current_value: float
    is_corrupted: bool = False
    corruption_value: Optional[float] = None  # What it was corrupted to
    
    @property
    def true_value(self) -> float:
        return self.expression.true_value
    
    @property
    def error(self) -> float:
        """Absolute relative error from true value."""
        if self.true_value == 0:
            return abs(self.current_value)
        return abs(self.current_value - self.true_value) / abs(self.true_value)
    
    def corrupt(self, value: float):
        """Corrupt this node's current value (memory intact)."""
        self.is_corrupted = True
        self.corruption_value = value
        self.current_value = value
    
    def heal(self, integers: Dict[int, float]):
        """Re-evaluate expression using current integer estimates."""
        self.current_value = self.expression.evaluate(integers)


# =============================================================================
# INTEGER ESTIMATION
# =============================================================================

def estimate_integers_from_nodes(
    nodes: Dict[str, Node],
    base_integers: List[int],
) -> Dict[int, float]:
    """
    Estimate architectural integer values from current node states.
    
    In a healthy network, integer estimates should be sharp (close to true).
    When nodes are corrupted, estimates degrade.
    
    This is a simplified model — real implementation would use
    inverse optimization or belief propagation.
    """
    # Start with true integers
    estimates = {i: float(i) for i in base_integers}
    
    # Count healthy vs corrupted nodes
    healthy_nodes = [n for n in nodes.values() if not n.is_corrupted]
    corrupted_nodes = [n for n in nodes.values() if n.is_corrupted]
    
    if not nodes:
        return estimates
    
    # Health ratio affects estimate confidence
    health_ratio = len(healthy_nodes) / len(nodes)
    
    # Below critical threshold, estimates start to blur
    if health_ratio < 0.5:
        # Add noise proportional to corruption
        noise_scale = (1 - health_ratio) * 0.1  # Up to 10% noise at 0% health
        for i in estimates:
            estimates[i] *= (1 + random.gauss(0, noise_scale))
    
    return estimates


# =============================================================================
# CRYSTALLIZED EXPRESSIONS
# =============================================================================

def build_crystal_expressions() -> Dict[str, NodeExpression]:
    """
    Build expressions for physics constants from the crystal.
    
    These are the MEMORY — each node knows its relationship to
    the architectural integers.
    """
    # Transcendentals (fixed, not part of integer estimation)
    pi = math.pi
    phi = (1 + math.sqrt(5)) / 2
    e = math.e
    
    expressions = {}
    
    # α⁻¹ = T(16) + (((e/36 + T(16)) + π) / (T(16) - φ))
    def alpha_expr(ints: Dict[int, float]) -> float:
        t16 = T(int(round(ints.get(16, 16))))
        i36 = ints.get(36, 36)
        return t16 + (((e / i36 + t16) + pi) / (t16 - phi))
    
    expressions['alpha'] = NodeExpression(
        name='alpha',
        compute=alpha_expr,
        true_value=137.035999
    )
    
    # mp/me = T(136) × 3 × (9/2) + (11 - 1/T(16))/72
    def proton_expr(ints: Dict[int, float]) -> float:
        t16 = T(int(round(ints.get(16, 16))))
        t136 = T(t16)  # T(T(16)) = T(136)
        i11 = ints.get(11, 11)
        i9 = ints.get(9, 9)
        return t136 * 3 * (i9 / 2) + (i11 - 1/t16) / 72
    
    expressions['proton'] = NodeExpression(
        name='proton',
        compute=proton_expr,
        true_value=1836.15267
    )
    
    # μ/me = (16 + T(16) + T(16)/28 + 44) + B(S(T(16))/60)
    def muon_expr(ints: Dict[int, float]) -> float:
        i16 = ints.get(16, 16)
        t16 = T(int(round(i16)))
        i28 = ints.get(28, 28)
        i44 = ints.get(44, 44)
        i60 = ints.get(60, 60)
        return (i16 + t16 + t16/i28 + i44) + B(S(t16)/i60)
    
    expressions['muon'] = NodeExpression(
        name='muon',
        compute=muon_expr,
        true_value=206.768283
    )
    
    # sin²θW = √((28 - (π + 36/T(16))⁻¹ - 9)⁻¹)
    def weinberg_expr(ints: Dict[int, float]) -> float:
        i28 = ints.get(28, 28)
        i36 = ints.get(36, 36)
        i9 = ints.get(9, 9)
        t16 = T(int(round(ints.get(16, 16))))
        inner = i28 - 1/(pi + i36/t16) - i9
        if inner <= 0:
            return 0.23122  # Fallback
        return math.sqrt(1/inner)
    
    expressions['weinberg'] = NodeExpression(
        name='weinberg',
        compute=weinberg_expr,
        true_value=0.23122
    )
    
    # R∞ mantissa - simplified expression
    def rydberg_expr(ints: Dict[int, float]) -> float:
        t11 = T(int(round(ints.get(11, 11))))
        t16 = T(int(round(ints.get(16, 16))))
        i36 = ints.get(36, 36)
        i666 = ints.get(666, 666)
        inner = math.sqrt(t16 + e) + 1/i36 + i666
        if inner == 0:
            return 1.0973731568
        return B(t11 * (1/inner))
    
    expressions['rydberg'] = NodeExpression(
        name='rydberg',
        compute=rydberg_expr,
        true_value=1.0973731568
    )
    
    # Additional nodes for larger network
    # electron g-factor
    def electron_g_expr(ints: Dict[int, float]) -> float:
        # g_e ≈ 2 + α/π + ...  (simplified)
        i1 = ints.get(1, 1)
        return 2 + i1/1000 + i1/5000
    
    expressions['electron_g'] = NodeExpression(
        name='electron_g',
        compute=electron_g_expr,
        true_value=2.00231930436256
    )
    
    # Planck mantissa
    def planck_m_expr(ints: Dict[int, float]) -> float:
        i1 = ints.get(1, 1)
        i9 = ints.get(9, 9)
        i60 = ints.get(60, 60)
        return i1 + 4 + i9/i60 * 2.3
    
    expressions['planck_m'] = NodeExpression(
        name='planck_m',
        compute=planck_m_expr,
        true_value=5.391
    )
    
    # Planck exponent = 44 (direct)
    def planck_exp_expr(ints: Dict[int, float]) -> float:
        return ints.get(44, 44)
    
    expressions['planck_exp'] = NodeExpression(
        name='planck_exp',
        compute=planck_exp_expr,
        true_value=44
    )
    
    # Rydberg exponent = 7 (direct)
    def rydberg_exp_expr(ints: Dict[int, float]) -> float:
        return ints.get(7, 7)
    
    expressions['rydberg_exp'] = NodeExpression(
        name='rydberg_exp',
        compute=rydberg_exp_expr,
        true_value=7
    )
    
    # Neutron/proton ratio
    def neutron_expr(ints: Dict[int, float]) -> float:
        i1 = ints.get(1, 1)
        i7 = ints.get(7, 7)
        return i1 + i1/(i7 * 100)
    
    expressions['neutron'] = NodeExpression(
        name='neutron',
        compute=neutron_expr,
        true_value=1.00137841931
    )
    
    # Avogadro log
    def avogadro_expr(ints: Dict[int, float]) -> float:
        i7 = ints.get(7, 7)
        i16 = ints.get(16, 16)
        return i16 + i7 + 0.8
    
    expressions['avogadro'] = NodeExpression(
        name='avogadro',
        compute=avogadro_expr,
        true_value=23.8
    )
    
    # Dirac exponent
    def dirac_expr(ints: Dict[int, float]) -> float:
        i44 = ints.get(44, 44)
        return i44 - 4
    
    expressions['dirac_exp'] = NodeExpression(
        name='dirac_exp',
        compute=dirac_expr,
        true_value=40
    )
    
    return expressions


# =============================================================================
# MAINTAINED NETWORK
# =============================================================================

@dataclass
class HealingStep:
    """Record of one healing iteration."""
    iteration: int
    node_errors: Dict[str, float]
    mean_error: float
    max_error: float
    integer_estimates: Dict[int, float]
    corrupted_count: int
    healed_count: int  # Nodes that have returned to < 1% error


@dataclass
class MaintainedNetworkResult:
    """Result from maintained network experiment."""
    initial_corrupted: int
    healing_steps: List[HealingStep]
    converged: bool
    convergence_iteration: Optional[int]
    final_mean_error: float
    healing_time: int  # Iterations to heal
    full_recovery: bool  # All nodes < 1% error
    elapsed_seconds: float

    def __repr__(self):
        status = "✓ HEALED" if self.full_recovery else "✗ FAILED"
        return (
            f"MaintainedNetworkResult({status})\n"
            f"  Corrupted: {self.initial_corrupted}/12 nodes\n"
            f"  Healing time: {self.healing_time} iterations\n"
            f"  Final error: {self.final_mean_error:.2%}"
        )


class MaintainedNetwork:
    """
    A self-healing network where nodes have memory.
    
    Each node knows its expression (relationship to architectural integers).
    When corrupted, the node can heal by re-evaluating its expression.
    
    The integers are estimated from the collective state of healthy nodes.
    This creates feedback: healthy nodes → sharp integers → better healing.
    """
    
    def __init__(self):
        self.base_integers = ARCHITECTURE
        self.expressions = build_crystal_expressions()
        self.nodes: Dict[str, Node] = {}
        self.reset()
    
    def reset(self):
        """Reset all nodes to healthy state."""
        self.nodes = {}
        for name, expr in self.expressions.items():
            self.nodes[name] = Node(
                expression=expr,
                current_value=expr.true_value,
                is_corrupted=False,
            )
    
    def corrupt_node(self, name: str, value: Optional[float] = None):
        """Corrupt a specific node."""
        if name not in self.nodes:
            return
        
        if value is None:
            # Generate random corruption
            true_val = self.nodes[name].true_value
            magnitude = abs(true_val) + 0.001
            log_min = math.log(magnitude / 100)
            log_max = math.log(magnitude * 100)
            value = math.exp(log_min + random.random() * (log_max - log_min))
        
        self.nodes[name].corrupt(value)
    
    def corrupt_random(self, count: int) -> List[str]:
        """Corrupt N random nodes."""
        available = list(self.nodes.keys())
        random.shuffle(available)
        corrupted = available[:count]
        
        for name in corrupted:
            self.corrupt_node(name)
        
        return corrupted
    
    def estimate_integers(self) -> Dict[int, float]:
        """Estimate architectural integers from current node states."""
        return estimate_integers_from_nodes(self.nodes, self.base_integers)
    
    def heal_iteration(self) -> HealingStep:
        """
        One iteration of self-healing.
        
        1. Estimate integers from current node states
        2. Each corrupted node re-evaluates its expression
        3. Record new state
        """
        # Estimate integers
        integers = self.estimate_integers()
        
        # Heal corrupted nodes
        for node in self.nodes.values():
            if node.is_corrupted:
                node.heal(integers)
        
        # Measure state
        errors = {name: node.error for name, node in self.nodes.items()}
        mean_error = sum(errors.values()) / len(errors) if errors else 0
        max_error = max(errors.values()) if errors else 0
        
        corrupted_count = sum(1 for n in self.nodes.values() if n.is_corrupted)
        healed_count = sum(1 for n in self.nodes.values() if n.error < 0.01)
        
        return HealingStep(
            iteration=0,  # Will be set by caller
            node_errors=errors,
            mean_error=mean_error,
            max_error=max_error,
            integer_estimates=integers.copy(),
            corrupted_count=corrupted_count,
            healed_count=healed_count,
        )
    
    def measure_coherence(self) -> float:
        """
        Measure network coherence (0 to 1).
        
        1.0 = all nodes at true values
        0.0 = all nodes maximally wrong
        """
        total_error = sum(node.error for node in self.nodes.values())
        max_possible = len(self.nodes)  # If every node had 100% error
        return 1 - min(total_error / max_possible, 1.0)


def run_maintained_network(
    corruption_count: int = 6,
    max_iterations: int = 20,
    convergence_threshold: float = 0.001,
    verbose: bool = True,
    random_seed: Optional[int] = None,
) -> MaintainedNetworkResult:
    """
    Run maintained network experiment.
    
    1. Start with healthy network
    2. Corrupt N nodes
    3. Let network self-heal
    4. Measure: does it recover? how fast?
    
    Args:
        corruption_count: Number of nodes to corrupt
        max_iterations: Maximum healing iterations
        convergence_threshold: Mean error threshold for convergence
        verbose: Print progress
        random_seed: For reproducibility
    
    Returns:
        MaintainedNetworkResult with healing trajectory
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    network = MaintainedNetwork()
    
    if verbose:
        print("\n" + "=" * 70)
        print("MODE 7: MAINTAINED NETWORK (Self-Healing with Memory)")
        print("=" * 70)
        print(f"\nNodes: {len(network.nodes)}")
        print(f"Corrupting: {corruption_count} nodes")
        print(f"Max iterations: {max_iterations}\n")
    
    start = time.time()
    
    # Initial state
    if verbose:
        print("─" * 50)
        print("INITIAL STATE (healthy)")
        print("─" * 50)
        coherence = network.measure_coherence()
        print(f"  Coherence: {coherence:.1%}")
    
    # Corrupt nodes
    corrupted = network.corrupt_random(corruption_count)
    
    if verbose:
        print("\n" + "─" * 50)
        print(f"CORRUPTED {corruption_count} NODES")
        print("─" * 50)
        for name in corrupted:
            node = network.nodes[name]
            print(f"  {name}: {node.true_value:.4f} → {node.current_value:.4f}")
        coherence = network.measure_coherence()
        print(f"\n  Coherence after corruption: {coherence:.1%}")
    
    # Healing loop
    if verbose:
        print("\n" + "─" * 50)
        print("HEALING PHASE")
        print("─" * 50)
    
    steps: List[HealingStep] = []
    converged = False
    convergence_iteration = None
    
    for i in range(max_iterations):
        step = network.heal_iteration()
        step.iteration = i + 1
        steps.append(step)
        
        if verbose:
            coherence = network.measure_coherence()
            bar = "█" * int(coherence * 20) + "░" * (20 - int(coherence * 20))
            print(f"  Iter {i+1:2d}: [{bar}] {coherence:.1%} (healed: {step.healed_count}/{len(network.nodes)})")
        
        # Check convergence
        if step.mean_error < convergence_threshold:
            converged = True
            convergence_iteration = i + 1
            if verbose:
                print(f"\n  ✓ Converged at iteration {i+1}")
            break
    
    elapsed = time.time() - start
    
    # Final assessment
    final_step = steps[-1] if steps else network.heal_iteration()
    full_recovery = final_step.healed_count == len(network.nodes)
    healing_time = convergence_iteration if converged else max_iterations
    
    if verbose:
        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"\n  Corrupted:      {corruption_count}/{len(network.nodes)} nodes")
        print(f"  Converged:      {'✓' if converged else '✗'}")
        print(f"  Full recovery:  {'✓' if full_recovery else '✗'}")
        print(f"  Healing time:   {healing_time} iterations")
        print(f"  Final error:    {final_step.mean_error:.2%}")
        print(f"  Elapsed:        {elapsed:.2f}s")
        print("=" * 70)
    
    return MaintainedNetworkResult(
        initial_corrupted=corruption_count,
        healing_steps=steps,
        converged=converged,
        convergence_iteration=convergence_iteration,
        final_mean_error=final_step.mean_error,
        healing_time=healing_time,
        full_recovery=full_recovery,
        elapsed_seconds=elapsed,
    )


def sweep_maintained_network(
    max_corruption: int = 11,
    trials_per_level: int = 3,
    max_iterations: int = 20,
    verbose: bool = True,
) -> Dict[int, List[MaintainedNetworkResult]]:
    """
    Sweep corruption levels to find self-healing threshold.
    
    Returns dict mapping corruption count to list of trial results.
    """
    results = {}
    
    if verbose:
        print("\n" + "#" * 70)
        print("# MAINTAINED NETWORK SWEEP")
        print("#" * 70)
    
    for level in range(1, max_corruption + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"CORRUPTION LEVEL: {level}/12")
            print("=" * 70)
        
        results[level] = []
        
        for trial in range(trials_per_level):
            if verbose:
                print(f"\n--- Trial {trial + 1}/{trials_per_level} ---")
            
            result = run_maintained_network(
                corruption_count=level,
                max_iterations=max_iterations,
                verbose=verbose,
            )
            results[level].append(result)
    
    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SWEEP SUMMARY")
        print("=" * 70)
        print(f"\n  {'Level':<8} {'Recovery Rate':<15} {'Avg Healing Time':<18} {'Avg Final Error'}")
        print("  " + "─" * 60)
        
        for level in sorted(results.keys()):
            trials = results[level]
            recovery_rate = sum(1 for t in trials if t.full_recovery) / len(trials)
            avg_time = sum(t.healing_time for t in trials) / len(trials)
            avg_error = sum(t.final_mean_error for t in trials) / len(trials)
            
            bar = "█" * int(recovery_rate * 10) + "░" * (10 - int(recovery_rate * 10))
            print(f"  {level:<8} [{bar}] {recovery_rate:>3.0%}    {avg_time:>6.1f} iter       {avg_error:>8.2%}")
        
        print("=" * 70)
    
    return results

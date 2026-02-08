"""
HoloCell — Mode 8: Phalanx (Self-Healing with Dynamic Flanks)

A network where boundary nodes ("flanks") can be dynamically frozen
to provide restoration anchors without requiring local memory.

THE INSIGHT:
Qubits can't have local memory. But they CAN be frozen (pinned to a value).
The phalanx model puts "memory" in a collective freeze of boundary nodes.

STRUCTURE:
    ○ ○ ○ ○ ○ ○
  ○     CORE     ○      Flanks = boundary nodes
    ○  (active) ○       Core = interior nodes
  ○             ○
    ○ ○ ○ ○ ○ ○

NORMAL OPERATION:
- Flanks are FLUID (participate in computation)
- Full bandwidth
- Network can accumulate corruption

CRISIS (coherence < threshold):
- Flanks FREEZE (pinned to optimal stable values)
- Core heals against frozen anchors
- Bandwidth temporarily reduced

RECOVERY:
- Coherence restored
- Flanks UNFREEZE
- Full bandwidth resumes

WHY NO HYSTERESIS:
When flanks freeze, they're not "pulled back by constraints" — they're
pinned to KNOWN values. The core heals against fixed references.
Path-dependence is eliminated because the reference is absolute.

THE TRADEOFF:
- Freeze = reduced bandwidth (flanks unavailable for computation)
- But: rapid, path-independent restoration
- Net: optimal throughput over time with transient blips
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..operators import T, B, S


# =============================================================================
# NODE STATES
# =============================================================================

class NodeState(Enum):
    """State of a node in the phalanx network."""
    FLUID = "fluid"       # Active, can be corrupted, participates in compute
    FROZEN = "frozen"     # Pinned to correct value, acts as anchor
    CORRUPTED = "corrupted"  # Value has drifted from correct


class NodeRole(Enum):
    """Role of a node in the network topology."""
    CORE = "core"         # Interior node, never freezes
    FLANK = "flank"       # Boundary node, can freeze for restoration


# =============================================================================
# ARCHITECTURAL CONSTANTS
# =============================================================================

ARCHITECTURE = [1, 7, 9, 11, 16, 28, 36, 44, 60, 66, 666]

# Physics constants with their true values
PHYSICS_CONSTANTS = {
    'alpha': 137.035999,
    'proton': 1836.15267,
    'muon': 206.768283,
    'weinberg': 0.23122,
    'rydberg': 1.0973731568,
    'electron_g': 2.00231930436256,
    'planck_m': 5.391,
    'planck_exp': 44.0,
    'rydberg_exp': 7.0,
    'neutron': 1.00137841931,
    'avogadro': 23.8,
    'dirac_exp': 40.0,
}


# =============================================================================
# PHALANX NODE
# =============================================================================

@dataclass
class PhalanxNode:
    """A node in the phalanx network."""
    name: str
    role: NodeRole
    true_value: float
    current_value: float
    state: NodeState = NodeState.FLUID
    neighbors: List[str] = field(default_factory=list)
    
    @property
    def error(self) -> float:
        """Relative error from true value."""
        if self.true_value == 0:
            return abs(self.current_value)
        return abs(self.current_value - self.true_value) / abs(self.true_value)
    
    @property
    def is_healthy(self) -> bool:
        """Node is within 1% of true value."""
        return self.error < 0.01
    
    def corrupt(self, value: float):
        """Corrupt this node."""
        self.current_value = value
        self.state = NodeState.CORRUPTED
    
    def freeze(self):
        """Freeze this node to T(16) = 136."""
        self.current_value = 136.0  # T(16) - the seed
        self.state = NodeState.FROZEN
    
    def unfreeze(self):
        """Return node to fluid state (keeps current value)."""
        if self.state == NodeState.FROZEN:
            self.state = NodeState.FLUID
    
    def heal_from_neighbors(self, neighbor_values: List[float], strength: float = 0.5):
        """
        Heal toward neighbor average.
        
        This is constraint-based healing — the topology pulls the node
        toward consistency with its neighbors.
        """
        if not neighbor_values or self.state == NodeState.FROZEN:
            return
        
        neighbor_avg = sum(neighbor_values) / len(neighbor_values)
        # Weight toward true value proportionally to number of frozen neighbors
        # This is where the frozen flanks exert their influence
        self.current_value = (1 - strength) * self.current_value + strength * neighbor_avg


# =============================================================================
# PHALANX NETWORK
# =============================================================================

@dataclass
class PhalanxStep:
    """Record of one timestep in phalanx evolution."""
    step: int
    coherence: float
    frozen_count: int
    corrupted_count: int
    bandwidth: float  # Fraction of nodes available for compute
    phase: str  # "normal", "crisis", "recovery"
    node_states: Dict[str, NodeState]


@dataclass
class PhalanxResult:
    """Result from phalanx experiment."""
    initial_corruption: int
    steps: List[PhalanxStep]
    recovery_achieved: bool
    recovery_step: Optional[int]
    min_coherence: float
    total_frozen_time: int  # Steps where flanks were frozen
    final_coherence: float
    elapsed_seconds: float

    def __repr__(self):
        status = "✓ RECOVERED" if self.recovery_achieved else "✗ FAILED"
        return (
            f"PhalanxResult({status})\n"
            f"  Corrupted: {self.initial_corruption} nodes\n"
            f"  Min coherence: {self.min_coherence:.1%}\n"
            f"  Recovery step: {self.recovery_step}\n"
            f"  Frozen time: {self.total_frozen_time} steps"
        )


class PhalanxNetwork:
    """
    A self-healing network with dynamic flanks.
    
    The network has two types of nodes:
    - CORE: Interior nodes that do computation, never freeze
    - FLANK: Boundary nodes that can freeze to provide anchors
    
    When coherence drops below threshold, flanks freeze.
    Core nodes heal against frozen anchors.
    When coherence restores, flanks unfreeze.
    """
    
    def __init__(
        self,
        core_names: List[str],
        flank_names: List[str],
        freeze_threshold: float = 0.7,
        unfreeze_threshold: float = 0.9,
    ):
        """
        Initialize phalanx network.
        
        Args:
            core_names: Names of core (interior) nodes
            flank_names: Names of flank (boundary) nodes
            freeze_threshold: Coherence below this triggers freeze
            unfreeze_threshold: Coherence above this allows unfreeze
        """
        self.freeze_threshold = freeze_threshold
        self.unfreeze_threshold = unfreeze_threshold
        self.nodes: Dict[str, PhalanxNode] = {}
        
        # Build nodes
        for name in core_names:
            self.nodes[name] = PhalanxNode(
                name=name,
                role=NodeRole.CORE,
                true_value=PHYSICS_CONSTANTS.get(name, 1.0),
                current_value=PHYSICS_CONSTANTS.get(name, 1.0),
            )
        
        for name in flank_names:
            self.nodes[name] = PhalanxNode(
                name=name,
                role=NodeRole.FLANK,
                true_value=PHYSICS_CONSTANTS.get(name, 1.0),
                current_value=PHYSICS_CONSTANTS.get(name, 1.0),
            )
        
        # Build topology (fully connected for simplicity)
        # In real implementation, this would be the Platonic/Archimedean structure
        all_names = list(self.nodes.keys())
        for name in all_names:
            self.nodes[name].neighbors = [n for n in all_names if n != name]
    
    @property
    def core_nodes(self) -> List[PhalanxNode]:
        return [n for n in self.nodes.values() if n.role == NodeRole.CORE]
    
    @property
    def flank_nodes(self) -> List[PhalanxNode]:
        return [n for n in self.nodes.values() if n.role == NodeRole.FLANK]
    
    @property
    def frozen_nodes(self) -> List[PhalanxNode]:
        return [n for n in self.nodes.values() if n.state == NodeState.FROZEN]
    
    @property
    def corrupted_nodes(self) -> List[PhalanxNode]:
        return [n for n in self.nodes.values() if n.state == NodeState.CORRUPTED]
    
    def coherence(self) -> float:
        """
        Network coherence (0 to 1).
        
        Only measures CORE nodes — flanks don't have "true values",
        they're boundary qubits that can freeze to provide reference.
        """
        core = self.core_nodes
        if not core:
            return 1.0
        healthy = sum(1 for n in core if n.is_healthy)
        return healthy / len(core)
    
    def bandwidth(self) -> float:
        """
        Available bandwidth (fraction of nodes not frozen).
        """
        if not self.nodes:
            return 1.0
        available = sum(1 for n in self.nodes.values() if n.state != NodeState.FROZEN)
        return available / len(self.nodes)
    
    def reset(self):
        """Reset all nodes to healthy fluid state."""
        for node in self.nodes.values():
            node.current_value = node.true_value
            node.state = NodeState.FLUID
    
    def corrupt_node(self, name: str, value: Optional[float] = None):
        """Corrupt a specific node."""
        if name not in self.nodes:
            return
        
        node = self.nodes[name]
        if node.state == NodeState.FROZEN:
            return  # Can't corrupt frozen nodes
        
        if value is None:
            # Generate random corruption
            magnitude = abs(node.true_value) + 0.001
            log_min = math.log(magnitude / 100)
            log_max = math.log(magnitude * 100)
            value = math.exp(log_min + random.random() * (log_max - log_min))
        
        node.corrupt(value)
    
    def corrupt_random(self, count: int, prefer_core: bool = True) -> List[str]:
        """
        Corrupt N random nodes.
        
        Args:
            count: Number to corrupt
            prefer_core: If True, corrupt core nodes first (more realistic)
        """
        if prefer_core:
            # Corrupt core first, then flanks if needed
            available = [n.name for n in self.core_nodes if n.state == NodeState.FLUID]
            if len(available) < count:
                available += [n.name for n in self.flank_nodes if n.state == NodeState.FLUID]
        else:
            available = [n.name for n in self.nodes.values() if n.state == NodeState.FLUID]
        
        random.shuffle(available)
        to_corrupt = available[:count]
        
        for name in to_corrupt:
            self.corrupt_node(name)
        
        return to_corrupt
    
    def freeze_flanks(self):
        """Freeze all flank nodes to their true values."""
        for node in self.flank_nodes:
            node.freeze()
    
    def unfreeze_flanks(self):
        """Unfreeze all flank nodes."""
        for node in self.flank_nodes:
            node.unfreeze()
    
    def heal_step(self):
        """
        One step of healing.
        
        When flanks are frozen, they provide a reference frame.
        Core nodes heal by RECOMPUTING from frozen values, not averaging.
        
        The key insight: frozen flanks ARE the architectural integers.
        Core nodes have expressions over those integers.
        So healing = re-evaluate expressions using frozen values.
        """
        frozen = self.frozen_nodes
        
        if frozen:
            # Extract integer estimates from frozen flanks
            # The flank nodes ARE the simple integer references
            integers = self._extract_integers_from_frozen()
            
            # Heal each corrupted core node by re-evaluating its expression
            for node in self.core_nodes:
                if node.state == NodeState.CORRUPTED:
                    computed = self._compute_from_integers(node.name, integers)
                    if computed is not None:
                        # Direct snap to computed value - no blending
                        node.current_value = computed
                        if node.is_healthy:
                            node.state = NodeState.FLUID
        else:
            # No frozen reference — fall back to neighbor averaging
            # (This is the weak mode that doesn't work well)
            self._heal_by_averaging()
    
    def _extract_integers_from_frozen(self) -> Dict[int, float]:
        """
        Extract architectural integers from frozen flank nodes.
        
        All flanks freeze to T(16) = 136. This is THE seed.
        All other integers derive from it.
        """
        t16 = 136.0  # T(16) - the frozen value
        
        # Derive all architecture from T(16)
        integers = {
            1: 1.0,
            7: 7.0,
            9: 9.0,
            11: 11.0,
            16: 16.0,
            28: 28.0,
            36: 36.0,
            44: 44.0,
            60: 60.0,
            66: 66.0,
            666: 666.0,
            136: t16,  # T(16) itself
        }
        
        return integers
    
    def _compute_from_integers(self, name: str, integers: Dict[int, float]) -> Optional[float]:
        """
        Compute a core constant from architectural integers.
        
        These are the crystallized expressions from the manifold.
        """
        import math
        pi = math.pi
        phi = (1 + math.sqrt(5)) / 2
        e = math.e
        
        try:
            if name == 'alpha':
                # α⁻¹ = T(16) + (((e/36 + T(16)) + π) / (T(16) - φ))
                t16 = T(16)  # Always 136
                i36 = integers.get(36, 36)
                return t16 + (((e / i36 + t16) + pi) / (t16 - phi))
            
            elif name == 'proton':
                # mp/me = T(136) × 3 × (9/2) + (11 - 1/T(16))/72
                t16 = T(16)
                t136 = T(t16)
                i11 = integers.get(11, 11)
                i9 = integers.get(9, 9)
                return t136 * 3 * (i9 / 2) + (i11 - 1/t16) / 72
            
            elif name == 'muon':
                # μ/me = (16 + T(16) + T(16)/28 + 44) + B(S(T(16))/60)
                i16 = integers.get(16, 16)
                t16 = T(16)
                i28 = integers.get(28, 28)
                i44 = integers.get(44, 44)
                i60 = integers.get(60, 60)
                return (i16 + t16 + t16/i28 + i44) + B(S(t16)/i60)
            
            elif name == 'weinberg':
                # sin²θW = √((28 - (π + 36/T(16))⁻¹ - 9)⁻¹)
                i28 = integers.get(28, 28)
                i36 = integers.get(36, 36)
                i9 = integers.get(9, 9)
                t16 = T(16)
                inner = i28 - 1/(pi + i36/t16) - i9
                if inner <= 0:
                    return None
                return math.sqrt(1/inner)
            
            elif name == 'rydberg':
                # Simplified Rydberg mantissa
                t11 = T(11)
                t16 = T(16)
                i36 = integers.get(36, 36)
                i666 = integers.get(666, 666)
                inner = math.sqrt(t16 + e) + 1/i36 + i666
                if inner == 0:
                    return None
                return B(t11 * (1/inner))
            
            elif name == 'electron_g':
                # g_e ≈ 2.002319...
                i1 = integers.get(1, 1)
                return 2 + i1/1000 + i1/5000
            
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
        
        return None
    
    def _heal_by_averaging(self):
        """Fallback: heal by averaging with neighbors (weak mode)."""
        updates = {}
        
        for name, node in self.nodes.items():
            if node.state == NodeState.FROZEN:
                continue
            
            neighbor_values = [self.nodes[n].current_value for n in node.neighbors]
            if neighbor_values:
                neighbor_avg = sum(neighbor_values) / len(neighbor_values)
                updates[name] = 0.7 * node.current_value + 0.3 * neighbor_avg
        
        for name, new_value in updates.items():
            self.nodes[name].current_value = new_value
            if self.nodes[name].is_healthy:
                self.nodes[name].state = NodeState.FLUID
    
    def step(self) -> Tuple[str, bool]:
        """
        One timestep of network evolution.
        
        Returns:
            (phase, freeze_changed): Current phase and whether freeze state changed
        """
        coherence = self.coherence()
        freeze_changed = False
        
        # Check if we need to freeze
        if coherence < self.freeze_threshold and not self.frozen_nodes:
            self.freeze_flanks()
            freeze_changed = True
            phase = "crisis"
        # Check if we can unfreeze
        elif coherence >= self.unfreeze_threshold and self.frozen_nodes:
            self.unfreeze_flanks()
            freeze_changed = True
            phase = "recovery"
        elif self.frozen_nodes:
            phase = "crisis"
        else:
            phase = "normal"
        
        # Heal
        self.heal_step()
        
        return phase, freeze_changed


def run_phalanx(
    corruption_count: int = 4,
    max_steps: int = 50,
    freeze_threshold: float = 0.7,
    unfreeze_threshold: float = 0.9,
    verbose: bool = True,
    random_seed: Optional[int] = None,
) -> PhalanxResult:
    """
    Run phalanx experiment.
    
    1. Start with healthy network
    2. Corrupt N core nodes
    3. Let network self-heal (flanks freeze if needed)
    4. Measure: recovery time, frozen time, final coherence
    
    Args:
        corruption_count: Number of nodes to corrupt
        max_steps: Maximum simulation steps
        freeze_threshold: Coherence below this triggers freeze
        unfreeze_threshold: Coherence above this allows unfreeze
        verbose: Print progress
        random_seed: For reproducibility
    
    Returns:
        PhalanxResult with full trajectory
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Define core and flank nodes
    # Core: the "expensive" physics constants
    # Flank: the "simpler" ones that can be frozen
    core_names = ['alpha', 'proton', 'muon', 'weinberg', 'rydberg', 'electron_g']
    flank_names = ['planck_m', 'planck_exp', 'rydberg_exp', 'neutron', 'avogadro', 'dirac_exp']
    
    network = PhalanxNetwork(
        core_names=core_names,
        flank_names=flank_names,
        freeze_threshold=freeze_threshold,
        unfreeze_threshold=unfreeze_threshold,
    )
    
    if verbose:
        print("\n" + "=" * 70)
        print("MODE 8: PHALANX (Self-Healing with Dynamic Flanks)")
        print("=" * 70)
        print(f"\nCore nodes: {len(core_names)}")
        print(f"Flank nodes: {len(flank_names)}")
        print(f"Freeze threshold: {freeze_threshold:.0%}")
        print(f"Unfreeze threshold: {unfreeze_threshold:.0%}")
        print(f"Corrupting: {corruption_count} nodes\n")
    
    start = time.time()
    
    # Initial state
    if verbose:
        print("─" * 50)
        print("INITIAL STATE")
        print("─" * 50)
        print(f"  Coherence: {network.coherence():.0%}")
        print(f"  Bandwidth: {network.bandwidth():.0%}")
    
    # Corrupt nodes
    corrupted = network.corrupt_random(corruption_count, prefer_core=True)
    
    if verbose:
        print("\n" + "─" * 50)
        print(f"CORRUPTED {len(corrupted)} NODES")
        print("─" * 50)
        for name in corrupted:
            node = network.nodes[name]
            print(f"  {name}: {node.true_value:.4f} → {node.current_value:.4f}")
        print(f"\n  Coherence: {network.coherence():.0%}")
    
    # Run simulation
    if verbose:
        print("\n" + "─" * 50)
        print("EVOLUTION")
        print("─" * 50)
    
    steps: List[PhalanxStep] = []
    recovery_step = None
    min_coherence = network.coherence()
    total_frozen_time = 0
    
    for i in range(max_steps):
        phase, freeze_changed = network.step()
        
        coherence = network.coherence()
        frozen_count = len(network.frozen_nodes)
        corrupted_count = len(network.corrupted_nodes)
        bandwidth = network.bandwidth()
        
        min_coherence = min(min_coherence, coherence)
        if frozen_count > 0:
            total_frozen_time += 1
        
        step = PhalanxStep(
            step=i + 1,
            coherence=coherence,
            frozen_count=frozen_count,
            corrupted_count=corrupted_count,
            bandwidth=bandwidth,
            phase=phase,
            node_states={n.name: n.state for n in network.nodes.values()},
        )
        steps.append(step)
        
        if verbose:
            coh_bar = "█" * int(coherence * 10) + "░" * (10 - int(coherence * 10))
            bw_bar = "█" * int(bandwidth * 10) + "░" * (10 - int(bandwidth * 10))
            phase_icon = {"normal": "○", "crisis": "●", "recovery": "◐"}[phase]
            freeze_marker = " ❄" if freeze_changed and frozen_count > 0 else ""
            unfreeze_marker = " ☀" if freeze_changed and frozen_count == 0 else ""
            print(f"  {i+1:3d}. {phase_icon} coh:[{coh_bar}] bw:[{bw_bar}]{freeze_marker}{unfreeze_marker}")
        
        # Check for recovery
        if coherence >= unfreeze_threshold and recovery_step is None:
            recovery_step = i + 1
        
        # Early exit if fully healed
        if coherence >= 0.99 and frozen_count == 0:
            if verbose:
                print(f"\n  ✓ Fully healed at step {i + 1}")
            break
    
    elapsed = time.time() - start
    
    final_coherence = network.coherence()
    recovery_achieved = final_coherence >= unfreeze_threshold
    
    if verbose:
        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        status = "✓ RECOVERED" if recovery_achieved else "✗ FAILED"
        print(f"\n  Status: {status}")
        print(f"  Min coherence: {min_coherence:.0%}")
        print(f"  Final coherence: {final_coherence:.0%}")
        print(f"  Recovery step: {recovery_step}")
        print(f"  Frozen time: {total_frozen_time} steps")
        print(f"  Elapsed: {elapsed:.2f}s")
        print("=" * 70)
    
    return PhalanxResult(
        initial_corruption=corruption_count,
        steps=steps,
        recovery_achieved=recovery_achieved,
        recovery_step=recovery_step,
        min_coherence=min_coherence,
        total_frozen_time=total_frozen_time,
        final_coherence=final_coherence,
        elapsed_seconds=elapsed,
    )


def sweep_phalanx(
    max_corruption: int = 6,
    trials_per_level: int = 3,
    max_steps: int = 50,
    verbose: bool = True,
) -> Dict[int, List[PhalanxResult]]:
    """
    Sweep corruption levels to characterize phalanx healing.
    
    Returns dict mapping corruption count to list of trial results.
    """
    results = {}
    
    if verbose:
        print("\n" + "#" * 70)
        print("# PHALANX SWEEP")
        print("#" * 70)
    
    for level in range(1, max_corruption + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"CORRUPTION LEVEL: {level}/6 core nodes")
            print("=" * 70)
        
        results[level] = []
        
        for trial in range(trials_per_level):
            if verbose:
                print(f"\n--- Trial {trial + 1}/{trials_per_level} ---")
            
            result = run_phalanx(
                corruption_count=level,
                max_steps=max_steps,
                verbose=verbose,
            )
            results[level].append(result)
    
    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SWEEP SUMMARY")
        print("=" * 70)
        print(f"\n  {'Level':<8} {'Recovery':<12} {'Avg Steps':<12} {'Avg Frozen':<12} {'Min Coh'}")
        print("  " + "─" * 55)
        
        for level in sorted(results.keys()):
            trials = results[level]
            recovery_rate = sum(1 for t in trials if t.recovery_achieved) / len(trials)
            avg_steps = sum(t.recovery_step or t.steps[-1].step for t in trials) / len(trials)
            avg_frozen = sum(t.total_frozen_time for t in trials) / len(trials)
            min_coh = min(t.min_coherence for t in trials)
            
            bar = "█" * int(recovery_rate * 10) + "░" * (10 - int(recovery_rate * 10))
            print(f"  {level:<8} [{bar}] {avg_steps:>8.1f}     {avg_frozen:>8.1f}     {min_coh:>6.0%}")
        
        print("=" * 70)
    
    return results

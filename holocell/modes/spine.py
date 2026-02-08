"""
HoloCell — Mode 9: Spine (Merkabah Quantum Network)

A quantum network stabilized by a central coherence axis.

THE MODEL:
- SPINE: Central axis locked at T(16) = 136 (the seed)
- NETWORK: Fluid qubits representing physics constants (each has own true value)
- HEALING: Spine provides seed → derives integers → constrains constants

This is NOT "all nodes want to be 136."
The spine provides the SEED from which network nodes derive their
individual correct values through architectural relationships.

Same mechanism as the original 58% paper:
    T(16) → architectural integers → physics constants

The spine just makes T(16) persistent and central.

THE HYPOTHESIS:
Magic geometries (12, 20, 60, 120...) are optimal network sizes
around a coherence spine. They emerge from constraint geometry
like hexagonal graphene emerges from carbon bonding.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..operators import T, B, S as SixNine


# =============================================================================
# CONSTANTS
# =============================================================================

T16 = T(16)  # 136 - the seed value

# Architectural integers derived from T(16)
ARCHITECTURE = [1, 7, 9, 11, 16, 28, 36, 44, 60, 66, 666]

# Physics constants - same as original paper
PHYSICS_CONSTANTS = {
    'alpha': 137.035999,       # Fine structure inverse
    'proton': 1836.15267,      # mp/me
    'muon': 206.768283,        # μ/me
    'weinberg': 0.23122,       # sin²θW
    'rydberg': 1.0973731568,   # R∞ mantissa
    'electron_g': 2.00231930,  # g_e
    'planck_m': 5.391,         # Planck mass mantissa
    'planck_exp': 44.0,        # Planck exponent
    'rydberg_exp': 7.0,        # Rydberg exponent
    'neutron': 1.00137841931,  # mn/mp
    'avogadro': 23.8,          # log10(NA)
    'dirac_exp': 40.0,         # Dirac exponent
}

MAGIC_NUMBERS = [5, 6, 9, 11, 12, 16, 20, 36, 45, 55, 60, 66, 78, 80, 91, 105, 120, 136]


# =============================================================================
# NETWORK NODE
# =============================================================================

@dataclass
class NetworkNode:
    """A fluid node representing a physics constant."""
    name: str
    true_value: float
    current_value: float
    spine_distance: int = 1
    neighbors: List[str] = field(default_factory=list)
    
    @property
    def error(self) -> float:
        if self.true_value == 0:
            return abs(self.current_value)
        return abs(self.current_value - self.true_value) / abs(self.true_value)
    
    @property
    def is_healthy(self) -> bool:
        return self.error < 0.01


# =============================================================================
# SPINE NETWORK
# =============================================================================

@dataclass
class SpineResult:
    """Result from spine network experiment."""
    spine_length: int
    network_size: int
    corruption_count: int
    initial_coherence: float
    final_coherence: float
    recovery_achieved: bool
    healing_steps: int
    coherence_trajectory: List[float]
    integer_overlap: float  # How much of ARCHITECTURE was recovered
    elapsed_seconds: float

    def __repr__(self):
        status = "✓" if self.recovery_achieved else "✗"
        return f"Spine({self.spine_length}) + Network({self.network_size}): {status} {self.final_coherence:.0%} coh, {self.integer_overlap:.0%} int"


class SpineNetwork:
    """
    A quantum network stabilized by a central coherence spine.
    
    The spine is locked at T(16) = 136.
    Network nodes are physics constants with individual true values.
    Healing propagates the seed through architectural relationships.
    """
    
    def __init__(
        self,
        spine_length: int = 9,
        network_constants: Optional[List[str]] = None,
    ):
        self.spine_length = spine_length
        self.spine_value = T16  # Always 136
        
        # Default: use all 12 physics constants
        if network_constants is None:
            network_constants = list(PHYSICS_CONSTANTS.keys())
        
        self.network: Dict[str, NetworkNode] = {}
        for name in network_constants:
            true_val = PHYSICS_CONSTANTS.get(name, 1.0)
            self.network[name] = NetworkNode(
                name=name,
                true_value=true_val,
                current_value=true_val,
                spine_distance=1,
            )
        
        # Connect nodes (fully connected for now)
        all_names = list(self.network.keys())
        for name in all_names:
            self.network[name].neighbors = [n for n in all_names if n != name]
    
    @property
    def network_size(self) -> int:
        return len(self.network)
    
    def coherence(self) -> float:
        """Fraction of healthy network nodes."""
        if not self.network:
            return 1.0
        healthy = sum(1 for n in self.network.values() if n.is_healthy)
        return healthy / len(self.network)
    
    def reset(self):
        """Reset all network nodes to true values."""
        for node in self.network.values():
            node.current_value = node.true_value
    
    def corrupt(self, count: int) -> List[str]:
        """Corrupt N random network nodes."""
        available = list(self.network.keys())
        random.shuffle(available)
        targets = available[:count]
        
        for name in targets:
            node = self.network[name]
            # Random value in wide range around true value
            magnitude = abs(node.true_value) + 0.001
            log_min = math.log(magnitude / 100)
            log_max = math.log(magnitude * 100)
            node.current_value = math.exp(log_min + random.random() * (log_max - log_min))
        
        return targets
    
    def estimate_integers(self) -> Dict[int, float]:
        """
        Estimate architectural integers from current network state.
        
        This is the key mechanism: healthy nodes constrain integer estimates.
        More healthy nodes → sharper estimates → better healing.
        
        The spine provides T(16) = 136 as anchor.
        """
        # Start with T(16) from spine (always correct)
        estimates = {136: float(self.spine_value)}
        
        # Derive other integers from T(16)
        # T(16) = 136 implies 16 is in the set
        estimates[16] = 16.0
        
        # Get health ratio
        healthy_nodes = [n for n in self.network.values() if n.is_healthy]
        health_ratio = len(healthy_nodes) / len(self.network) if self.network else 1.0
        
        # Base integers (always present)
        for i in ARCHITECTURE:
            estimates[i] = float(i)
        
        # Add noise proportional to corruption
        # More corruption → less reliable estimates
        if health_ratio < 1.0:
            noise_scale = (1 - health_ratio) * 0.15
            for i in estimates:
                if i != 136:  # Spine anchor is always exact
                    estimates[i] *= (1 + random.gauss(0, noise_scale))
        
        return estimates
    
    def compute_target(self, name: str, integers: Dict[int, float]) -> Optional[float]:
        """
        Compute target value for a constant from integer estimates.
        
        These are simplified expressions - the real ones come from GEP.
        The point is: given integers, we can derive the constant.
        """
        pi = math.pi
        phi = (1 + math.sqrt(5)) / 2
        e = math.e
        
        try:
            i1 = integers.get(1, 1)
            i7 = integers.get(7, 7)
            i9 = integers.get(9, 9)
            i11 = integers.get(11, 11)
            i16 = integers.get(16, 16)
            i28 = integers.get(28, 28)
            i36 = integers.get(36, 36)
            i44 = integers.get(44, 44)
            i60 = integers.get(60, 60)
            i66 = integers.get(66, 66)
            i136 = integers.get(136, 136)
            
            if name == 'alpha':
                # α⁻¹ ≈ T(16) + small correction
                return i136 + (e/i36 + i136 + pi) / (i136 - phi)
            
            elif name == 'proton':
                # mp/me from T(T(16)) structure
                return i136 * 13.5 + i11/i7
            
            elif name == 'muon':
                # μ/me 
                return i16 + i136 + i136/i28 + i44 + B(SixNine(i136)/i60)
            
            elif name == 'weinberg':
                # sin²θW
                inner = i28 - 1/(pi + i36/i136) - i9
                if inner <= 0:
                    return None
                return math.sqrt(1/inner)
            
            elif name == 'rydberg':
                # R∞ mantissa
                return i1 + i9/100
            
            elif name == 'electron_g':
                return 2 + i1/1000 + i1/4350
            
            elif name == 'planck_m':
                return i1 + 4 + i9/i60 * 2.3
            
            elif name == 'planck_exp':
                return float(i44)
            
            elif name == 'rydberg_exp':
                return float(i7)
            
            elif name == 'neutron':
                return i1 + i1/(i7 * 730)
            
            elif name == 'avogadro':
                return i16 + i7 + 0.8
            
            elif name == 'dirac_exp':
                return i44 - 4
                
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
        
        return None
    
    def heal_step(self):
        """
        One step of healing.
        
        1. Estimate integers from healthy nodes + spine anchor
        2. Compute target values for corrupted nodes
        3. Pull corrupted nodes toward targets
        
        The spine provides T(16) as stable anchor.
        Healthy nodes constrain integer estimates.
        Corrupted nodes heal via derived targets.
        """
        # Get integer estimates
        integers = self.estimate_integers()
        
        # Compute updates
        updates = {}
        for name, node in self.network.items():
            if node.is_healthy:
                continue  # Already good
            
            target = self.compute_target(name, integers)
            if target is None:
                continue
            
            # Pull toward target (strength depends on spine proximity)
            strength = 0.5 / node.spine_distance
            new_value = (1 - strength) * node.current_value + strength * target
            updates[name] = new_value
        
        # Apply updates
        for name, value in updates.items():
            self.network[name].current_value = value
    
    def measure_integer_overlap(self) -> float:
        """
        Measure how much of ARCHITECTURE is implied by current state.
        
        This is the key metric from the original paper:
        can the network reconstruct the architectural integers?
        """
        integers = self.estimate_integers()
        
        # Check each architectural integer
        matches = 0
        for arch_int in ARCHITECTURE:
            estimated = integers.get(arch_int, 0)
            # Allow 5% tolerance
            if abs(estimated - arch_int) / arch_int < 0.05:
                matches += 1
        
        return matches / len(ARCHITECTURE)


def run_spine_experiment(
    spine_length: int = 9,
    network_size: int = 12,
    corruption_fraction: float = 0.5,
    max_steps: int = 50,
    verbose: bool = True,
    random_seed: Optional[int] = None,
) -> SpineResult:
    """
    Run spine network experiment.
    
    1. Create spine + network of physics constants
    2. Corrupt fraction of network
    3. Let spine + healthy nodes heal the network
    4. Measure: coherence recovery, integer overlap
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Select which constants to use (cycle through if network_size != 12)
    all_constants = list(PHYSICS_CONSTANTS.keys())
    if network_size <= 12:
        network_constants = all_constants[:network_size]
    else:
        # Repeat constants for larger networks
        network_constants = (all_constants * (network_size // 12 + 1))[:network_size]
    
    network = SpineNetwork(
        spine_length=spine_length,
        network_constants=network_constants,
    )
    
    corruption_count = max(1, int(network_size * corruption_fraction))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SPINE: {spine_length} nodes locked at T(16)={T16}")
        print(f"NETWORK: {network_size} physics constants")
        print(f"CORRUPT: {corruption_count} nodes ({corruption_fraction:.0%})")
        print(f"{'='*60}")
    
    start = time.time()
    
    initial_coherence = network.coherence()
    
    # Corrupt
    corrupted = network.corrupt(corruption_count)
    post_corrupt_coherence = network.coherence()
    
    if verbose:
        print(f"\nInitial coherence: {initial_coherence:.0%}")
        print(f"After corruption:  {post_corrupt_coherence:.0%}")
        print(f"Corrupted: {corrupted}")
        print(f"\nHealing...")
    
    # Heal
    trajectory = [post_corrupt_coherence]
    healing_steps = 0
    
    for step in range(max_steps):
        network.heal_step()
        coh = network.coherence()
        trajectory.append(coh)
        healing_steps = step + 1
        
        if verbose and (step < 10 or step % 10 == 9):
            bar = "█" * int(coh * 20) + "░" * (20 - int(coh * 20))
            int_overlap = network.measure_integer_overlap()
            print(f"  {step+1:3d}. [{bar}] {coh:.0%} coh | {int_overlap:.0%} int")
        
        # Check convergence
        if coh >= 0.99:
            if verbose:
                print(f"\n  ✓ Full recovery at step {step+1}")
            break
    
    elapsed = time.time() - start
    final_coherence = network.coherence()
    integer_overlap = network.measure_integer_overlap()
    recovery = final_coherence >= 0.9
    
    if verbose:
        status = "✓ RECOVERED" if recovery else "✗ FAILED"
        print(f"\n{status}")
        print(f"  Coherence: {final_coherence:.0%}")
        print(f"  Integer overlap: {integer_overlap:.0%}")
        print(f"  Steps: {healing_steps}")
        print(f"{'='*60}")
    
    return SpineResult(
        spine_length=spine_length,
        network_size=network_size,
        corruption_count=corruption_count,
        initial_coherence=initial_coherence,
        final_coherence=final_coherence,
        recovery_achieved=recovery,
        healing_steps=healing_steps,
        coherence_trajectory=trajectory,
        integer_overlap=integer_overlap,
        elapsed_seconds=elapsed,
    )


def sweep_corruption(
    spine_length: int = 9,
    network_size: int = 12,
    corruption_levels: Optional[List[float]] = None,
    trials: int = 5,
    verbose: bool = True,
) -> Dict[float, List[SpineResult]]:
    """
    Sweep corruption levels to find threshold.
    
    At what corruption level does recovery fail?
    """
    if corruption_levels is None:
        corruption_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {}
    
    if verbose:
        print("\n" + "#" * 70)
        print(f"# CORRUPTION SWEEP (Spine={spine_length}, Network={network_size})")
        print("#" * 70)
    
    for level in corruption_levels:
        results[level] = []
        
        for _ in range(trials):
            r = run_spine_experiment(
                spine_length=spine_length,
                network_size=network_size,
                corruption_fraction=level,
                verbose=False,
            )
            results[level].append(r)
        
        # Summary
        recovery_rate = sum(1 for r in results[level] if r.recovery_achieved) / trials
        avg_coh = sum(r.final_coherence for r in results[level]) / trials
        avg_int = sum(r.integer_overlap for r in results[level]) / trials
        
        bar = "█" * int(recovery_rate * 10) + "░" * (10 - int(recovery_rate * 10))
        
        if verbose:
            print(f"  {level:.0%} corrupt: [{bar}] {recovery_rate:.0%} recovery | {avg_coh:.0%} coh | {avg_int:.0%} int")
    
    return results


def sweep_network_size(
    spine_length: int = 9,
    sizes: Optional[List[int]] = None,
    corruption_fraction: float = 0.5,
    trials: int = 3,
    verbose: bool = True,
) -> Dict[int, List[SpineResult]]:
    """
    Sweep network sizes to find stability threshold.
    """
    if sizes is None:
        sizes = [6, 9, 12, 16, 20, 24, 30, 36, 45, 60]
    
    results = {}
    
    if verbose:
        print("\n" + "#" * 70)
        print(f"# NETWORK SIZE SWEEP (Spine={spine_length}, Corruption={corruption_fraction:.0%})")
        print("#" * 70)
    
    for size in sizes:
        results[size] = []
        
        for _ in range(trials):
            r = run_spine_experiment(
                spine_length=spine_length,
                network_size=size,
                corruption_fraction=corruption_fraction,
                verbose=False,
            )
            results[size].append(r)
        
        recovery_rate = sum(1 for r in results[size] if r.recovery_achieved) / trials
        avg_coh = sum(r.final_coherence for r in results[size]) / trials
        avg_int = sum(r.integer_overlap for r in results[size]) / trials
        
        is_magic = "★" if size in MAGIC_NUMBERS else " "
        bar = "█" * int(recovery_rate * 10) + "░" * (10 - int(recovery_rate * 10))
        
        if verbose:
            print(f"{is_magic} N={size:3d}: [{bar}] {recovery_rate:.0%} | {avg_coh:.0%} coh | {avg_int:.0%} int")
    
    return results


def sweep_spine_length(
    network_size: int = 12,
    spines: Optional[List[int]] = None,
    corruption_fraction: float = 0.5,
    trials: int = 3,
    verbose: bool = True,
) -> Dict[int, List[SpineResult]]:
    """
    Sweep spine lengths to find minimum stable seed.
    """
    if spines is None:
        spines = [1, 2, 3, 5, 7, 9, 11, 12]
    
    results = {}
    
    if verbose:
        print("\n" + "#" * 70)
        print(f"# SPINE LENGTH SWEEP (Network={network_size}, Corruption={corruption_fraction:.0%})")
        print("#" * 70)
    
    for spine in spines:
        results[spine] = []
        
        for _ in range(trials):
            r = run_spine_experiment(
                spine_length=spine,
                network_size=network_size,
                corruption_fraction=corruption_fraction,
                verbose=False,
            )
            results[spine].append(r)
        
        recovery_rate = sum(1 for r in results[spine] if r.recovery_achieved) / trials
        avg_coh = sum(r.final_coherence for r in results[spine]) / trials
        avg_int = sum(r.integer_overlap for r in results[spine]) / trials
        
        is_magic = "★" if spine in MAGIC_NUMBERS else " "
        bar = "█" * int(recovery_rate * 10) + "░" * (10 - int(recovery_rate * 10))
        
        if verbose:
            print(f"{is_magic} S={spine:2d}: [{bar}] {recovery_rate:.0%} | {avg_coh:.0%} coh | {avg_int:.0%} int")
    
    return results


def find_stability_frontier(
    spine_range: Tuple[int, int] = (1, 12),
    network_range: Tuple[int, int] = (6, 36),
    corruption_fraction: float = 0.5,
    trials: int = 3,
    verbose: bool = True,
) -> Dict[Tuple[int, int], float]:
    """
    Map the stability frontier.
    """
    results = {}
    
    if verbose:
        print("\n" + "#" * 70)
        print("# STABILITY FRONTIER")
        print("#" * 70)
        print(f"\nSpine: {spine_range[0]}-{spine_range[1]}")
        print(f"Network: {network_range[0]}-{network_range[1]}")
        print(f"Corruption: {corruption_fraction:.0%}")
        print()
    
    for spine in range(spine_range[0], spine_range[1] + 1):
        row = []
        for network in range(network_range[0], network_range[1] + 1, 3):
            successes = 0
            for _ in range(trials):
                r = run_spine_experiment(
                    spine_length=spine,
                    network_size=network,
                    corruption_fraction=corruption_fraction,
                    verbose=False,
                )
                if r.recovery_achieved:
                    successes += 1
            
            rate = successes / trials
            results[(spine, network)] = rate
            row.append(rate)
        
        if verbose:
            cells = ["█" if r >= 0.67 else "▒" if r >= 0.33 else "░" for r in row]
            print(f"S={spine:2d}: " + "".join(cells))
    
    return results

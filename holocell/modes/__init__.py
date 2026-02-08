"""
HoloCell Modes — The Ten Modes of Sight

Mode 1: Fixed Focus    — Standard GEP with fixed terminals
Mode 2: Coherent Zoom  — Co-evolve integer set itself
Mode 3: Seth Mode      — Dual set partition (archive/transmitted)
Mode 4: Moon Pools     — Multi-pool eigenvalue triangulation
Mode 5: Coherence Test — N-node corruption sweep for fault tolerance
Mode 6: Weave          — Incremental corruption & restoration dynamics
Mode 7: Maintained     — Self-healing network with memory (flawed model)
Mode 8: Phalanx        — Self-healing with dynamic frozen flanks
Mode 9: Spine          — Merkabah quantum network (central axis coherence)
Mode 10: Vogel         — Crystal Harmonic Resonance (buckyball + Vogel spine)

All modes use GEPEvolver for the genetic engine.
"""

from .fixed_focus import evolve_constant, test_seeds, EvolutionResult
from .coherent_zoom import evolve_coherent, CoherentResult
from .seth_mode import evolve_seth, SethResult
from .moon_pools import run_moon_pools, MoonPoolResult
from .coherence_test import run_coherence_sweep, CoherenceSweepResult
from .weave import weave, compare_strategies, WeaveResult, WeaveStep, SelectionStrategy
from .maintained import (
    run_maintained_network,
    sweep_maintained_network,
    MaintainedNetwork,
    MaintainedNetworkResult,
    HealingStep,
)
from .phalanx import (
    run_phalanx,
    sweep_phalanx,
    PhalanxNetwork,
    PhalanxResult,
    PhalanxStep,
    NodeState,
    NodeRole,
)
from .spine import (
    run_spine_experiment,
    sweep_network_size,
    sweep_spine_length,
    find_stability_frontier,
    SpineNetwork,
    SpineResult,
)
from .vogel import (
    run_vogel_experiment,
    sweep_corruption as vogel_sweep_corruption,
    compare_geometries,
    VogelNetwork,
    VogelResult,
)

__all__ = [
    # Mode 1: Fixed Focus
    "evolve_constant",
    "test_seeds",
    "EvolutionResult",
    # Mode 2: Coherent Zoom
    "evolve_coherent",
    "CoherentResult",
    # Mode 3: Seth Mode
    "evolve_seth",
    "SethResult",
    # Mode 4: Moon Pools
    "run_moon_pools",
    "MoonPoolResult",
    # Mode 5: Coherence Test
    "run_coherence_sweep",
    "CoherenceSweepResult",
    # Mode 6: Weave
    "weave",
    "compare_strategies",
    "WeaveResult",
    "WeaveStep",
    "SelectionStrategy",
    # Mode 7: Maintained
    "run_maintained_network",
    "sweep_maintained_network",
    "MaintainedNetwork",
    "MaintainedNetworkResult",
    "HealingStep",
    # Mode 8: Phalanx
    "run_phalanx",
    "sweep_phalanx",
    "PhalanxNetwork",
    "PhalanxResult",
    "PhalanxStep",
    "NodeState",
    "NodeRole",
    # Mode 9: Spine
    "run_spine_experiment",
    "sweep_network_size",
    "sweep_spine_length",
    "find_stability_frontier",
    "SpineNetwork",
    "SpineResult",
    # Mode 10: Vogel
    "run_vogel_experiment",
    "vogel_sweep_corruption",
    "compare_geometries",
    "VogelNetwork",
    "VogelResult",
]

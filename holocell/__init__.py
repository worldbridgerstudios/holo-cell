"""
HoloCell - Architectural discovery engine for physics constants and self-healing networks

T(16) = 136 as the eigenvalue of fundamental physics.

USAGE:
    from holocell import T, B, S, CRYSTAL, verify_all
    
    # The seed
    print(T(16))  # 136
    
    # Verify all constants
    results = verify_all()
    
    # Access specific constant
    proton = CRYSTAL["mp/me"]
    print(f"Error: {proton.error_percent:.2e}%")

SELF-HEALING NETWORKS:
    from holocell.networks import (
        test_egyptian_candidates,
        healing_score,
        icosahedron_edges,
    )
    
    # Test the sequence: 12, 36, 60, 80, 120, 136, 240, 408
    results = test_egyptian_candidates()
    for n, analysis in sorted(results.items()):
        print(f"{n}: self-healing={analysis.is_self_healing}")

SIX MODES OF SIGHT:
    from holocell.modes import (
        evolve_constant,      # Mode 1: Fixed Focus
        evolve_coherent,      # Mode 2: Coherent Zoom
        evolve_seth,          # Mode 3: Seth Mode
        run_moon_pools,       # Mode 4: Moon Pools
        run_coherence_sweep,  # Mode 5: Coherence Test
        weave,                # Mode 6: Weave
        run_maintained_network,  # Mode 7: Maintained
    )
    
    # Mode 1: Evolve single constant with fixed seed
    result = evolve_constant("alpha")
    
    # Mode 2: Co-evolve integer set itself
    result = evolve_coherent(integer_set_size=6)
    
    # Mode 3: Dual set partition (archive/transmitted)
    result = evolve_seth()
    
    # Mode 4: Multi-pool eigenvalue triangulation
    result = run_moon_pools(num_pools=4)
    
    # Mode 5: N-node corruption sweep (fault tolerance)
    result = run_coherence_sweep(max_corruption=8)
    
    # Mode 6: Incremental weave (healing dynamics)
    result = weave(max_corruption=6, strategy=SelectionStrategy.RANDOM)
    
    # Mode 7: Maintained network (self-healing with memory)
    result = run_maintained_network(corruption_count=6)
    
    # Mode 8: Phalanx (self-healing with dynamic flanks)
    result = run_phalanx(corruption_count=4)
    
    # Mode 9: Spine (merkabah quantum network)
    result = run_spine_experiment(spine_length=9, network_size=36)

CLI:
    holocell verify              # Verify crystallized expressions
    holocell evolve <constant>   # Mode 1: Fixed Focus
    holocell coherent            # Mode 2: Coherent Zoom
    holocell seth                # Mode 3: Seth Mode
    holocell moonpools           # Mode 4: Moon Pools
    holocell sweep               # Mode 5: Coherence Test
    holocell weave               # Mode 6: Weave
    holocell maintain            # Mode 7: Maintained
"""

__version__ = "0.5.0"

from .operators import T, B, S, triangular, bilateral, six_nine
from .constants import CRYSTAL, ARCHITECTURE, verify_all, verify_constant
from .magic import (
    FINE_STRUCTURE_INV,
    PROTON_ELECTRON_MASS,
    MUON_ELECTRON_MASS,
    WEINBERG_ANGLE,
    RYDBERG_MANTISSA,
    find_closest_magic,
    nearest_triangular,
)

# Self-healing networks
from .networks import (
    T as T_network,  # Alias to avoid collision
    inverse_T,
    is_triangular,
    generate_candidates,
    icosahedron_edges,
    dodecahedron_edges,
    regular_polygon_edges,
    healing_score,
    is_connected,
    analyze_network,
    test_egyptian_candidates,
    NetworkCandidate,
    NetworkAnalysis,
    WHEEL,
    SPINE,
    HOURGLASS,
    DIRECTED,
)

# Six Modes of Sight (re-exported for convenience)
from .modes import (
    # Mode 1: Fixed Focus
    evolve_constant,
    test_seeds,
    EvolutionResult,
    # Mode 2: Coherent Zoom
    evolve_coherent,
    CoherentResult,
    # Mode 3: Seth Mode
    evolve_seth,
    SethResult,
    # Mode 4: Moon Pools
    run_moon_pools,
    MoonPoolResult,
    # Mode 5: Coherence Test
    run_coherence_sweep,
    CoherenceSweepResult,
    # Mode 6: Weave
    weave,
    compare_strategies,
    WeaveResult,
    WeaveStep,
    SelectionStrategy,
    # Mode 7: Maintained
    run_maintained_network,
    sweep_maintained_network,
    MaintainedNetwork,
    MaintainedNetworkResult,
    HealingStep,
    # Mode 8: Phalanx
    run_phalanx,
    sweep_phalanx,
    PhalanxNetwork,
    PhalanxResult,
    PhalanxStep,
    NodeState,
    NodeRole,
    # Mode 9: Spine
    run_spine_experiment,
    sweep_network_size,
    sweep_spine_length,
    find_stability_frontier,
    SpineNetwork,
    SpineResult,
)

__all__ = [
    # Version
    "__version__",
    # Operators
    "T", "B", "S",
    "triangular", "bilateral", "six_nine",
    # Constants
    "CRYSTAL", "ARCHITECTURE",
    "verify_all", "verify_constant",
    # Magic numbers
    "FINE_STRUCTURE_INV",
    "PROTON_ELECTRON_MASS",
    "MUON_ELECTRON_MASS",
    "WEINBERG_ANGLE",
    "RYDBERG_MANTISSA",
    "find_closest_magic",
    "nearest_triangular",
    # Networks
    "inverse_T",
    "is_triangular",
    "generate_candidates",
    "icosahedron_edges",
    "dodecahedron_edges",
    "regular_polygon_edges",
    "healing_score",
    "is_connected",
    "analyze_network",
    "test_egyptian_candidates",
    "NetworkCandidate",
    "NetworkAnalysis",
    "WHEEL", "SPINE", "HOURGLASS", "DIRECTED",
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
]

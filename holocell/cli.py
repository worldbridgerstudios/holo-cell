"""
HoloCell CLI — Command-line interface for the Seven Modes of Sight.

Usage:
    holocell verify              # Verify crystallized expressions
    holocell evolve <constant>   # Mode 1: Fixed Focus
    holocell seed-test           # Mode 1: Seed comparison
    holocell coherent            # Mode 2: Coherent Zoom
    holocell seth                # Mode 3: Seth Mode
    holocell moonpools           # Mode 4: Moon Pools
    holocell sweep               # Mode 5: Coherence Test
    holocell weave               # Mode 6: Weave
    holocell maintain            # Mode 7: Maintained
"""

import argparse
import sys


def cmd_verify():
    """Verify all crystallized expressions."""
    from holocell import verify_all, CRYSTAL

    print("\n" + "=" * 60)
    print("HOLOCELL CRYSTAL VERIFICATION")
    print("=" * 60 + "\n")

    results = verify_all()

    for name, passed in results.items():
        c = CRYSTAL[name]
        status = "✓" if passed else "✗"
        print(f"{status} {c.symbol:>10}: {c.error_percent:.2e}% error")

    total_pass = sum(results.values())
    print(f"\n{total_pass}/{len(results)} passed verification\n")

    return 0 if all(results.values()) else 1


def cmd_evolve(args):
    """Mode 1: Fixed Focus - evolve expression for a single constant."""
    from holocell.modes import evolve_constant

    result = evolve_constant(
        args.constant,
        seed_value=args.seed,
        generations=args.generations,
        pop_size=args.population,
        verbose=True,
    )

    return 0 if result.error_percent < 0.01 else 1


def cmd_seed_test(args):
    """Mode 1: Seed testing - compare candidate seeds."""
    from holocell.modes import test_seeds
    from holocell.modes.targets import CANDIDATE_SEEDS

    seeds = CANDIDATE_SEEDS if args.all else [136, 137, 66, 36, 11]

    ranking = test_seeds(
        seeds=seeds,
        generations_per_target=args.generations,
        pop_size=args.population,
        verbose=True,
    )

    return 0


def cmd_coherent(args):
    """Mode 2: Coherent Zoom - co-evolve integer set."""
    from holocell.modes import evolve_coherent

    result = evolve_coherent(
        integer_set_size=args.integers,
        pop_size=args.population,
        generations=args.generations,
        verbose=True,
    )

    return 0 if result.converged else 1


def cmd_seth(args):
    """Mode 3: Seth Mode - dual set partition."""
    from holocell.modes import evolve_seth

    result = evolve_seth(
        pop_size=args.population,
        generations=args.generations,
        verbose=True,
    )

    return 0 if result.converged else 1


def cmd_moonpools(args):
    """Mode 4: Moon Pools - multi-pool triangulation."""
    from holocell.modes import run_moon_pools

    result = run_moon_pools(
        num_pools=args.pools,
        max_runtime_seconds=args.runtime,
        verbose=True,
    )

    return 0


def cmd_sweep(args):
    """Mode 5: Coherence Test - N-node corruption sweep."""
    from holocell.modes import run_coherence_sweep

    result = run_coherence_sweep(
        max_corruption=args.max_corruption,
        trials_per_level=args.trials,
        verbose=True,
    )

    print(f"\nFault tolerance: {result.fault_tolerance_threshold} nodes")
    return 0


def cmd_weave(args):
    """Mode 6: Weave - incremental corruption and restoration."""
    from holocell.modes import weave, compare_strategies, SelectionStrategy

    if args.compare:
        # Run all three strategies
        compare_strategies(
            max_corruption=args.max_corruption,
            generations_per_step=args.generations,
            verbose=True,
        )
    else:
        # Single strategy
        strategy_map = {
            "random": SelectionStrategy.RANDOM,
            "worst": SelectionStrategy.WORST_FIRST,
            "best": SelectionStrategy.BEST_FIRST,
        }
        strategy = strategy_map.get(args.strategy, SelectionStrategy.RANDOM)

        result = weave(
            max_corruption=args.max_corruption,
            strategy=strategy,
            generations_per_step=args.generations,
            verbose=True,
        )

        print(f"\nHysteresis: {result.hysteresis_score:.1%}")

    return 0


def cmd_maintain(args):
    """Mode 7: Maintained - self-healing network with memory."""
    from holocell.modes import run_maintained_network, sweep_maintained_network

    if args.sweep:
        # Sweep all corruption levels
        sweep_maintained_network(
            max_corruption=args.max_corruption,
            trials_per_level=args.trials,
            max_iterations=args.iterations,
            verbose=True,
        )
    else:
        # Single run
        result = run_maintained_network(
            corruption_count=args.corruption,
            max_iterations=args.iterations,
            verbose=True,
        )

        status = "✓ HEALED" if result.full_recovery else "✗ FAILED"
        print(f"\n{status} in {result.healing_time} iterations")

    return 0


def cmd_phalanx(args):
    """Mode 8: Phalanx - self-healing with dynamic frozen flanks."""
    from holocell.modes import run_phalanx, sweep_phalanx

    if args.sweep:
        # Sweep all corruption levels
        sweep_phalanx(
            max_corruption=args.max_corruption,
            trials_per_level=args.trials,
            max_steps=args.steps,
            verbose=True,
        )
    else:
        # Single run
        result = run_phalanx(
            corruption_count=args.corruption,
            max_steps=args.steps,
            freeze_threshold=args.freeze_threshold,
            unfreeze_threshold=args.unfreeze_threshold,
            verbose=True,
        )

        status = "✓ RECOVERED" if result.recovery_achieved else "✗ FAILED"
        print(f"\n{status}")

    return 0


def cmd_spine(args):
    """Mode 9: Spine - merkabah quantum network."""
    from holocell.modes import (
        run_spine_experiment,
        sweep_network_size,
        sweep_spine_length,
        find_stability_frontier,
    )

    if args.frontier:
        # Map stability frontier
        find_stability_frontier(
            spine_range=(1, args.max_spine),
            network_range=(6, args.max_network),
            step=args.step,
            corruption_fraction=args.corruption,
            trials=args.trials,
            verbose=True,
        )
    elif args.sweep_network:
        # Sweep network sizes
        sweep_network_size(
            spine_length=args.spine,
            corruption_fraction=args.corruption,
            trials=args.trials,
            verbose=True,
        )
    elif args.sweep_spine:
        # Sweep spine lengths
        sweep_spine_length(
            network_size=args.network,
            corruption_fraction=args.corruption,
            trials=args.trials,
            verbose=True,
        )
    else:
        # Single run
        result = run_spine_experiment(
            spine_length=args.spine,
            network_size=args.network,
            corruption_fraction=args.corruption,
            verbose=True,
        )

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="HoloCell: T(16) = 136 as the eigenvalue of physics constants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NINE MODES OF SIGHT:

  Mode 1: Fixed Focus    — Standard GEP with fixed terminals
      holocell evolve alpha
      holocell seed-test

  Mode 2: Coherent Zoom  — Co-evolve integer set itself
      holocell coherent

  Mode 3: Seth Mode      — Dual set partition (archive/transmitted)
      holocell seth

  Mode 4: Moon Pools     — Multi-pool eigenvalue triangulation
      holocell moonpools

  Mode 5: Coherence Test — N-node corruption sweep
      holocell sweep

  Mode 6: Weave          — Incremental corruption/restoration dynamics
      holocell weave
      holocell weave --compare

  Mode 7: Maintained     — Self-healing network with memory (flawed)
      holocell maintain
      holocell maintain --sweep

  Mode 8: Phalanx        — Self-healing with dynamic frozen flanks
      holocell phalanx
      holocell phalanx --sweep

  Mode 9: Spine          — Merkabah quantum network (central axis)
      holocell spine
      holocell spine --sweep-network
      holocell spine --frontier

VERIFICATION:
      holocell verify         — Verify crystallized expressions

Constants (for Mode 1):
    alpha     Fine structure constant inverse (α⁻¹)
    proton    Proton-electron mass ratio (mp/me)
    muon      Muon-electron mass ratio (μ/me)
    weinberg  Weinberg angle (sin²θW)
    rydberg   Rydberg constant mantissa (R∞)
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # verify
    subparsers.add_parser("verify", help="Verify crystallized expressions")

    # evolve (Mode 1)
    sub_evolve = subparsers.add_parser("evolve", help="Mode 1: Evolve expression for a constant")
    sub_evolve.add_argument("constant", help="Constant name (alpha, proton, muon, weinberg, rydberg)")
    sub_evolve.add_argument("--seed", type=int, default=136, help="Seed value (default: 136)")
    sub_evolve.add_argument("--generations", type=int, default=1000, help="Max generations")
    sub_evolve.add_argument("--population", type=int, default=300, help="Population size")

    # seed-test (Mode 1)
    sub_seed = subparsers.add_parser("seed-test", help="Mode 1: Test candidate seeds")
    sub_seed.add_argument("--all", action="store_true", help="Test all candidate seeds")
    sub_seed.add_argument("--generations", type=int, default=500, help="Generations per target")
    sub_seed.add_argument("--population", type=int, default=200, help="Population size")

    # coherent (Mode 2)
    sub_coh = subparsers.add_parser("coherent", help="Mode 2: Coherent Zoom")
    sub_coh.add_argument("--integers", type=int, default=6, help="Integer set size (default: 6)")
    sub_coh.add_argument("--generations", type=int, default=2000, help="Max generations")
    sub_coh.add_argument("--population", type=int, default=300, help="Population size")

    # seth (Mode 3)
    sub_seth = subparsers.add_parser("seth", help="Mode 3: Seth Mode (dual partition)")
    sub_seth.add_argument("--generations", type=int, default=3000, help="Max generations")
    sub_seth.add_argument("--population", type=int, default=400, help="Population size")

    # moonpools (Mode 4)
    sub_moon = subparsers.add_parser("moonpools", help="Mode 4: Moon Pools")
    sub_moon.add_argument("--pools", type=int, default=4, help="Number of pools (default: 4)")
    sub_moon.add_argument("--runtime", type=float, default=180, help="Max runtime seconds")

    # sweep (Mode 5)
    sub_sweep = subparsers.add_parser("sweep", help="Mode 5: Coherence Test sweep")
    sub_sweep.add_argument("--max-corruption", type=int, default=8, help="Max corruption level")
    sub_sweep.add_argument("--trials", type=int, default=3, help="Trials per level")

    # weave (Mode 6)
    sub_weave = subparsers.add_parser("weave", help="Mode 6: Weave (incremental dynamics)")
    sub_weave.add_argument("--max-corruption", type=int, default=6, help="Max corruption level")
    sub_weave.add_argument("--generations", type=int, default=100, help="Generations per step")
    sub_weave.add_argument("--strategy", choices=["random", "worst", "best"], default="random",
                          help="Node selection strategy")
    sub_weave.add_argument("--compare", action="store_true", help="Compare all three strategies")

    # maintain (Mode 7)
    sub_maintain = subparsers.add_parser("maintain", help="Mode 7: Maintained (self-healing with memory)")
    sub_maintain.add_argument("--corruption", type=int, default=6, help="Number of nodes to corrupt")
    sub_maintain.add_argument("--iterations", type=int, default=20, help="Max healing iterations")
    sub_maintain.add_argument("--sweep", action="store_true", help="Sweep all corruption levels")
    sub_maintain.add_argument("--max-corruption", type=int, default=11, help="Max corruption for sweep")
    sub_maintain.add_argument("--trials", type=int, default=3, help="Trials per level for sweep")

    # phalanx (Mode 8)
    sub_phalanx = subparsers.add_parser("phalanx", help="Mode 8: Phalanx (dynamic frozen flanks)")
    sub_phalanx.add_argument("--corruption", type=int, default=4, help="Number of nodes to corrupt")
    sub_phalanx.add_argument("--steps", type=int, default=50, help="Max simulation steps")
    sub_phalanx.add_argument("--freeze-threshold", type=float, default=0.7, help="Coherence to trigger freeze")
    sub_phalanx.add_argument("--unfreeze-threshold", type=float, default=0.9, help="Coherence to allow unfreeze")
    sub_phalanx.add_argument("--sweep", action="store_true", help="Sweep all corruption levels")
    sub_phalanx.add_argument("--max-corruption", type=int, default=6, help="Max corruption for sweep")
    sub_phalanx.add_argument("--trials", type=int, default=3, help="Trials per level for sweep")

    # spine (Mode 9)
    sub_spine = subparsers.add_parser("spine", help="Mode 9: Spine (merkabah quantum network)")
    sub_spine.add_argument("--spine", type=int, default=9, help="Spine length")
    sub_spine.add_argument("--network", type=int, default=36, help="Network size")
    sub_spine.add_argument("--corruption", type=float, default=0.5, help="Corruption fraction")
    sub_spine.add_argument("--sweep-network", action="store_true", help="Sweep network sizes")
    sub_spine.add_argument("--sweep-spine", action="store_true", help="Sweep spine lengths")
    sub_spine.add_argument("--frontier", action="store_true", help="Map stability frontier")
    sub_spine.add_argument("--max-spine", type=int, default=12, help="Max spine for frontier")
    sub_spine.add_argument("--max-network", type=int, default=60, help="Max network for frontier")
    sub_spine.add_argument("--step", type=int, default=3, help="Step size for frontier")
    sub_spine.add_argument("--trials", type=int, default=3, help="Trials per configuration")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "verify": cmd_verify,
        "evolve": lambda: cmd_evolve(args),
        "seed-test": lambda: cmd_seed_test(args),
        "coherent": lambda: cmd_coherent(args),
        "seth": lambda: cmd_seth(args),
        "moonpools": lambda: cmd_moonpools(args),
        "sweep": lambda: cmd_sweep(args),
        "weave": lambda: cmd_weave(args),
        "maintain": lambda: cmd_maintain(args),
        "phalanx": lambda: cmd_phalanx(args),
        "spine": lambda: cmd_spine(args),
    }

    handler = commands.get(args.command)
    if handler:
        return handler() if callable(handler) and handler.__name__ == '<lambda>' else handler()

    return 0


if __name__ == "__main__":
    sys.exit(main())

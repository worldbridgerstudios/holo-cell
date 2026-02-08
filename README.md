# HoloCell

**T(16) = 136 as the eigenvalue of fundamental physics constants.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18183435.svg)](https://doi.org/10.5281/zenodo.18183435)
[![PyPI version](https://badge.fury.io/py/holocell.svg)](https://pypi.org/project/holocell/)

## Installation

```bash
pip install holocell
```

Requires [GEPEvolver](https://github.com/worldbridgerstudios/GEPEvolver) (installed automatically).

## Quick Start

```python
from holocell import T, B, S, CRYSTAL, verify_all

# The seed
print(T(16))  # 136

# Verify all 5 constants
results = verify_all()
# {'mp/me': True, 'R∞': True, 'α⁻¹': True, 'μ/me': True, 'sin²θW': True}

# Access a specific crystal constant
proton = CRYSTAL["mp/me"]
print(f"Computed:  {proton.computed}")
print(f"Measured:  {proton.measured}")
print(f"Error:     {proton.error_percent:.2e}%")
```

## The Discovery

Five fundamental physics constants emerge from a single seed: **T(16) = 136**.

| Constant | Expression | Error |
|----------|------------|-------|
| mp/me | T(136) × 3 × (9/2) + (11 - 1/T(16))/72 | **1.21×10⁻⁷%** |
| R∞ | B(T(11) × (√(T(16) + e) + 1/36 + 666)⁻¹) | 1.02×10⁻⁵% |
| α⁻¹ | T(16) + (((e/36 + T(16)) + π) / (T(16) - φ)) | 1.35×10⁻⁵% |
| μ/me | (16 + T(16) + T(16)/28 + 44) + B(S(T(16))/60) | 1.40×10⁻⁵% |
| sin²θW | √((28 - (π + 36/T(16))⁻¹ - 9)⁻¹) | 4.67×10⁻⁴% |

## Six Modes of Sight

HoloCell provides six evolutionary modes for discovering and validating architectural structure:

```python
from holocell.modes import (
    evolve_constant,      # Mode 1: Fixed Focus
    evolve_coherent,      # Mode 2: Coherent Zoom
    evolve_seth,          # Mode 3: Seth Mode
    run_moon_pools,       # Mode 4: Moon Pools
    run_coherence_sweep,  # Mode 5: Coherence Test
    weave,                # Mode 6: Weave
)
```

### Mode 1: Fixed Focus
Standard GEP evolution with fixed terminals.
```python
result = evolve_constant("alpha", seed_value=136)
```

### Mode 2: Coherent Zoom
Co-evolve the integer set itself across all constants simultaneously.
```python
result = evolve_coherent(integer_set_size=6)
print(f"Discovered integers: {result.discovered_integers}")
```

### Mode 3: Seth Mode
Dual set partition — discover which constants need the full archive vs filtered subset.
```python
result = evolve_seth()
print(f"Archive: {result.archive}")
print(f"Transmitted: {result.transmitted}")
```

### Mode 4: Moon Pools
Multi-pool eigenvalue triangulation — find crossing bands.
```python
result = run_moon_pools(num_pools=4, max_runtime_seconds=180)
```

### Mode 5: Coherence Test
N-node corruption sweep — measure fault tolerance threshold.
```python
result = run_coherence_sweep(max_corruption=8)
print(f"Fault tolerance: {result.fault_tolerance_threshold} nodes")
```

### Mode 6: Weave

**Incremental corruption and restoration — reveal healing dynamics.**

Instead of batch corruption (Mode 5), weave between states one node at a time:

```
BATCH (Mode 5):              WEAVE (Mode 6):

corrupt 7 → evolve 1000      corrupt 1 → evolve 50 → measure
           → measure         corrupt 1 → evolve 50 → measure
                             ...
corrupt 7 → evolve 1000      restore 1 → evolve 50 → measure
           → measure         restore 1 → evolve 50 → measure
                             ...
```

**What this reveals:**

| Metric | Mode 5 (Batch) | Mode 6 (Weave) |
|--------|----------------|----------------|
| Threshold | Yes | Yes |
| Healing dynamics | No | **Per-step trajectory** |
| Hysteresis | No | **Does path matter?** |
| Phase transitions | Coarse | **Sharp or gradual?** |
| Selection effects | No | **Which nodes matter?** |

**Three selection strategies:**

| Strategy | Corrupt Order | Restore Order |
|----------|---------------|---------------|
| `RANDOM` | Any order | Any order |
| `WORST_FIRST` | Highest error first | Lowest error first |
| `BEST_FIRST` | Lowest error first | Highest error first |

**Usage:**

```python
from holocell import weave, compare_strategies, SelectionStrategy

# Single strategy
result = weave(
    max_corruption=6,
    strategy=SelectionStrategy.RANDOM,
    generations_per_step=100,
)
print(f"Hysteresis: {result.hysteresis_score:.1%}")
print(f"Recovery: {'✓' if result.recovery_complete else '✗'}")

# Compare all strategies
results = compare_strategies(max_corruption=6)
for strategy, result in results.items():
    print(f"{strategy.value}: hysteresis={result.hysteresis_score:.1%}")
```

**Key insight:** If the manifold is a true basin of attraction, the restore trajectory mirrors the degrade trajectory (hysteresis ≈ 0%). If there's memory of damage, hysteresis > 0%.

## Self-Healing Networks

**The optimal seed geometry is the octahedron: 6 nodes forming 3 bilateral pairs.**

This minimal Platonic solid outperforms all tested alternatives including the buckyball (60 nodes) and vortex engine (144 nodes). The center must remain empty — adding a hub node degrades performance.

| Seed | Frozen | Avg Rate | Steps to 90% |
|------|--------|----------|--------------|
| **octahedron** | **6** | **0.0259** | **24.7** |
| tetrahedron | 4 | 0.0248 | 39.7 |
| buckyball | 60 | 0.0242 | 27.3 |
| vortex_engine | 144 | 0.0242 | 31.0 |

**The HoloCell Geometry:**
- Octahedron = outer shell (6 vertices, 3 bilateral pairs on Trinition axes)
- T(16) = 136 = eigenvalue at center (not a node — a frequency)
- 408 = T(16) × 3 = scale invariance marker

The eigenvalue isn't a physical node. It's what the structure *resonates at*. The octahedron is the antenna; T(16) is the frequency.

```python
from holocell.networks import test_egyptian_candidates

results = test_egyptian_candidates()
for n, analysis in sorted(results.items()):
    print(f"N={n}: self-healing={analysis.is_self_healing}")
```

## Operators

Three architectural operators from Egyptian cosmological mathematics:

```python
from holocell import T, B, S

T(16)    # 136 — Triangular number: n(n+1)/2
B(T(16)) # 137 — Bilateral covenant: x + 1
S(9)     # 19.5 — Six-nine harmonic: x×6/9 + x×9/6
```

## CLI

```bash
# Verify crystallized expressions
holocell verify

# Mode 1: Fixed Focus
holocell evolve alpha
holocell seed-test

# Mode 2: Coherent Zoom
holocell coherent

# Mode 3: Seth Mode
holocell seth

# Mode 4: Moon Pools
holocell moonpools

# Mode 5: Coherence Test
holocell sweep

# Mode 6: Weave
holocell weave                    # Random strategy
holocell weave --strategy worst   # Worst-first
holocell weave --strategy best    # Best-first
holocell weave --compare          # Compare all strategies
```

## License

CC0 1.0 Universal — Public Domain

## Citation

```bibtex
@software{brown2025holocell,
  title={HoloCell: T(16) = 136 as the Eigenvalue of Fundamental Physics Constants},
  author={Brown, Nicholas David},
  year={2025},
  doi={10.5281/zenodo.18183435}
}
```

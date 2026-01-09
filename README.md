# HoloCell

**T(16) = 136 as the eigenvalue of fundamental physics constants.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18183435.svg)](https://doi.org/10.5281/zenodo.18183435)
[![PyPI version](https://badge.fury.io/py/holocell.svg)](https://pypi.org/project/holocell/)

## Installation

```bash
pip install holocell
```

## Quick Start

```python
from holocell import T, B, S, CRYSTAL, verify_all

# The seed
print(T(16))  # 136

# Verify all 5 constants
results = verify_all()
print(results)
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

## Self-Healing Networks

HoloCell also provides tools for analyzing self-healing network topologies derived from Platonic solids and Egyptian phonemic architecture.

**The Sequence:** 12 → 36 → 60 → 80 → 120 → 136 → 240 → 408

| N | Structure | κ | Survives |
|---|-----------|---|----------|
| 12 | Icosahedron | 5 | 4 nodes |
| 36 | T(8) | 6 | 5 nodes |
| 60 | Buckyball | 6 | 5 nodes |
| 80 | Wheel × Hourglass | 6 | 5 nodes |
| 120 | 600-cell / T(15) | 6 | 5 nodes |
| 136 | T(16) | 6 | 5 nodes |
| 240 | Directed pairs | 6 | 5 nodes |
| 408 | Full grammar | 6 | 5 nodes |

```python
from holocell.networks import test_egyptian_candidates, healing_score

# Test all 8 candidates
results = test_egyptian_candidates()
for n, analysis in sorted(results.items()):
    print(f"N={n}: self-healing={analysis.is_self_healing}")
```

📖 **[Full documentation: docs/self-healing-networks.md](docs/self-healing-networks.md)**

## Operators

Three architectural operators from Egyptian cosmological mathematics:

```python
from holocell import T, B, S

# T(n) - Triangular number: n(n+1)/2
T(16)  # 136 - THE SEED

# B(x) - Bilateral covenant: x + 1
B(T(16))  # 137 ≈ α⁻¹

# S(x) - Six-nine harmonic: x×6/9 + x×9/6
S(9)  # 19.5 (the breath)
```

## Methodology Replication

The full discovery process is reproducible via the `evolve` subpackage:

```python
from holocell.evolve import evolve_constant, test_seeds, replicate_methodology

# Stage 1: Evolve expression for fine structure constant
result = evolve_constant("alpha")

# Stage 2: Test candidate seeds to confirm T(16)=136 is optimal
ranking = test_seeds()

# Stage 3: Full methodology replication
results = replicate_methodology()
```

## CLI

```bash
# Verify crystallized expressions
holocell verify

# Evolve expression for a single constant
holocell evolve alpha
holocell evolve proton --generations 2000

# Test candidate seeds
holocell seed-test

# Full methodology replication
holocell replicate
```

## Architecture

```python
from holocell import ARCHITECTURE, WHEEL, SPINE, HOURGLASS

ARCHITECTURE  # [1, 7, 9, 11, 16, 28, 36, 44, 60, 66, 666]
WHEEL         # 16 (phonemes)
SPINE         # 3 (axes)
HOURGLASS     # 5 (positions)
```

## Documentation

- **[Self-Healing Networks](docs/self-healing-networks.md)** — Network topology analysis
- **[Zenodo Record](https://doi.org/10.5281/zenodo.18183435)** — Academic paper and citation

## Citation

```bibtex
@article{brown2025holocell,
  title={HoloCell: The Reality Crystal — T(16) = 136 as the Eigenvalue of Fundamental Physics Constants},
  author={Brown, Nicholas David},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.18183435}
}
```

## License

MIT

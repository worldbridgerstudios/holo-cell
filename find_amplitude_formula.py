#!/usr/bin/env python3
"""
Find formula for optimal amplitude using GEPEvolver

Target: ~0.30 rad (optimal pulse amplitude from rhythm test)
Constants: T(16)=136, φ, π, e, vertex counts 1, 3, 12, 16
"""

import sys
sys.path.insert(0, '/Users/nick/Projects/GEPEvolver')

from gepevolver import GlyphSet, evolve_with_glyphs
import numpy as np

# =============================================================================
# TARGETS — Multiple optimal values from test
# =============================================================================

TARGETS = [0.20, 0.30, 0.50]  # All performed well

# =============================================================================
# GLYPH SET — The magic numbers
# =============================================================================

glyphs = GlyphSet("entanglement_engine")

# Triangular numbers
glyphs.add('a', 'T₁₆', 136, formula='T(16)')
glyphs.add('b', 'T₁₂', 78, formula='T(12)')
glyphs.add('c', 'T₃', 6, formula='T(3)')
glyphs.add('d', 'T₄', 10, formula='T(4)')

# Vertex counts
glyphs.add('e', '1', 1)
glyphs.add('f', '3', 3)
glyphs.add('g', '12', 12)
glyphs.add('h', '16', 16)

# Transcendentals
glyphs.add('p', 'π', np.pi)
glyphs.add('q', 'φ', (1 + np.sqrt(5)) / 2)
glyphs.add('r', 'e', np.e)

# Common divisors
glyphs.add('s', '2', 2)
glyphs.add('t', '360', 360)
glyphs.add('u', '180', 180)

# =============================================================================
# EVOLVE — Run for each target
# =============================================================================

for TARGET in TARGETS:
    print("\n" + "="*70)
    print(f"TARGET: {TARGET}")
    print("="*70)
    print("\nSearching for elegant formula...\n")

    result, population, engine = evolve_with_glyphs(
        glyph_set=glyphs,
        target=TARGET,
        pop_size=500,
        head_len=8,
        generations=2000,
        mutation_rate=0.1,
        crossover_rate=0.7,
        verbose=True,
    )

    print("\n" + "-"*70)
    print("RESULT")
    print("-"*70)
    print(f"Expression: {result.expression}")
    print(f"Value: {result.value}")
    print(f"Error: {abs(result.value - TARGET):.6f}")
    print(f"Error %: {abs(result.value - TARGET) / TARGET * 100:.4f}%")

    # Show top candidates from final population
    print("\nTop candidates:")
    scored = []
    for g in population:
        val = engine.evaluate(g)
        if val is not None and not np.isnan(val) and not np.isinf(val):
            expr = engine.to_elegant(g)
            scored.append((abs(val - TARGET), val, expr))
    scored.sort(key=lambda x: x[0])

    seen = set()
    for error, val, expr in scored[:30]:
        if expr not in seen and error < 0.05:
            print(f"  {expr} = {val:.6f}  (error: {error:.6f})")
            seen.add(expr)
            if len(seen) >= 5:
                break

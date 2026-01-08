"""HoloCell Test Suite"""

import math
import sys

# Test framework
passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"✓ {name}")
    except Exception as e:
        failed += 1
        print(f"✗ {name}")
        print(f"  {e}")

def assertEqual(actual, expected, msg=''):
    if actual != expected:
        raise AssertionError(f"{msg} Expected {expected}, got {actual}")

def assertClose(actual, expected, tol, msg=''):
    diff = abs(actual - expected)
    if diff > tol:
        raise AssertionError(f"{msg} Expected ~{expected}, got {actual} (diff: {diff})")

def assertTrue(cond, msg=''):
    if not cond:
        raise AssertionError(msg or 'Failed')


# === TESTS ===

from holocell import T, B, S, CRYSTAL, ARCHITECTURE, verify_all, verify_constant
from holocell.constants import SEED, TRINITION

print("\n=== Triangular Numbers ===\n")
test("T(1) = 1", lambda: assertEqual(T(1), 1))
test("T(2) = 3", lambda: assertEqual(T(2), 3))
test("T(8) = 36", lambda: assertEqual(T(8), 36))
test("T(11) = 66", lambda: assertEqual(T(11), 66))
test("T(16) = 136", lambda: assertEqual(T(16), 136))
test("T(36) = 666", lambda: assertEqual(T(36), 666))
test("T(60) = 1830", lambda: assertEqual(T(60), 1830))

print("\n=== Bilateral Operator ===\n")
test("B(0) = 1", lambda: assertEqual(B(0), 1))
test("B(136) = 137", lambda: assertEqual(B(136), 137))
test("B(T(16)) = 137", lambda: assertEqual(B(T(16)), 137))

print("\n=== Six-Nine Operator ===\n")
test("S(0) = 0", lambda: assertEqual(S(0), 0))
test("S(9) = 19.5", lambda: assertEqual(S(9), 19.5))

print("\n=== Architecture ===\n")
test("SEED = 136", lambda: assertEqual(SEED, 136))
test("TRINITION = 408", lambda: assertEqual(TRINITION, 408))
test("T(16) × 3 = 408", lambda: assertEqual(T(16) * 3, 408))
test("Architecture has 11 elements", lambda: assertEqual(len(ARCHITECTURE), 11))

print("\n=== Crystal Constants ===\n")
test("mp/me computed", lambda: assertClose(CRYSTAL["mp/me"].computed, 1836.1526756, 1e-6))
test("mp/me error < 1.3e-7%", lambda: assertTrue(CRYSTAL["mp/me"].error_percent < 1.3e-7))
test("α⁻¹ computed", lambda: assertClose(CRYSTAL["α⁻¹"].computed, 137.0359805, 1e-6))
test("R∞ computed", lambda: assertClose(CRYSTAL["R∞"].computed, 1.0973730448, 1e-9))
test("μ/me computed", lambda: assertClose(CRYSTAL["μ/me"].computed, 206.768254, 1e-5))
test("sin²θW computed", lambda: assertClose(CRYSTAL["sin²θW"].computed, 0.231209, 1e-5))

print("\n=== Verification ===\n")
test("verify_all passes", lambda: assertTrue(all(verify_all(0.001).values())))
test("verify mp/me", lambda: assertTrue(verify_constant("mp/me", 0.0001)))

print("\n" + "=" * 50)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 50 + "\n")

sys.exit(1 if failed > 0 else 0)

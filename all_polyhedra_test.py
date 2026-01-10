#!/usr/bin/env python3
"""
Comprehensive Polyhedra Test — All 18 Vertex-Transitive Solids

Hypothesis: The octahedron (6V, 3 bilateral pairs) is optimal.
Test: Run coherence rate comparison on all 5 Platonic + 13 Archimedean solids.

Run detached:
    nohup python3 all_polyhedra_test.py > test_output.log 2>&1 &
    echo $! > test_pid.txt

Check progress:
    tail -f test_output.log
    cat all_polyhedra_results.json
"""

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
RESULTS_FILE = Path(__file__).parent / "all_polyhedra_results.json"

# Test parameters — exponential back-off
POOL_SIZES = [100, 300, 1000, 3000, 10000]
CORRUPTION_FRAC = 0.5
MAX_STEPS = 100
TRIALS_PER_CONFIG = 5

# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def _distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

def _normalize(v: Tuple[float, ...]) -> Tuple[float, ...]:
    mag = math.sqrt(sum(x*x for x in v))
    return tuple(x/mag for x in v) if mag > 1e-10 else v

def _build_adjacency(vertices: List[Tuple[float, float, float]], tol: float = 0.01) -> Dict[int, List[int]]:
    n = len(vertices)
    if n < 2:
        return {i: [] for i in range(n)}
    
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append((_distance(vertices[i], vertices[j]), i, j))
    distances.sort()
    min_dist = distances[0][0]
    
    adj = {i: [] for i in range(n)}
    for d, i, j in distances:
        if d <= min_dist * (1 + tol):
            adj[i].append(j)
            adj[j].append(i)
    return adj

# =============================================================================
# PLATONIC SOLIDS (5)
# =============================================================================

def tetrahedron() -> Tuple[int, Dict[int, List[int]]]:
    """4 vertices, degree 3"""
    adj = {0: [1,2,3], 1: [0,2,3], 2: [0,1,3], 3: [0,1,2]}
    return 4, adj

def octahedron() -> Tuple[int, Dict[int, List[int]]]:
    """6 vertices, degree 4 — THE HYPOTHESIS"""
    adj = {
        0: [2,3,4,5], 1: [2,3,4,5],  # X pair
        2: [0,1,4,5], 3: [0,1,4,5],  # Y pair
        4: [0,1,2,3], 5: [0,1,2,3],  # Z pair
    }
    return 6, adj

def cube() -> Tuple[int, Dict[int, List[int]]]:
    """8 vertices, degree 3"""
    vertices = [(x,y,z) for x in [1,-1] for y in [1,-1] for z in [1,-1]]
    adj = _build_adjacency(vertices)
    return 8, adj

def icosahedron() -> Tuple[int, Dict[int, List[int]]]:
    """12 vertices, degree 5"""
    vertices = [
        (0,1,PHI), (0,1,-PHI), (0,-1,PHI), (0,-1,-PHI),
        (1,PHI,0), (1,-PHI,0), (-1,PHI,0), (-1,-PHI,0),
        (PHI,0,1), (PHI,0,-1), (-PHI,0,1), (-PHI,0,-1),
    ]
    adj = _build_adjacency(vertices)
    return 12, adj

def dodecahedron() -> Tuple[int, Dict[int, List[int]]]:
    """20 vertices, degree 3"""
    vertices = []
    for x in [1,-1]:
        for y in [1,-1]:
            for z in [1,-1]:
                vertices.append((x,y,z))
    for x in [PHI,-PHI]:
        for y in [1/PHI,-1/PHI]:
            vertices.append((x,y,0))
    for y in [PHI,-PHI]:
        for z in [1/PHI,-1/PHI]:
            vertices.append((0,y,z))
    for z in [PHI,-PHI]:
        for x in [1/PHI,-1/PHI]:
            vertices.append((x,0,z))
    adj = _build_adjacency(vertices)
    return 20, adj


# =============================================================================
# COMPOUND: STELLA OCTANGULA (Star Tetrahedron)
# =============================================================================

def stella_octangula() -> Tuple[int, Dict[int, List[int]]]:
    """
    Stella Octangula: Two interpenetrating tetrahedra.
    8 vertices (cube vertices), degree 3.
    Octahedron forms at the intersection.
    This is the core geometry of the vortex engine.
    """
    # Tetrahedron 1: even parity cube vertices
    tet1 = [(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]
    # Tetrahedron 2: odd parity cube vertices  
    tet2 = [(-1,-1,-1), (-1,1,1), (1,-1,1), (1,1,-1)]
    
    # Each vertex connects to its own tetrahedron's 3 other vertices
    adj = {
        # Tet 1
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2],
        # Tet 2
        4: [5, 6, 7],
        5: [4, 6, 7],
        6: [4, 5, 7],
        7: [4, 5, 6],
    }
    return 8, adj


# =============================================================================
# 4D POLYTOPE PROJECTIONS (Nested geometries with radial connections)
# =============================================================================

def nested_tetrahedron() -> Tuple[int, Dict[int, List[int]]]:
    """
    5-cell projection: Tetrahedron nested in tetrahedron.
    8 vertices, each inner connected to corresponding outer.
    """
    # Outer tetrahedron: 0-3, Inner tetrahedron: 4-7
    adj = {
        # Outer shell
        0: [1, 2, 3, 4],  # + radial to inner 4
        1: [0, 2, 3, 5],
        2: [0, 1, 3, 6],
        3: [0, 1, 2, 7],
        # Inner shell
        4: [5, 6, 7, 0],  # + radial to outer 0
        5: [4, 6, 7, 1],
        6: [4, 5, 7, 2],
        7: [4, 5, 6, 3],
    }
    return 8, adj

def tesseract() -> Tuple[int, Dict[int, List[int]]]:
    """
    Tesseract (8-cell/hypercube) projection: Cube nested in cube.
    16 vertices, each inner connected to corresponding outer.
    """
    # Outer cube: 0-7, Inner cube: 8-15
    # Cube adjacency: each vertex connects to 3 neighbors
    outer = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6],
    }
    adj = {}
    # Outer cube + radial connection
    for i in range(8):
        adj[i] = outer[i] + [i + 8]
    # Inner cube + radial connection
    for i in range(8):
        adj[i + 8] = [j + 8 for j in outer[i]] + [i]
    return 16, adj

def hexadecachoron() -> Tuple[int, Dict[int, List[int]]]:
    """
    16-cell projection: Octahedron nested in octahedron.
    12 vertices, each inner connected to corresponding outer.
    The 16-cell is dual to the tesseract.
    """
    # Outer octahedron: 0-5, Inner octahedron: 6-11
    outer = {
        0: [2, 3, 4, 5],  # +X
        1: [2, 3, 4, 5],  # -X
        2: [0, 1, 4, 5],  # +Y
        3: [0, 1, 4, 5],  # -Y
        4: [0, 1, 2, 3],  # +Z
        5: [0, 1, 2, 3],  # -Z
    }
    adj = {}
    # Outer octahedron + radial
    for i in range(6):
        adj[i] = outer[i] + [i + 6]
    # Inner octahedron + radial
    for i in range(6):
        adj[i + 6] = [j + 6 for j in outer[i]] + [i]
    return 12, adj

def icositetrachoron() -> Tuple[int, Dict[int, List[int]]]:
    """
    24-cell projection: Cuboctahedron nested in cuboctahedron.
    24 vertices. The 24-cell is self-dual.
    """
    # Cuboctahedron: 12 vertices, degree 4
    vertices = [
        (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
        (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
        (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1),
    ]
    outer = _build_adjacency(vertices)
    adj = {}
    # Outer + radial
    for i in range(12):
        adj[i] = list(outer[i]) + [i + 12]
    # Inner + radial
    for i in range(12):
        adj[i + 12] = [j + 12 for j in outer[i]] + [i]
    return 24, adj

def nested_icosahedron() -> Tuple[int, Dict[int, List[int]]]:
    """
    600-cell projection approximation: Icosahedron nested in icosahedron.
    24 vertices, degree 6 (5 + 1 radial).
    """
    vertices = [
        (0,1,PHI), (0,1,-PHI), (0,-1,PHI), (0,-1,-PHI),
        (1,PHI,0), (1,-PHI,0), (-1,PHI,0), (-1,-PHI,0),
        (PHI,0,1), (PHI,0,-1), (-PHI,0,1), (-PHI,0,-1),
    ]
    outer = _build_adjacency(vertices)
    adj = {}
    for i in range(12):
        adj[i] = list(outer[i]) + [i + 12]
    for i in range(12):
        adj[i + 12] = [j + 12 for j in outer[i]] + [i]
    return 24, adj


# =============================================================================
# ARCHIMEDEAN SOLIDS (13)
# =============================================================================

def truncated_tetrahedron() -> Tuple[int, Dict[int, List[int]]]:
    """12 vertices, degree 3"""
    tet = [(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]
    vertices = []
    for i, v in enumerate(tet):
        others = [tet[j] for j in range(4) if j != i]
        for other in others:
            vertices.append(tuple(v[k] + (other[k]-v[k])/3 for k in range(3)))
    adj = _build_adjacency(vertices)
    return 12, adj

def cuboctahedron() -> Tuple[int, Dict[int, List[int]]]:
    """12 vertices, degree 4"""
    vertices = [
        (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
        (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
        (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1),
    ]
    adj = _build_adjacency(vertices)
    return 12, adj

def truncated_cube() -> Tuple[int, Dict[int, List[int]]]:
    """24 vertices, degree 3"""
    a = 1 + math.sqrt(2)
    vertices = []
    for x in [a,-a]:
        for y in [1,-1]:
            for z in [1,-1]:
                vertices.append((x,y,z))
    for x in [1,-1]:
        for y in [a,-a]:
            for z in [1,-1]:
                vertices.append((x,y,z))
    for x in [1,-1]:
        for y in [1,-1]:
            for z in [a,-a]:
                vertices.append((x,y,z))
    adj = _build_adjacency(vertices)
    return 24, adj

def truncated_octahedron() -> Tuple[int, Dict[int, List[int]]]:
    """24 vertices, degree 3"""
    vertices = []
    for perm in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                v = [0,0,0]
                v[perm[1]] = s1
                v[perm[2]] = s2 * 2
                vertices.append(tuple(v))
    adj = _build_adjacency(vertices)
    return 24, adj

def rhombicuboctahedron() -> Tuple[int, Dict[int, List[int]]]:
    """24 vertices, degree 4"""
    a = 1 + math.sqrt(2)
    vertices = []
    for x in [1,-1]:
        for y in [1,-1]:
            for z in [a,-a]:
                vertices.append((x,y,z))
    for x in [1,-1]:
        for y in [a,-a]:
            for z in [1,-1]:
                vertices.append((x,y,z))
    for x in [a,-a]:
        for y in [1,-1]:
            for z in [1,-1]:
                vertices.append((x,y,z))
    adj = _build_adjacency(vertices)
    return 24, adj

def truncated_cuboctahedron() -> Tuple[int, Dict[int, List[int]]]:
    """48 vertices, degree 3"""
    a = 1 + math.sqrt(2)
    b = 1 + 2*math.sqrt(2)
    base = [1, a, b]
    vertices = []
    for signs in [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),(-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
        for perm in [(0,1,2),(1,2,0),(2,0,1),(0,2,1),(1,0,2),(2,1,0)]:
            vertices.append((signs[0]*base[perm[0]], signs[1]*base[perm[1]], signs[2]*base[perm[2]]))
    unique = []
    for v in vertices:
        if not any(_distance(v,u) < 0.01 for u in unique):
            unique.append(v)
    adj = _build_adjacency(unique[:48])
    return 48, adj

def snub_cube() -> Tuple[int, Dict[int, List[int]]]:
    """24 vertices, degree 5"""
    t = (1 + (19 + 3*math.sqrt(33))**(1/3) + (19 - 3*math.sqrt(33))**(1/3)) / 3
    vertices = []
    for perm in [(0,1,2),(1,2,0),(2,0,1)]:
        for signs in [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]:
            base = [1, 1/t, t]
            vertices.append(tuple(signs[i]*base[perm[i]] for i in range(3)))
    for perm in [(0,2,1),(1,0,2),(2,1,0)]:
        for signs in [(-1,1,1),(1,1,-1),(1,-1,1),(-1,-1,-1)]:
            base = [1, 1/t, t]
            vertices.append(tuple(signs[i]*base[perm[i]] for i in range(3)))
    adj = _build_adjacency(vertices)
    return 24, adj

def icosidodecahedron() -> Tuple[int, Dict[int, List[int]]]:
    """30 vertices, degree 4"""
    vertices = []
    for i in range(3):
        for s in [PHI,-PHI]:
            v = [0,0,0]
            v[i] = s
            vertices.append(tuple(v))
    a, b, c = 0.5, PHI/2, (1+PHI)/2
    for perm in [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                for s3 in [1,-1]:
                    vertices.append((s1*perm[0], s2*perm[1], s3*perm[2]))
    unique = []
    for v in vertices:
        if not any(_distance(v,u) < 0.01 for u in unique):
            unique.append(v)
    adj = _build_adjacency(unique[:30])
    return 30, adj

def truncated_dodecahedron() -> Tuple[int, Dict[int, List[int]]]:
    """60 vertices, degree 3"""
    _, dod_adj = dodecahedron()
    dod_verts = []
    for x in [1,-1]:
        for y in [1,-1]:
            for z in [1,-1]:
                dod_verts.append((x,y,z))
    for x in [PHI,-PHI]:
        for y in [1/PHI,-1/PHI]:
            dod_verts.append((x,y,0))
    for y in [PHI,-PHI]:
        for z in [1/PHI,-1/PHI]:
            dod_verts.append((0,y,z))
    for z in [PHI,-PHI]:
        for x in [1/PHI,-1/PHI]:
            dod_verts.append((x,0,z))
    dod_adj = _build_adjacency(dod_verts)
    
    vertices = []
    for i, v in enumerate(dod_verts):
        for j in dod_adj[i]:
            other = dod_verts[j]
            vertices.append(tuple(v[k] + (other[k]-v[k])/3 for k in range(3)))
    adj = _build_adjacency(vertices)
    return 60, adj

def truncated_icosahedron() -> Tuple[int, Dict[int, List[int]]]:
    """60 vertices, degree 3 — The buckyball"""
    vertices = []
    for perm in [(0,1,2),(1,2,0),(2,0,1)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                v = [0,0,0]
                v[perm[1]] = s1
                v[perm[2]] = s2 * 3 * PHI
                vertices.append(tuple(v))
    a, b, c = 1, 2+PHI, 2*PHI
    for perm in [(0,1,2),(1,2,0),(2,0,1)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                for s3 in [1,-1]:
                    base = [a,b,c]
                    vertices.append((s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]]))
    a, b, c = PHI, 2, 2*PHI+1
    for perm in [(0,1,2),(1,2,0),(2,0,1)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                for s3 in [1,-1]:
                    base = [a,b,c]
                    vertices.append((s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]]))
    unique = []
    for v in vertices:
        if not any(_distance(v,u) < 0.01 for u in unique):
            unique.append(v)
    adj = _build_adjacency(unique[:60])
    return 60, adj

def rhombicosidodecahedron() -> Tuple[int, Dict[int, List[int]]]:
    """60 vertices, degree 4"""
    vertices = []
    a = PHI ** 3
    for perm in [(0,1,2),(1,2,0),(2,0,1)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                for s3 in [1,-1]:
                    base = [1,1,a]
                    vertices.append((s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]]))
    a, b, c = PHI**2, PHI, 2*PHI
    for perm in [(0,1,2),(1,2,0),(2,0,1)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                for s3 in [1,-1]:
                    base = [a,b,c]
                    vertices.append((s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]]))
    a, b = 2+PHI, PHI**2
    for perm in [(0,1,2),(1,2,0),(2,0,1)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                base = [a,0,b]
                vertices.append((s1*base[perm[0]], base[perm[1]], s2*base[perm[2]]))
    unique = []
    for v in vertices:
        if not any(_distance(v,u) < 0.01 for u in unique):
            unique.append(v)
    adj = _build_adjacency(unique[:60])
    return 60, adj

def truncated_icosidodecahedron() -> Tuple[int, Dict[int, List[int]]]:
    """120 vertices, degree 3"""
    n_ico, ico_adj = icosidodecahedron()
    # Rebuild vertices
    ico_verts = []
    for i in range(3):
        for s in [PHI,-PHI]:
            v = [0,0,0]
            v[i] = s
            ico_verts.append(tuple(v))
    a, b, c = 0.5, PHI/2, (1+PHI)/2
    for perm in [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                for s3 in [1,-1]:
                    ico_verts.append((s1*perm[0], s2*perm[1], s3*perm[2]))
    unique = []
    for v in ico_verts:
        if not any(_distance(v,u) < 0.01 for u in unique):
            unique.append(v)
    ico_verts = unique[:30]
    ico_adj = _build_adjacency(ico_verts)
    
    vertices = []
    for i, v in enumerate(ico_verts):
        for j in ico_adj[i]:
            other = ico_verts[j]
            vertices.append(tuple(v[k] + (other[k]-v[k])/3 for k in range(3)))
    unique = []
    for v in vertices:
        if not any(_distance(v,u) < 0.01 for u in unique):
            unique.append(v)
    adj = _build_adjacency(unique[:120])
    return 120, adj

def snub_dodecahedron() -> Tuple[int, Dict[int, List[int]]]:
    """60 vertices, degree 5"""
    # Approximate using icosahedron expansion
    ico_verts = [
        (0,1,PHI), (0,1,-PHI), (0,-1,PHI), (0,-1,-PHI),
        (1,PHI,0), (1,-PHI,0), (-1,PHI,0), (-1,-PHI,0),
        (PHI,0,1), (PHI,0,-1), (-PHI,0,1), (-PHI,0,-1),
    ]
    vertices = list(ico_verts)
    for i in range(len(ico_verts)):
        for j in range(i+1, len(ico_verts)):
            mid = tuple((ico_verts[i][k] + ico_verts[j][k])/2 for k in range(3))
            mid = tuple(m * 1.1 for m in _normalize(mid))
            vertices.append(mid)
    unique = []
    for v in vertices:
        if not any(_distance(v,u) < 0.05 for u in unique):
            unique.append(v)
    adj = _build_adjacency(unique[:60], tol=0.1)
    return 60, adj

# =============================================================================
# POLYHEDRA REGISTRY
# =============================================================================

POLYHEDRA = {
    # Platonic (5)
    'tetrahedron': tetrahedron,
    'octahedron': octahedron,
    'cube': cube,
    'icosahedron': icosahedron,
    'dodecahedron': dodecahedron,
    # Compound
    'stella_octangula': stella_octangula,
    # 4D Projections (nested with radial connections)
    'nested_tetrahedron': nested_tetrahedron,
    'tesseract': tesseract,
    'hexadecachoron': hexadecachoron,
    'icositetrachoron': icositetrachoron,
    'nested_icosahedron': nested_icosahedron,
    # Archimedean (13)
    'truncated_tetrahedron': truncated_tetrahedron,
    'cuboctahedron': cuboctahedron,
    'truncated_cube': truncated_cube,
    'truncated_octahedron': truncated_octahedron,
    'rhombicuboctahedron': rhombicuboctahedron,
    'truncated_cuboctahedron': truncated_cuboctahedron,
    'snub_cube': snub_cube,
    'icosidodecahedron': icosidodecahedron,
    'truncated_dodecahedron': truncated_dodecahedron,
    'truncated_icosahedron': truncated_icosahedron,
    'rhombicosidodecahedron': rhombicosidodecahedron,
    'truncated_icosidodecahedron': truncated_icosidodecahedron,
    'snub_dodecahedron': snub_dodecahedron,
}

# =============================================================================
# NETWORK SIMULATION
# =============================================================================

class Network:
    def __init__(self, adj: Dict[int, List[int]], frozen: List[int]):
        self.values = {i: 1.0 for i in adj}
        self.adj = adj
        self.frozen = set(frozen)
    
    def corrupt(self, frac: float):
        fluid = [i for i in self.values if i not in self.frozen]
        n = max(1, int(len(fluid) * frac))
        for i in random.sample(fluid, min(n, len(fluid))):
            self.values[i] = random.uniform(-2, 4)
    
    def step(self):
        updates = {}
        for i in self.values:
            if i in self.frozen:
                continue
            neighbors = [j for j in self.adj.get(i, []) if j in self.values]
            if neighbors:
                avg = sum(self.values[j] for j in neighbors) / len(neighbors)
                updates[i] = self.values[i] + 0.5 * (avg - self.values[i])
        for i, v in updates.items():
            self.values[i] = v
    
    def coherence(self) -> float:
        fluid = [i for i in self.values if i not in self.frozen]
        if not fluid:
            return 1.0
        cohs = [max(0, 1 - abs(self.values[i] - 1)) for i in fluid]
        return sum(cohs) / len(cohs)


def build_shell_network(seed_adj: Dict[int, List[int]], n_frozen: int, n_fluid: int) -> Tuple[Dict[int, List[int]], List[int]]:
    """Frozen seed as shell, fluid nodes inside with sparse contact."""
    adj = {i: list(seed_adj.get(i, [])) for i in range(n_frozen)}
    
    n_layers = max(1, int(n_fluid ** 0.5))
    per_layer = max(1, n_fluid // n_layers)
    
    fluid_start = n_frozen
    for layer in range(n_layers):
        for pos in range(per_layer):
            idx = fluid_start + layer * per_layer + pos
            if idx >= n_frozen + n_fluid:
                break
            adj[idx] = []
            
            if layer == 0:
                frozen_contact = pos % n_frozen
                adj[idx].append(frozen_contact)
                adj[frozen_contact].append(idx)
            else:
                prev_idx = fluid_start + (layer-1) * per_layer + pos
                if prev_idx in adj:
                    adj[idx].append(prev_idx)
                    adj[prev_idx].append(idx)
            
            if pos > 0:
                prev_pos = fluid_start + layer * per_layer + pos - 1
                if prev_pos in adj:
                    adj[idx].append(prev_pos)
                    adj[prev_pos].append(idx)
    
    frozen = list(range(n_frozen))
    return adj, frozen


def measure_recovery_rate(seed_name: str, n_fluid: int, trials: int = 3) -> dict:
    """Measure coherence recovery rate for a seed geometry."""
    n_frozen, seed_adj = POLYHEDRA[seed_name]()
    
    rates = []
    steps_to_90 = []
    final_cohs = []
    
    for _ in range(trials):
        adj, frozen = build_shell_network(seed_adj, n_frozen, n_fluid)
        net = Network(adj, frozen)
        net.corrupt(CORRUPTION_FRAC)
        
        initial = net.coherence()
        
        # Measure rate in first 10 steps
        cohs = [initial]
        for step in range(MAX_STEPS):
            net.step()
            cohs.append(net.coherence())
            if len(cohs) <= 11:
                continue
        
        # Average rate in first 10 steps
        rate = (cohs[10] - cohs[0]) / 10 if len(cohs) > 10 else 0
        rates.append(rate)
        
        # Steps to 90%
        hit_90 = MAX_STEPS
        for i, c in enumerate(cohs):
            if c >= 0.9:
                hit_90 = i
                break
        steps_to_90.append(hit_90)
        final_cohs.append(cohs[-1])
    
    return {
        'avg_rate': sum(rates) / len(rates),
        'avg_steps_to_90': sum(steps_to_90) / len(steps_to_90),
        'avg_final_coherence': sum(final_cohs) / len(final_cohs),
    }

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def save_results(results: dict):
    """Save results to JSON file (atomic write)."""
    temp_file = RESULTS_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(results, f, indent=2)
    temp_file.rename(RESULTS_FILE)

def load_results() -> dict:
    """Load existing results if present."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}

def run_all_tests():
    """Run coherence test on all 18 polyhedra."""
    print("=" * 70)
    print("COMPREHENSIVE POLYHEDRA TEST — All 18 Vertex-Transitive Solids")
    print("=" * 70)
    print(f"\nPool sizes: {POOL_SIZES}")
    print(f"Corruption: {CORRUPTION_FRAC*100:.0f}%")
    print(f"Trials per config: {TRIALS_PER_CONFIG}")
    print(f"Results file: {RESULTS_FILE}")
    print()
    
    # Load any existing results (for resumption)
    results = load_results()
    if results:
        print(f"Resuming from {len(results)} existing results\n")
    
    start_time = time.time()
    
    for name, fn in POLYHEDRA.items():
        if name in results:
            print(f"[SKIP] {name} (already tested)")
            continue
        
        n_frozen, _ = fn()
        print(f"\n{name} ({n_frozen} frozen vertices)")
        print("-" * 50)
        
        seed_results = {
            'n_frozen': n_frozen,
            'pool_results': {},
        }
        
        for pool_size in POOL_SIZES:
            print(f"  Pool {pool_size:>5}...", end=" ", flush=True)
            pool_start = time.time()
            
            result = measure_recovery_rate(name, pool_size, TRIALS_PER_CONFIG)
            
            seed_results['pool_results'][str(pool_size)] = result
            elapsed = time.time() - pool_start
            
            print(f"rate={result['avg_rate']:.4f}  →90%={result['avg_steps_to_90']:.1f}  [{elapsed:.1f}s]")
        
        # Calculate averages across pool sizes
        all_rates = [r['avg_rate'] for r in seed_results['pool_results'].values()]
        all_steps = [r['avg_steps_to_90'] for r in seed_results['pool_results'].values()]
        seed_results['avg_rate'] = sum(all_rates) / len(all_rates)
        seed_results['avg_steps_to_90'] = sum(all_steps) / len(all_steps)
        
        results[name] = seed_results
        save_results(results)  # Save after each polyhedron
        
        print(f"  → Average: rate={seed_results['avg_rate']:.4f}, steps={seed_results['avg_steps_to_90']:.1f}")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("RANKINGS BY POOL SIZE — Does the winner change at scale?")
    print("=" * 70)
    
    for pool_size in POOL_SIZES:
        pool_key = str(pool_size)
        pool_ranked = sorted(
            [(name, data['pool_results'][pool_key]['avg_rate'], data['n_frozen']) 
             for name, data in results.items()],
            key=lambda x: x[1], reverse=True
        )
        print(f"\nPool {pool_size:,}:")
        for i, (name, rate, verts) in enumerate(pool_ranked[:5], 1):
            mark = "★" if name == 'octahedron' else " "
            print(f"  {i}. {name:<28} ({verts:>3}V) rate={rate:.4f}{mark}")
        oct_rank = next(i for i, (n, _, _) in enumerate(pool_ranked, 1) if n == 'octahedron')
        if oct_rank > 5:
            print(f"  ... octahedron ranked #{oct_rank}")
    
    print("\n" + "=" * 70)
    print("OVERALL RANKINGS (averaged across pool sizes)")
    print("=" * 70)
    
    ranked = sorted(results.items(), key=lambda x: x[1]['avg_rate'], reverse=True)
    
    print(f"\n{'Rank':<6}{'Polyhedron':<30}{'Vertices':<10}{'Rate':<12}{'Steps→90%':<12}")
    print("-" * 70)
    
    for i, (name, data) in enumerate(ranked, 1):
        mark = "★" if name == 'octahedron' else " "
        print(f"{i:<6}{name:<30}{data['n_frozen']:<10}{data['avg_rate']:.4f}{mark:<7}{data['avg_steps_to_90']:.1f}")
    
    winner = ranked[0][0]
    print(f"\n🏆 OPTIMAL GEOMETRY: {winner}")
    
    if winner == 'octahedron':
        print("✅ HYPOTHESIS CONFIRMED: Octahedron is optimal!")
    else:
        print(f"❌ HYPOTHESIS REJECTED: {winner} beats octahedron")
        oct_rank = next(i for i, (n, _) in enumerate(ranked, 1) if n == 'octahedron')
        print(f"   Octahedron ranked #{oct_rank}")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_all_tests()

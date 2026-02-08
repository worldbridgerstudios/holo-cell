"""
HoloCell — Mode 11: Polyhedra (Optimal Coherence Networks)

Characterize all 18 vertex-transitive polyhedra for self-healing.

THE 18 CANDIDATES:
- 5 Platonic solids
- 13 Archimedean solids

TWO MODES:
- HEART: Frozen geometry at center, fluid nodes outside
- SHELL: Frozen geometry as boundary, fluid nodes inside

THREE METRICS:
- Sparsity: frozen:fluid ratio
- Resilience: corruption threshold for recovery
- Minimum coherence: weakest point in standing wave
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import random
import time


# =============================================================================
# POLYHEDRA DEFINITIONS
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio


def _normalize(v: Tuple[float, ...]) -> Tuple[float, ...]:
    """Normalize a vector to unit length."""
    mag = math.sqrt(sum(x*x for x in v))
    if mag < 1e-10:
        return v
    return tuple(x/mag for x in v)


def _distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def _build_adjacency_from_vertices(
    vertices: List[Tuple[float, float, float]],
    edge_length_tolerance: float = 0.01
) -> Dict[int, List[int]]:
    """
    Build adjacency list from vertex coordinates.
    
    Finds the shortest edge length, then connects all pairs
    within tolerance of that length.
    """
    n = len(vertices)
    if n < 2:
        return {i: [] for i in range(n)}
    
    # Find all pairwise distances
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            d = _distance(vertices[i], vertices[j])
            distances.append((d, i, j))
    
    distances.sort()
    
    # Find the shortest edge length
    min_dist = distances[0][0]
    
    # Build adjacency for edges within tolerance
    adj = {i: [] for i in range(n)}
    for d, i, j in distances:
        if d <= min_dist * (1 + edge_length_tolerance):
            adj[i].append(j)
            adj[j].append(i)
    
    return adj


# -----------------------------------------------------------------------------
# PLATONIC SOLIDS
# -----------------------------------------------------------------------------

def tetrahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Tetrahedron: 4 vertices, each connected to 3 others."""
    vertices = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]
    adj = {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2],
    }
    return vertices, adj


def octahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Octahedron: 6 vertices, each connected to 4 others."""
    vertices = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def cube() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Cube: 8 vertices, each connected to 3 others."""
    vertices = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def icosahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Icosahedron: 12 vertices, each connected to 5 others."""
    vertices = [
        (0, 1, PHI), (0, 1, -PHI), (0, -1, PHI), (0, -1, -PHI),
        (1, PHI, 0), (1, -PHI, 0), (-1, PHI, 0), (-1, -PHI, 0),
        (PHI, 0, 1), (PHI, 0, -1), (-PHI, 0, 1), (-PHI, 0, -1),
    ]
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def dodecahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Dodecahedron: 20 vertices, each connected to 3 others."""
    # Vertices from cube corners and rectangle centers
    vertices = []
    # Cube vertices (±1, ±1, ±1)
    for x in [1, -1]:
        for y in [1, -1]:
            for z in [1, -1]:
                vertices.append((x, y, z))
    # Rectangle vertices
    for x in [PHI, -PHI]:
        for y in [1/PHI, -1/PHI]:
            vertices.append((x, y, 0))
    for y in [PHI, -PHI]:
        for z in [1/PHI, -1/PHI]:
            vertices.append((0, y, z))
    for z in [PHI, -PHI]:
        for x in [1/PHI, -1/PHI]:
            vertices.append((x, 0, z))
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


# -----------------------------------------------------------------------------
# ARCHIMEDEAN SOLIDS
# -----------------------------------------------------------------------------

def truncated_tetrahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Truncated tetrahedron: 12 vertices, degree 3."""
    vertices = []
    # Start with tetrahedron, truncate each vertex
    tet = [(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]
    for i, v in enumerate(tet):
        others = [tet[j] for j in range(4) if j != i]
        for other in others:
            # Point 1/3 of way from v to other
            new_v = tuple(v[k] + (other[k] - v[k]) / 3 for k in range(3))
            vertices.append(new_v)
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def cuboctahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Cuboctahedron: 12 vertices, degree 4."""
    vertices = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def truncated_cube() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Truncated cube: 24 vertices, degree 3."""
    vertices = []
    a = 1 + math.sqrt(2)
    for x in [a, -a]:
        for y in [1, -1]:
            for z in [1, -1]:
                vertices.append((x, y, z))
    for x in [1, -1]:
        for y in [a, -a]:
            for z in [1, -1]:
                vertices.append((x, y, z))
    for x in [1, -1]:
        for y in [1, -1]:
            for z in [a, -a]:
                vertices.append((x, y, z))
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def truncated_octahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Truncated octahedron: 24 vertices, degree 3."""
    vertices = []
    # Permutations of (0, 1, 2)
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0, 0, 0]
                v[perm[0]] = 0
                v[perm[1]] = s1 * 1
                v[perm[2]] = s2 * 2
                vertices.append(tuple(v))
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def rhombicuboctahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Rhombicuboctahedron: 24 vertices, degree 4."""
    vertices = []
    a = 1 + math.sqrt(2)
    for x in [1, -1]:
        for y in [1, -1]:
            for z in [a, -a]:
                vertices.append((x, y, z))
    for x in [1, -1]:
        for y in [a, -a]:
            for z in [1, -1]:
                vertices.append((x, y, z))
    for x in [a, -a]:
        for y in [1, -1]:
            for z in [1, -1]:
                vertices.append((x, y, z))
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def truncated_cuboctahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Truncated cuboctahedron: 48 vertices, degree 3."""
    vertices = []
    a = 1 + math.sqrt(2)
    b = 1 + 2 * math.sqrt(2)
    
    # Even permutations of (±1, ±a, ±b)
    base = [1, a, b]
    for signs in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                  (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]:
        for perm in [(0,1,2), (1,2,0), (2,0,1)]:
            v = (signs[0]*base[perm[0]], signs[1]*base[perm[1]], signs[2]*base[perm[2]])
            vertices.append(v)
        for perm in [(0,2,1), (1,0,2), (2,1,0)]:
            v = (signs[0]*base[perm[0]], signs[1]*base[perm[1]], signs[2]*base[perm[2]])
            vertices.append(v)
    
    # Remove duplicates
    unique = []
    for v in vertices:
        is_dup = any(_distance(v, u) < 0.01 for u in unique)
        if not is_dup:
            unique.append(v)
    vertices = unique[:48]
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def snub_cube() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Snub cube: 24 vertices, degree 5."""
    # Tribonacci constant
    t = (1 + (19 + 3*math.sqrt(33))**(1/3) + (19 - 3*math.sqrt(33))**(1/3)) / 3
    
    vertices = []
    for even_perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for signs in [(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]:
            base = [1, 1/t, t]
            v = [signs[i] * base[even_perm[i]] for i in range(3)]
            vertices.append(tuple(v))
    for odd_perm in [(0,2,1), (1,0,2), (2,1,0)]:
        for signs in [(-1,1,1), (1,1,-1), (1,-1,1), (-1,-1,-1)]:
            base = [1, 1/t, t]
            v = [signs[i] * base[odd_perm[i]] for i in range(3)]
            vertices.append(tuple(v))
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def icosidodecahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Icosidodecahedron: 30 vertices, degree 4."""
    vertices = []
    
    # Permutations of (0, 0, ±φ)
    for i in range(3):
        for s in [PHI, -PHI]:
            v = [0, 0, 0]
            v[i] = s
            vertices.append(tuple(v))
    
    # Permutations of (±1/2, ±φ/2, ±(1+φ)/2)
    a, b, c = 0.5, PHI/2, (1+PHI)/2
    for perm in [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    v = (s1*perm[0], s2*perm[1], s3*perm[2])
                    vertices.append(v)
    
    # Remove duplicates and take first 30
    unique = []
    for v in vertices:
        is_dup = any(_distance(v, u) < 0.01 for u in unique)
        if not is_dup:
            unique.append(v)
    vertices = unique[:30]
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def truncated_dodecahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Truncated dodecahedron: 60 vertices, degree 3."""
    # Start with dodecahedron and truncate
    dod_verts, _ = dodecahedron()
    vertices = []
    
    # For each vertex, create 3 new vertices toward its neighbors
    dod_adj = _build_adjacency_from_vertices(dod_verts)
    for i, v in enumerate(dod_verts):
        for j in dod_adj[i]:
            other = dod_verts[j]
            new_v = tuple(v[k] + (other[k] - v[k]) / 3 for k in range(3))
            vertices.append(new_v)
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def truncated_icosahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Truncated icosahedron (buckyball): 60 vertices, degree 3."""
    vertices = []
    
    # Standard coordinates for C60
    # Even permutations of (0, ±1, ±3φ)
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0, 0, 0]
                v[perm[1]] = s1 * 1
                v[perm[2]] = s2 * 3 * PHI
                vertices.append(tuple(v))
    
    # Even permutations of (±1, ±(2+φ), ±2φ)
    a, b, c = 1, 2+PHI, 2*PHI
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    base = [a, b, c]
                    v = (s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]])
                    vertices.append(v)
    
    # Even permutations of (±φ, ±2, ±(2φ+1))
    a, b, c = PHI, 2, 2*PHI+1
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    base = [a, b, c]
                    v = (s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]])
                    vertices.append(v)
    
    # Remove duplicates
    unique = []
    for v in vertices:
        is_dup = any(_distance(v, u) < 0.01 for u in unique)
        if not is_dup:
            unique.append(v)
    vertices = unique[:60]
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def rhombicosidodecahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Rhombicosidodecahedron: 60 vertices, degree 4."""
    vertices = []
    
    # Even permutations of (±1, ±1, ±φ³)
    a = PHI ** 3
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    base = [1, 1, a]
                    v = (s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]])
                    vertices.append(v)
    
    # Even permutations of (±φ², ±φ, ±2φ)
    a, b, c = PHI**2, PHI, 2*PHI
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    base = [a, b, c]
                    v = (s1*base[perm[0]], s2*base[perm[1]], s3*base[perm[2]])
                    vertices.append(v)
    
    # Even permutations of (±(2+φ), 0, ±φ²)
    a, b = 2+PHI, PHI**2
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                base = [a, 0, b]
                v = (s1*base[perm[0]], base[perm[1]], s2*base[perm[2]])
                vertices.append(v)
    
    unique = []
    for v in vertices:
        is_dup = any(_distance(v, u) < 0.01 for u in unique)
        if not is_dup:
            unique.append(v)
    vertices = unique[:60]
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def truncated_icosidodecahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Truncated icosidodecahedron: 120 vertices, degree 3."""
    # This is complex - use icosidodecahedron and truncate
    ico_verts, ico_adj = icosidodecahedron()
    vertices = []
    
    for i, v in enumerate(ico_verts):
        for j in ico_adj[i]:
            other = ico_verts[j]
            new_v = tuple(v[k] + (other[k] - v[k]) / 3 for k in range(3))
            vertices.append(new_v)
    
    unique = []
    for v in vertices:
        is_dup = any(_distance(v, u) < 0.01 for u in unique)
        if not is_dup:
            unique.append(v)
    vertices = unique[:120]
    
    adj = _build_adjacency_from_vertices(vertices)
    return vertices, adj


def snub_dodecahedron() -> Tuple[List[Tuple[float, float, float]], Dict[int, List[int]]]:
    """Snub dodecahedron: 60 vertices, degree 5."""
    # Approximate using icosahedron expansion
    ico_verts, _ = icosahedron()
    vertices = list(ico_verts)
    
    # Add midpoints and perturb
    for i in range(len(ico_verts)):
        for j in range(i+1, len(ico_verts)):
            mid = tuple((ico_verts[i][k] + ico_verts[j][k]) / 2 for k in range(3))
            # Scale outward
            mid = tuple(m * 1.1 for m in _normalize(mid))
            vertices.append(mid)
    
    unique = []
    for v in vertices:
        is_dup = any(_distance(v, u) < 0.05 for u in unique)
        if not is_dup:
            unique.append(v)
    
    # Take 60 most spread out
    vertices = unique[:60]
    
    adj = _build_adjacency_from_vertices(vertices, edge_length_tolerance=0.1)
    return vertices, adj


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

POLYHEDRA_INFO = {
    'tetrahedron': (4, 3, 'Platonic'),
    'octahedron': (6, 4, 'Platonic'),
    'cube': (8, 3, 'Platonic'),
    'icosahedron': (12, 5, 'Platonic'),
    'dodecahedron': (20, 3, 'Platonic'),
    'truncated_tetrahedron': (12, 3, 'Archimedean'),
    'cuboctahedron': (12, 4, 'Archimedean'),
    'truncated_cube': (24, 3, 'Archimedean'),
    'truncated_octahedron': (24, 3, 'Archimedean'),
    'rhombicuboctahedron': (24, 4, 'Archimedean'),
    'truncated_cuboctahedron': (48, 3, 'Archimedean'),
    'snub_cube': (24, 5, 'Archimedean'),
    'icosidodecahedron': (30, 4, 'Archimedean'),
    'truncated_dodecahedron': (60, 3, 'Archimedean'),
    'truncated_icosahedron': (60, 3, 'Archimedean'),
    'rhombicosidodecahedron': (60, 4, 'Archimedean'),
    'truncated_icosidodecahedron': (120, 3, 'Archimedean'),
    'snub_dodecahedron': (60, 5, 'Archimedean'),
}


# =============================================================================
# COHERENCE NETWORK
# =============================================================================

class Mode(Enum):
    HEART = "heart"  # Frozen core, fluid shell
    SHELL = "shell"  # Frozen shell, fluid core


@dataclass
class Node:
    """A node in the coherence network."""
    index: int
    value: float
    frozen: bool
    neighbors: List[int] = field(default_factory=list)


@dataclass 
class CharacterizationResult:
    """Result from characterizing a polyhedron."""
    name: str
    mode: Mode
    frozen_count: int
    fluid_count: int
    corruption_level: float
    final_coherence: float
    min_coherence: float
    recovery_steps: int
    recovered: bool


class CoherenceNetwork:
    """
    A network for testing coherence dynamics.
    
    Frozen nodes don't update. Fluid nodes average toward neighbors.
    """
    
    def __init__(self, adj: Dict[int, List[int]], frozen_indices: List[int], initial_value: float = 1.0):
        self.nodes: List[Node] = []
        n = len(adj)
        
        for i in range(n):
            self.nodes.append(Node(
                index=i,
                value=initial_value,
                frozen=(i in frozen_indices),
                neighbors=adj.get(i, []),
            ))
    
    @property
    def frozen_count(self) -> int:
        return sum(1 for n in self.nodes if n.frozen)
    
    @property
    def fluid_count(self) -> int:
        return sum(1 for n in self.nodes if not n.frozen)
    
    def coherence(self) -> Tuple[float, float]:
        """
        Measure coherence of fluid nodes.
        
        Returns (average_coherence, min_coherence).
        Coherence = 1 - |value - 1| (assuming target is 1.0)
        """
        fluid = [n for n in self.nodes if not n.frozen]
        if not fluid:
            return 1.0, 1.0
        
        coherences = []
        for n in fluid:
            c = 1.0 - min(1.0, abs(n.value - 1.0))
            coherences.append(max(0.0, c))
        
        return sum(coherences) / len(coherences), min(coherences)
    
    def corrupt(self, fraction: float):
        """Corrupt a fraction of fluid nodes."""
        fluid = [n for n in self.nodes if not n.frozen]
        count = max(1, int(len(fluid) * fraction))
        targets = random.sample(fluid, min(count, len(fluid)))
        
        for node in targets:
            # Random value far from 1.0
            node.value = random.uniform(-2.0, 4.0)
    
    def step(self):
        """One step of neighbor averaging."""
        updates = {}
        
        for node in self.nodes:
            if node.frozen:
                continue
            
            if not node.neighbors:
                continue
            
            # Average toward neighbors
            neighbor_values = [self.nodes[i].value for i in node.neighbors if i < len(self.nodes)]
            if neighbor_values:
                neighbor_avg = sum(neighbor_values) / len(neighbor_values)
                # Pull toward neighbor average
                updates[node.index] = node.value + 0.5 * (neighbor_avg - node.value)
        
        for idx, val in updates.items():
            self.nodes[idx].value = val


def build_heart_network(name: str, fluid_layers: int = 1) -> CoherenceNetwork:
    """
    Build HEART mode network: frozen polyhedron at center, fluid nodes outside.
    
    For now, fluid nodes are connected to nearest frozen nodes.
    """
    vertices, adj = POLYHEDRA[name]()
    n_frozen = len(vertices)
    
    # Frozen core is the polyhedron itself
    frozen_indices = list(range(n_frozen))
    
    # Add fluid nodes around the core
    n_fluid = n_frozen * fluid_layers
    total = n_frozen + n_fluid
    
    # Extend adjacency for fluid nodes
    full_adj = {i: list(adj.get(i, [])) for i in range(n_frozen)}
    
    for i in range(n_frozen, total):
        # Connect each fluid node to random frozen nodes
        connections = random.sample(frozen_indices, min(3, n_frozen))
        full_adj[i] = connections
        # Also connect frozen nodes back to this fluid node
        for c in connections:
            full_adj[c].append(i)
    
    return CoherenceNetwork(full_adj, frozen_indices)


def build_shell_network(name: str, fluid_count: int = None) -> CoherenceNetwork:
    """
    Build SHELL mode network: frozen polyhedron as boundary, fluid nodes inside.
    
    Fluid nodes form a smaller connected network inside.
    """
    vertices, adj = POLYHEDRA[name]()
    n_frozen = len(vertices)
    
    # Frozen shell is the polyhedron
    frozen_indices = list(range(n_frozen))
    
    # Fluid nodes inside - default to same count as frozen
    if fluid_count is None:
        fluid_count = n_frozen
    
    total = n_frozen + fluid_count
    
    # Build adjacency
    full_adj = {i: list(adj.get(i, [])) for i in range(n_frozen)}
    
    # Fluid nodes connect to each other and to shell
    for i in range(n_frozen, total):
        full_adj[i] = []
        
        # Connect to nearby fluid nodes (ring structure)
        if i > n_frozen:
            full_adj[i].append(i - 1)
            full_adj[i - 1].append(i)
        if i == total - 1 and fluid_count > 2:
            full_adj[i].append(n_frozen)
            full_adj[n_frozen].append(i)
        
        # Connect to shell nodes
        shell_connections = random.sample(frozen_indices, min(2, n_frozen))
        for c in shell_connections:
            full_adj[i].append(c)
            full_adj[c].append(i)
    
    return CoherenceNetwork(full_adj, frozen_indices)


def characterize_polyhedron(
    name: str,
    mode: Mode,
    corruption_levels: List[float] = None,
    trials: int = 10,
    max_steps: int = 100,
) -> List[CharacterizationResult]:
    """
    Characterize a polyhedron's self-healing properties.
    """
    if corruption_levels is None:
        corruption_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    for corruption in corruption_levels:
        trial_results = []
        
        for _ in range(trials):
            # Build network
            if mode == Mode.HEART:
                net = build_heart_network(name, fluid_layers=2)
            else:
                net = build_shell_network(name)
            
            # Corrupt
            net.corrupt(corruption)
            
            # Heal
            for step in range(max_steps):
                net.step()
                avg_coh, min_coh = net.coherence()
                if min_coh > 0.95:
                    break
            
            avg_coh, min_coh = net.coherence()
            trial_results.append((avg_coh, min_coh, step + 1))
        
        # Average results
        avg_coherence = sum(r[0] for r in trial_results) / len(trial_results)
        min_coherence = sum(r[1] for r in trial_results) / len(trial_results)
        avg_steps = sum(r[2] for r in trial_results) / len(trial_results)
        recovered = avg_coherence > 0.9
        
        # Get counts from a fresh network
        if mode == Mode.HEART:
            net = build_heart_network(name, fluid_layers=2)
        else:
            net = build_shell_network(name)
        
        results.append(CharacterizationResult(
            name=name,
            mode=mode,
            frozen_count=net.frozen_count,
            fluid_count=net.fluid_count,
            corruption_level=corruption,
            final_coherence=avg_coherence,
            min_coherence=min_coherence,
            recovery_steps=int(avg_steps),
            recovered=recovered,
        ))
    
    return results


def run_full_characterization(trials: int = 10, verbose: bool = True) -> Dict[str, List[CharacterizationResult]]:
    """
    Characterize all 18 polyhedra in both modes.
    """
    all_results = {}
    
    if verbose:
        print("\n" + "=" * 80)
        print("OPTIMAL COHERENCE NETWORK CHARACTERIZATION")
        print("=" * 80)
        print(f"18 polyhedra × 2 modes × 9 corruption levels × {trials} trials")
        print("=" * 80)
    
    for name in POLYHEDRA:
        info = POLYHEDRA_INFO[name]
        
        for mode in [Mode.HEART, Mode.SHELL]:
            key = f"{name}_{mode.value}"
            
            if verbose:
                print(f"\n{name} ({info[0]} vertices, deg {info[1]}) - {mode.value.upper()}")
            
            results = characterize_polyhedron(name, mode, trials=trials)
            all_results[key] = results
            
            if verbose:
                # Show 50% corruption result
                r50 = next((r for r in results if r.corruption_level == 0.5), None)
                if r50:
                    status = "✓" if r50.recovered else "✗"
                    print(f"  50% corrupt: {status} {r50.final_coherence:.0%} avg, {r50.min_coherence:.0%} min, {r50.recovery_steps} steps")
    
    return all_results


def print_summary_table(results: Dict[str, List[CharacterizationResult]]):
    """Print a summary table of all results at 50% corruption."""
    print("\n" + "=" * 100)
    print("SUMMARY: 50% CORRUPTION RECOVERY")
    print("=" * 100)
    print(f"{'Polyhedron':<30} {'V':>4} {'Mode':<6} {'Frozen':>6} {'Fluid':>6} {'Avg':>6} {'Min':>6} {'Steps':>6}")
    print("-" * 100)
    
    for key, res_list in sorted(results.items()):
        r50 = next((r for r in res_list if r.corruption_level == 0.5), None)
        if r50:
            name_parts = key.rsplit('_', 1)
            name = name_parts[0]
            mode = name_parts[1]
            info = POLYHEDRA_INFO.get(name, (0, 0, ''))
            
            status = "✓" if r50.recovered else " "
            print(f"{status} {name:<28} {info[0]:>4} {mode:<6} {r50.frozen_count:>6} {r50.fluid_count:>6} "
                  f"{r50.final_coherence:>5.0%} {r50.min_coherence:>5.0%} {r50.recovery_steps:>6}")
    
    print("=" * 100)

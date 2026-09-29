"""Schemdraw diagrams, adapted from sircuitenum.visualize (Eli Weissler, Mohit Bhat).

The parallel-branch fan-out is ported under GPL-3.0; see LICENSE.
Enumeration-specific graph tables are replaced by NetworkX or explicit layouts.
"""

import networkx as nx


def draw_circuit_diagram(circuit, out=None, *, layout="planar", scale=4., spread=.4,
                         label=True, node_labels=True, label_loc=None, flux_labels=None):
    """Return a Schemdraw Drawing, optionally saved to SVG, PDF, or PNG.

    layout: 'planar' (spring fallback), 'spring', 'polygon', or {node: (x, y)}.
    Explicit positions are multiplied by scale; automatic layouts have shortest
    edge length scale. spread sets parallel-lane spacing as a fraction of that
    length. label controls branch names; label_loc overrides node label sides.
    flux_labels maps text (including $math$) to (x, y) positions, multiplied by
    scale just like explicit node positions. These are display-only annotations.
    Ground is drawn even when isolated. Voltage sources have + toward the source
    node. Boxed junctions include intrinsic C; an inductor's C is shown in parallel.
    Install the optional dependencies with ``pip install '.[draw]'``.
    """
    import numpy as np
    import schemdraw
    import schemdraw.elements as elm
    from schemdraw.segments import Segment

    if scale <= 0 or spread <= 0:
        raise ValueError("scale and spread must be positive")
    graph = nx.Graph((b.tail, b.head) for b in circuit.branches)
    if isinstance(layout, dict):
        if set(circuit.nodes) - set(layout):
            raise ValueError("layout must give a position for every circuit node")
        pos = {n: np.asarray(layout[n], dtype=float) * scale for n in circuit.nodes}
    else:
        if layout == "planar" and nx.is_planar(graph):
            pos = nx.planar_layout(graph)
        elif layout in ("planar", "spring"):
            pos = nx.spring_layout(graph, seed=0)
        elif layout == "polygon":
            pos = nx.circular_layout(graph)
        else:
            raise ValueError("layout must be planar, spring, polygon, or a node-position dict")
        # Put ground below the circuit; retain labels instead of renumbering nodes.
        center = np.mean(list(pos.values()), axis=0)
        pos = {n: p - center for n, p in pos.items()}
        if circuit.ground in pos:
            x, y = pos[circuit.ground]
            if x or y:
                rotation = np.array([[-y, x], [-x, -y]]) / np.hypot(x, y)
                pos = {n: rotation @ p for n, p in pos.items()}
        shortest = min(np.linalg.norm(pos[a] - pos[b]) for a, b in graph.edges)
        pos = {n: p * scale / shortest for n, p in pos.items()}
        if circuit.ground not in pos:
            pos[circuit.ground] = np.min(list(pos.values()), axis=0) - [2, 1]
    if any(p.shape != (2,) or not np.isfinite(p).all() for p in pos.values()):
        raise ValueError("node positions must be finite (x, y) pairs")
    if len({tuple(p) for p in pos.values()}) != len(pos):
        raise ValueError("node positions must be distinct")

    # Group by unordered endpoints without comparing heterogeneous node labels.
    groups = {}
    for b in circuit.branches:
        group = groups.setdefault(frozenset((b.tail, b.head)), [])
        group.append((b, b.type, b.name))
        if b.type == "L" and b.C != 0:
            group.append((b, "C", f"$C_{{{b.name}}}$"))
    min_r = min(np.linalg.norm(pos[a] - pos[b]) for a, b in graph.edges)
    drawing = schemdraw.Drawing(show=False, canvas="matplotlib", fontsize=11)
    elements = {"C": elm.Capacitor, "J": elm.Josephson, "L": elm.Inductor2, "V": elm.SourceV}
    for group in groups.values():
        first = group[0][0]
        a, b = pos[first.tail], pos[first.head]
        direction = b - a
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        if len(group) > 1:
            drawing.add(elm.Wire().at(a).to(a + direction / 4))
            drawing.add(elm.Wire().at(b).to(b - direction / 4))
        for i, (_, kind, name) in enumerate(group):
            start, end = a, b
            if len(group) > 1:
                offset = (i - (len(group) - 1) / 2) * spread * min_r * normal
                start, end = a + direction / 4 + offset, b - direction / 4 + offset
                # Common leads and perpendicular rails, as in sircuitenum.
                drawing.add(elm.Wire().at(a + direction / 4).to(start))
                drawing.add(elm.Wire().at(b - direction / 4).to(end))
            if kind == "V" and first.tail != circuit.ground:
                start, end = end, start  # SourceV has its + at end; V means V_source - V_ground.
            element = elements[kind]().endpoints(start, end)
            if kind == "J":
                gap = (float("nan"), float("nan"))
                element.segments[0] = Segment([(-.25, 0), gap, (.25, 0)])
                element.segments.append(Segment([(-.25, -.25), (.25, -.25), (.25, .25),
                                                 (-.25, .25), (-.25, -.25), (.25, .25),
                                                 gap, (-.25, .25), (.25, -.25)]))
                name = f"{name}, $C_{{{name}}}$"
            if label:
                element.label(name)
            drawing.add(element)
    sides = {"top": (0, 1), "right": (1, 0), "bottom": (0, -1), "left": (-1, 0)}
    for node, point in pos.items():
        dot = elm.Dot(radius=.07).at(point)
        if node_labels:
            neighbors = [pos[n] - point for n in graph.neighbors(node)] if node in graph else []
            directions = [v / np.linalg.norm(v) for v in neighbors]
            side = min(sides, key=lambda s: max((np.dot(sides[s], v) for v in directions), default=0))
            dot.label(str(node), loc=(label_loc or {}).get(node, "left" if node == circuit.ground else side))
        drawing.add(dot)
        if node == circuit.ground:
            drawing.add(elm.Ground().at(point))
    for text, point in (flux_labels or {}).items():
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("flux label positions must be finite (x, y) pairs")
        drawing.add(elm.Label().at(point * scale).label(text))
    if out is not None:
        drawing.save(str(out))
    return drawing

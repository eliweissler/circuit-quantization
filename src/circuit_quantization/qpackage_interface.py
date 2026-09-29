"""scqubits export, adapted from sircuitenum.qpackage_interface (GPL-3.0)."""

import math

import sympy as sym

from .units import farad_to_ghz, henry_to_ghz, subs_constants


def _scqubits_input(circuit, subs, node_order):
    if circuit.source_nodes:
        raise ValueError("scqubits export does not support voltage-source branches")
    order = tuple(circuit.internal_nodes if node_order is None else node_order)
    if len(order) != len(circuit.internal_nodes) or set(order) != set(circuit.internal_nodes):
        raise ValueError("node_order must contain each internal node exactly once")
    nodes = {node: i + 1 for i, node in enumerate(order)}
    nodes[circuit.ground] = 0
    rows, indices = [], {}

    def parameter(name, value):
        value = subs_constants(sym.sympify(value).subs(subs or {}))
        if value.free_symbols:
            raise ValueError(f"Missing substitutions for {name}: {value.free_symbols}")
        if value.is_real is not True or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return f"{name} = {float(value):.17g}"

    for i, branch in enumerate(circuit.branches):
        edge = f"{nodes[branch.tail]}, {nodes[branch.head]}"
        indices[branch.name] = len(rows)
        if branch.type == "J":
            ej = parameter(f"EJ_{i}", branch.EJ)
            ec = parameter(f"EC_{i}", farad_to_ghz(branch.C))
            rows.append(f"- [JJ, {edge}, {ej}, {ec}]")
        elif branch.type == "L":
            el = parameter(f"EL_{i}", henry_to_ghz(branch.L))
            rows.append(f"- [L, {edge}, {el}]")
        if branch.C != 0 and branch.type != "J":
            ec = parameter(f"EC_{i}", farad_to_ghz(branch.C))
            rows.append(f"- [C, {edge}, {ec}]")
    return "branches:\n" + "\n".join(rows) + "\n", indices


def scqubits_yaml(circuit, *, subs=None, node_order=None):
    """Return scqubits YAML with energy parameters in GHz; no scqubits import.

    Supply numeric substitutions for all circuit symbols. Constants are evaluated
    automatically. Ground maps to 0; node_order (default internal_nodes) maps to
    1, 2, ... . Parameter names EJ_i, EC_i, EL_i use circuit.branches indices.
    Junction C is included in JJ; inductor C becomes a parallel capacitor.
    """
    return _scqubits_input(circuit, subs, node_order)[0]


def to_scqubits(circuit, *, subs=None, node_order=None, symbolic=False,
                transformation_matrix=None, closure_branches=None,
                cutoff=None, trunc_num=None, **kwargs):
    """Build a scqubits Circuit (or SymbolicCircuit when symbolic=True).

    YAML conversion follows scqubits_yaml. transformation_matrix rows follow
    node_order. closure_branches is an ordered list of original branch names;
    scqubits validates the loop choice. Fluxes and offset charges are left at
    scqubits defaults: this exports topology and elements, not a Loops gauge.
    Set those parameters on the returned object in its own flux convention.

    cutoff is a scalar or a dict keyed by scqubits' 1-based mode indices.
    trunc_num optionally enables one subsystem per mode, with scalar or list
    truncations. These numerical options cannot be used with symbolic=True.
    kwargs are forwarded to Circuit or SymbolicCircuit.from_yaml.
    """
    import numpy as np
    import scqubits as scq

    yaml, indices = _scqubits_input(circuit, subs, node_order)
    if symbolic and (cutoff is not None or trunc_num is not None):
        raise ValueError("cutoff and trunc_num require a numerical Circuit")
    constructor = scq.SymbolicCircuit.from_yaml if symbolic else scq.Circuit
    result = constructor(yaml, from_file=False, **kwargs)
    config = {}
    if transformation_matrix is not None:
        config["transformation_matrix"] = np.asarray(transformation_matrix, dtype=float)
    if closure_branches is not None:
        names = tuple(closure_branches)
        if len(set(names)) != len(names) or set(names) - indices.keys():
            raise ValueError("closure_branches must be distinct circuit branch names")
        config["closure_branches"] = [result.branches[indices[name]] for name in names]
    if config:
        result.configure(**config)
    if not symbolic:
        modes = sorted(result.var_categories["periodic"] + result.var_categories["extended"])
        if cutoff is not None:
            for kind, prefix in (("periodic", "n"), ("extended", "ext")):
                for mode in result.var_categories[kind]:
                    setattr(result, f"cutoff_{prefix}_{mode}", cutoff[mode] if isinstance(cutoff, dict) else cutoff)
        if trunc_num is not None:
            dims = [trunc_num] * len(modes) if isinstance(trunc_num, int) else list(trunc_num)
            if len(dims) != len(modes):
                raise ValueError("trunc_num must have one entry per dynamical mode")
            if len(modes) > 1:
                result.configure(system_hierarchy=[[mode] for mode in modes], subsystem_trunc_dims=dims)
            elif modes:
                result.truncated_dim = dims[0]
    return result

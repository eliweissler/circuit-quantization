"""Numeric export regression for the Julia residual-frame workload."""
import numpy as np
import sympy as sym
import circuit_quantization as cq
from circuits import coupled_zero_pis


def test_numerified_coupled_zero_pis_random_float_frame_exports():
    circuit = coupled_zero_pis()
    loops = cq.Loops.from_tree(circuit, cq.default_tree(circuit))
    loops = cq.Loops(circuit, loops.G, loops.names, loops.closure_branches, dict.fromkeys(loops.names, 0))
    result = cq.quantize(circuit, loops)
    parameters = set(circuit.variables)
    values = {s: (1e-15 * (i+1) if s.name.startswith('C') else
                  1e-8 * (i+1) if s.name.startswith('L') else 2.)
              for i, s in enumerate(sorted(parameters, key=str))}
    result = cq.numerify(result, values)
    rng = np.random.default_rng(1042)
    S = np.eye(len(result.coords.types))
    for i, old in enumerate(result.coords.types):
        for j, new in enumerate(result.coords.types):
            if new != 'c' and (old, new) in {('c','e'),('c','h'),('e','e'),('e','h'),('h','e'),('h','h')}:
                S[i,j] += rng.normal(scale=.12)
    data = cq.to_spec(result.transform(S))
    assert cq.load_spec(data).E_L == cq.load_spec(data).E_L.T


# ------------------------------------------------- explicit round-off zeroing

from dataclasses import replace
import json
import math
import pytest
from circuits import fluxonium_pair


def numeric_pair():
    """Numerified two-extended-mode circuit whose matrices and omega a test replaces."""
    circuit = fluxonium_pair()
    res = cq.quantize(circuit, cq.Loops.from_tree(circuit, cq.default_tree(circuit)))
    values = {s: 1 for s in res.ham.E_C.free_symbols | res.ham.E_L.free_symbols
              if s.name not in {"e", "h", "h_GHz", "hbar", "Phi0"}}
    return cq.numerify(res, values)


def with_matrix(res, rows, key="E_L"):
    return replace(res, ham=replace(res.ham, **{key: sym.ImmutableMatrix(rows)}))


def with_omega(res, rows, types=None):
    coords = replace(res.coords, omega_J_tilde=sym.ImmutableMatrix(rows),
                     types=res.coords.types if types is None else types)
    return replace(res, coords=coords)


def export(res, key="E_L", **kwargs):
    data = cq.to_spec(res, **kwargs)
    cq.load_spec(data)
    return data[key]


@pytest.mark.parametrize("key", ["E_C", "E_L"])
def test_default_relative_zeroing_of_float_residue(key):
    res = numeric_pair()
    scale = 2.0
    small = with_matrix(res, [[1.0, 1e-13 * scale], [1e-13 * scale, scale]], key)
    assert export(small, key)[0][1] == export(small, key)[1][0] == "0"
    kept = with_matrix(res, [[1.0, 1e-11 * scale], [1e-11 * scale, scale]], key)
    rows = export(kept, key)
    assert rows[0][1] == rows[1][0] != "0"
    assert float(rows[0][1]) == pytest.approx(1e-11 * scale)


def test_explicit_relative_and_absolute_tolerances():
    res = with_matrix(numeric_pair(), [[1.0, 2e-11], [2e-11, 2.0]])
    assert export(res, zero_rtol=1e-9)[0][1] == "0"
    assert export(res, zero_rtol=0, zero_atol=1e-10)[0][1] == "0"
    assert export(res, zero_rtol=0, zero_atol=1e-11)[0][1] != "0"
    tiny = with_matrix(numeric_pair(), [[1.0, 2e-17], [2e-17, 2.0]])
    rows = export(tiny, zero_rtol=0, zero_atol=0)
    assert rows[0][1] == rows[1][0] != "0"
    assert float(rows[0][1]) == pytest.approx(2e-17)


def test_tolerance_acts_on_symmetrized_pair():
    # Asymmetric within 1e-10 of the scale; the pair is decided and written together.
    res = with_matrix(numeric_pair(), [[1.0, 1e-13], [3e-13, 2.0]])
    rows = export(res)
    assert rows[0][1] == rows[1][0] == "0"
    rows = export(res, zero_rtol=0)
    assert rows[0][1] == rows[1][0] != "0"
    assert float(rows[0][1]) == pytest.approx(2e-13)


def test_diagonal_float_residue_zeroed_but_exact_tiny_entry_kept():
    res = numeric_pair()
    rows = export(with_matrix(res, [[1e-14, 0.5], [0.5, 2.0]]))
    assert rows[0][0] == "0" and rows[1][1] != "0"
    exact = sym.Rational(1, 10**15)
    rows = export(with_matrix(res, [[exact, 0.5], [0.5, 2.0]]))
    assert rows[0][0] != "0"
    rows = export(with_matrix(res, [[2, exact], [exact, 2]]), zero_atol=1.0)
    assert rows[0][1] == rows[1][0] != "0"


def test_symbolic_matrix_is_not_zeroed():
    x = sym.Symbol("E_x", real=True)
    rows = export(with_matrix(numeric_pair(), [[x, 1e-20], [1e-20, 2.0]]), zero_atol=1.0)
    assert rows[0][1] == rows[1][0] != "0"


def test_asymmetry_threshold_unchanged():
    res = numeric_pair()
    export(with_matrix(res, [[1.0, 0.5], [0.5 + 1.8e-10, 2.0]]))
    with pytest.raises(ValueError, match=r"E_L.*asymmetry.*\[0,1\]"):
        cq.to_spec(with_matrix(res, [[1.0, 0.5], [0.5 + 2.2e-10, 2.0]]))
    with pytest.raises(ValueError, match=r"asymmetry.*\[0,1\]"):
        cq.to_spec(with_matrix(res, [[1.0, 0.5], [0.5 + 2.2e-10, 2.0]]), zero_rtol=1e-9)


@pytest.mark.parametrize("name", ["zero_rtol", "zero_atol"])
@pytest.mark.parametrize("value", [-1e-12, math.nan, math.inf, "1e-9", None, True, 1j])
def test_invalid_tolerances(name, value):
    with pytest.raises(ValueError, match=name):
        cq.to_spec(numeric_pair(), **{name: value})
    with pytest.raises(ValueError, match=name):
        cq.to_spec(None, **{name: value})  # Checked before any work.


def test_omega_float_residue_is_zeroed():
    res = with_omega(numeric_pair(), [[1.0, sym.Float(1e-17)], [sym.Float(-3e-17), 1.0]])
    data = cq.to_spec(res)
    assert [c["omega"] for c in data["cosines"]] == [[1, 0], [0, 1]]
    loaded = cq.load_spec(data)
    assert [[w != 0 for w in omega] for _, omega, _ in loaded.cosines] == [[True, False], [False, True]]
    data = cq.to_spec(res, zero_rtol=0)
    assert [c["omega"] for c in data["cosines"]] == [[1, -3e-17], [1e-17, 1]]
    cq.load_spec(data)


def test_omega_relative_scale_is_per_cosine_and_exact_entries_are_kept():
    rows = [[0.5, sym.Rational(1, 10**15)], [sym.Float(4e-13), sym.Float(1e-3)]]
    data = cq.to_spec(with_omega(numeric_pair(), rows))
    # Cosine 0 has scale 0.5: its 4e-13 is residue. Cosine 1 has scale 1e-3: nothing is.
    assert data["cosines"][0]["omega"] == [0.5, 0]
    assert data["cosines"][1]["omega"] == [1e-15, 1e-3]
    cq.load_spec(data)
    data = cq.to_spec(with_omega(numeric_pair(), rows), zero_rtol=0, zero_atol=1.0)
    assert data["cosines"][0]["omega"] == [0.5, 4e-13]  # zero_atol does not apply to omega.


def test_omega_zeroing_does_not_mask_invalid_compact_entry():
    res = numeric_pair()
    residue = with_omega(res, [[1.0, sym.Float(1e-17)], [0, 1.0]], types=("c", "e"))
    assert [c["omega"] for c in cq.to_spec(residue)["cosines"]] == [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="J2.*varphi_1.*integer"):
        cq.to_spec(residue, zero_rtol=0)
    for value, large in [(0.5, 1.0), (1e-3, 1e10)]:
        bad = with_omega(res, [[1.0, sym.Float(value)], [0, sym.Float(large)]], types=("c", "e"))
        with pytest.raises(ValueError, match="J2.*varphi_1.*integer"):
            cq.to_spec(bad)


def test_option_reaches_save_spec_and_to_record(tmp_path):
    res = with_matrix(numeric_pair(), [[1.0, 2e-11], [2e-11, 2.0]])
    path = tmp_path / "spec.json"
    data = cq.save_spec(res, path, zero_rtol=1e-9)
    assert data["E_L"][0][1] == json.loads(path.read_text())["E_L"][0][1] == "0"
    assert cq.save_spec(res, path)["E_L"][0][1] != "0"
    data = cq.to_record(res, tmp_path / "record.json", zero_rtol=0, zero_atol=1e-10)
    assert data["E_L"][0][1] == "0"
    cq.load_spec(data)
    assert cq.to_record(res)["E_L"][0][1] != "0"

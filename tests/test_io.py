"""Hamiltonian interchange: exact small cases, circuit coverage, and hostile input."""

from dataclasses import FrozenInstanceError, replace
import json
import math
import random

import pytest
import sympy as sym

import circuit_quantization as cq
from circuits import (
    COUPLED_ZERO_PI_TYPES, COUPLED_ZERO_PI_Z,
    coupled, coupled_zero_pis, fluxonium_pair, harmonium, kite, random_circuit,
    rhombus, transmon_fluxonium, zero_pi, zero_pi_asymmetric,
)


def fluxonium():
    ec, el, ej = sym.symbols("E_C E_L E_J", positive=True)
    flux = sym.Symbol("varphi_ext", real=True)
    circuit = cq.Circuit([
        cq.Branch("J", 1, 0, C=cq.ghz_to_farad(ec), EJ=ej),
        cq.Branch("L", 1, 0, L=cq.ghz_to_henry(el)),
    ], ground=0)
    loops = cq.Loops.from_tree(circuit, cq.default_tree(circuit), {"L": cq.phase_to_weber(flux)})
    return cq.quantize(circuit, loops)


def real_symbols(expr):
    # The interchange contract preserves names, but deliberately normalizes
    # source assumptions (e.g. positive=True) to real=True.
    return expr.xreplace({s: sym.Symbol(s.name, real=True) for s in expr.free_symbols})


@pytest.fixture
def spec():
    return {
        "format": "kronfree-spec", "version": 1, "units": "GHz",
        "parameters": ["E_C", "E_J", "n_g"],
        "modes": [{"name": "varphi_1", "type": "c"}],
        "E_C": [["E_C"]], "E_L": [["0"]],
        "flux_linear": ["0"], "n_g": ["n_g"],
        "cosines": [{"energy": "E_J", "omega": [1], "offset": "0"}],
    }


def test_fluxonium_round_trip_and_paths(tmp_path):
    res = fluxonium()
    data = cq.to_spec(res)
    loaded = cq.load_spec(data)
    assert isinstance(loaded, cq.HamiltonianSpec)
    assert loaded.modes == (("varphi_1", "e"),)
    assert loaded.E_C == sym.ImmutableMatrix([[sym.Symbol("E_C", real=True)]])
    assert data["cosines"][0]["omega"] == [1]
    assert loaded.flux_linear[0].has(sym.Symbol("varphi_ext", real=True))
    assert sym.simplify(loaded.hamiltonian() - real_symbols(cq.subs_constants(res.H))) == 0
    assert all(p.is_real for p in loaded.parameters)
    assert cq.load_spec(json.dumps(data)) == loaded
    path = tmp_path / "fluxonium.json"
    assert cq.save_spec(res, path) == data
    assert cq.load_spec(path) == cq.load_spec(str(path)) == loaded
    assert data["_generator"]["package"] == "circuit-quantization"
    assert data["_generator"]["branches"] == list(res.circuit.branch_names)
    with pytest.raises(FrozenInstanceError):
        loaded.constant = sym.S.One
    with pytest.raises(TypeError):
        loaded.E_C[0, 0] = 0


def test_substitutions_and_export_does_not_modify_result():
    res = fluxonium()
    original = res.H
    ec = next(s for s in res.H.free_symbols if s.name == "E_C")
    loaded = cq.load_spec(cq.to_spec(res, subs={ec: sym.Rational(1, 3)}, simplify=False))
    assert loaded.E_C[0] == sym.Rational(1, 3)
    assert "E_C" not in {p.name for p in loaded.parameters}
    assert sym.simplify(loaded.hamiltonian() - real_symbols(cq.subs_constants(res.H.subs(ec, sym.Rational(1, 3))))) == 0
    assert res.H == original


@pytest.mark.parametrize("factory", [
    harmonium, zero_pi, zero_pi_asymmetric, fluxonium_pair, transmon_fluxonium,
    kite, rhombus, coupled, coupled_zero_pis,
])
def test_named_circuit_round_trips(factory):
    circuit = factory()
    # Use the notes' established basis for the six-mode coupled zero-pi;
    # give its five capacitances distinct ratios of one symbolic scale to
    # keep symbolic flux assignment tractable. All six modes and fluxes remain.
    if factory is coupled_zero_pis:
        scale = sym.Symbol("E_cap", positive=True)
        capacitances = sorted({s for b in circuit.branches for s in b.C.free_symbols}, key=str)
        values = {s: cq.ghz_to_farad(scale * sym.Rational(i + 2, 10)) for i, s in enumerate(capacitances)}
        circuit = cq.Circuit([replace(b, C=b.C.subs(values)) for b in circuit.branches], ground=circuit.ground)
    basis = {"Z_full": COUPLED_ZERO_PI_Z, "types": COUPLED_ZERO_PI_TYPES} if factory is coupled_zero_pis else {}
    res = cq.quantize(circuit, cq.Loops.from_tree(circuit, cq.default_tree(circuit)), **basis)
    # Express SI circuit parameters in GHz before writing the coefficients.
    subs = {s: cq.ghz_to_farad(sym.Symbol("E_" + s.name, positive=True))
            for s in circuit.variables if s.name.startswith("C")}
    subs.update({s: cq.ghz_to_henry(sym.Symbol("E_" + s.name, positive=True))
                 for s in circuit.variables if s.name.startswith("L")})
    loaded = cq.load_spec(cq.to_spec(res, subs=subs))
    expected = real_symbols(cq.subs_constants(res.H.subs(subs)))
    actual = loaded.hamiltonian()
    # Large symbolic circuits are checked at five deterministic real points,
    # using 30-digit evaluation to avoid cancellation in SI voltage factors.
    symbols = sorted(expected.free_symbols | actual.free_symbols, key=str)
    rng = random.Random(418)
    for _ in range(5):
        values = {s: sym.Rational(rng.randint(5, 30), 11) for s in symbols}
        left, right = expected.evalf(30, subs=values), actual.evalf(30, subs=values)
        assert abs(left - right) <= sym.Float("1e-12") * max(abs(left), abs(right), 1)
    for _, omega, _ in loaded.cosines:
        assert all(w.is_Integer for w, (_, t) in zip(omega, loaded.modes) if t == "c")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_circuit_round_trips(seed):
    circuit = random_circuit(seed)
    res = cq.quantize(circuit, cq.Loops.from_tree(circuit, cq.default_tree(circuit)))
    difference = cq.load_spec(cq.to_spec(res)).hamiltonian() - real_symbols(cq.subs_constants(res.H))
    assert sym.simplify(difference) == 0


def test_lc_and_transmon():
    for branches, typ, cosines in [
        ([cq.Branch("C", 1, 0, C=cq.ghz_to_farad(1)), cq.Branch("L", 1, 0, L=cq.ghz_to_henry(2))], "h", 0),
        ([cq.Branch("J", 1, 0, C=cq.ghz_to_farad(1), EJ=3)], "c", 1),
    ]:
        circuit = cq.Circuit(branches, ground=0)
        res = cq.quantize(circuit, cq.Loops.from_tree(circuit, cq.default_tree(circuit)))
        loaded = cq.load_spec(cq.to_spec(res))
        assert loaded.modes == (("varphi_1", typ),)
        assert len(loaded.cosines) == cosines
        assert sym.simplify(loaded.hamiltonian() - real_symbols(cq.subs_constants(res.H))) == 0


def test_invalid_omega_names_junction_and_mode():
    res = fluxonium()
    for omega in (sym.Rational(1, 2), sym.Symbol("a", real=True)):
        coords = replace(res.coords, types=("c",), omega_J_tilde=sym.ImmutableMatrix([[omega]]))
        with pytest.raises(ValueError, match="J.*varphi_1"):
            cq.to_spec(replace(res, coords=coords))


@pytest.mark.parametrize("value", [sym.Rational(5, 19), sym.Float(0.12345678901234567), sym.Float("0.12345678901234567890123456789", 30)])
def test_numeric_precision(value):
    res = fluxonium()
    res = replace(res, ham=replace(res.ham, E_C=sym.ImmutableMatrix([[value]])))
    actual = cq.load_spec(cq.to_spec(res)).E_C[0]
    if value.is_Rational:
        assert actual == value
    else:
        assert abs(actual - value) <= sym.Float(2) ** (1 - value._prec) * abs(value)


@pytest.mark.parametrize("expression,expected", [
    ("1 // 3", sym.Rational(1, 3)),
    ("2 / 3 // 4", sym.Rational(8, 3)),
    ("-2^2 + 2^-2", sym.Rational(-15, 4)),
    ("2^3^2", sym.Integer(512)),
    ("2 .^ 2 .* 3 ./ 2 .+ 1 .- 4", sym.Integer(3)),
    (".-2 .+ .+1", sym.Integer(-1)),
    ("log(2, 8)", sym.Integer(3)),
    ("pi + π + ℯ", 2 * sym.pi + sym.E),
    ("abs(-2) + sign(-3) + real(2) + imag(2) + conj(2)", sym.Integer(5)),
    ("sqrt(4) + exp(0) + cos(0) + sin(0) + tan(0) + cosh(0) + sinh(0) + tanh(0) + log(1)", sym.Integer(5)),
])
def test_expression_grammar(spec, expression, expected):
    spec["constant"] = expression
    assert cq.load_spec(spec).constant == expected


@pytest.mark.parametrize("expression", [
    "run(`ls`)", "__import__('os')", "E_C.__class__", "E_C[0]", "[x for x in E_C]",
    "x = 1", "sqrt.(4)", "sin(1; x=2)", "sin(1, 2)", "im", "cis(1)", "cispi(1)",
    "I", "e", "Inf", "NaN", "sqrt(-1)", "log(-1)", "1/0", "1 + missing", "True", "", "1 # comment",
])
def test_reject_invalid_expressions(spec, expression):
    spec["E_C"] = [[expression]]
    with pytest.raises(ValueError, match="E_C"):
        cq.load_spec(spec)


@pytest.mark.parametrize("key,value", [
    ("format", "other"), ("version", 2), ("version", True), ("units", "MHz"),
    ("parameters", ["E_C", "E_C"]), ("parameters", ["pi"]), ("parameters", ["end"]),
    ("parameters", ["bad-name"]), ("parameters", ["n_1"]), ("parameters", [3]),
    ("modes", []), ("modes", [{"name": "v", "type": "fr"}]),
    ("E_C", [["1", "0"]]), ("E_L", ["0"]), ("flux_linear", []), ("n_g", [False]),
    ("cosines", [{"energy": "1", "omega": [0.5], "offset": "0"}]),
    ("cosines", [{"energy": "1", "omega": [True], "offset": "0"}]),
    ("cosines", [{"energy": "1", "omega": [math.inf], "offset": "0"}]),
    ("cosines", [{"energy": "1", "omega": ["1"], "offset": "0"}]),
    ("cosines", [{"energy": "1", "omega": [], "offset": "0"}]),
    ("cosines", [{"energy": "1", "omega": [1]}]),
    ("charge_cosines", [{"energy": "1", "kappa": [0], "offset": "0"}]),
    ("operators", []), ("noise_channels", []), ("typo", 1),
])
def test_validation(spec, key, value):
    spec[key] = value
    with pytest.raises(ValueError):
        cq.load_spec(spec)


@pytest.mark.parametrize("key", ["format", "version", "parameters", "modes", "E_C", "E_L", "flux_linear", "n_g", "cosines"])
def test_missing_fields(spec, key):
    del spec[key]
    with pytest.raises(ValueError, match=key):
        cq.load_spec(spec)


def test_optional_fields_and_metadata(spec):
    del spec["units"]
    spec["charge_cosines"] = []
    spec["_custom"] = {"anything": [1, 2, 3]}
    assert cq.load_spec(spec).constant == 0
    spec["modes"][0]["typo"] = 1
    with pytest.raises(ValueError, match="typo"):
        cq.load_spec(spec)


def test_bad_export_expressions(tmp_path):
    res = fluxonium()
    for bad in (sym.I, sym.Function("f")(sym.Symbol("x")), sym.Symbol("end"), sym.oo):
        changed = replace(res, lag=replace(res.lag, EJ=(bad,)))
        path = tmp_path / "protected.json"
        path.write_text("original")
        with pytest.raises(ValueError):
            cq.save_spec(changed, path)
        assert path.read_text() == "original"


def test_export_euler_constant():
    res = fluxonium()
    res = replace(res, lag=replace(res.lag, EJ=(sym.E,)))
    assert cq.load_spec(cq.to_spec(res)).cosines[0][0] == sym.E


def test_bad_top_level_and_duplicate_json_keys(spec):
    for source in ([], "[]", "null", '{"format":', 5):
        with pytest.raises(ValueError):
            cq.load_spec(source)
    source = json.dumps(spec).replace('"version": 1', '"version": 1, "version": 2')
    with pytest.raises(ValueError, match="duplicate"):
        cq.load_spec(source)


def test_distinct_symbols_with_the_same_name_are_rejected():
    res = fluxonium()
    x, positive_x = sym.Symbol("x", real=True), sym.Symbol("x", positive=True)
    res = replace(res, ham=replace(res.ham, E_C=sym.ImmutableMatrix([[x + positive_x]])))
    with pytest.raises(ValueError, match="share the name"):
        cq.to_spec(res)


def test_float_integer_omega_and_extended_noninteger_omega():
    res = fluxonium()
    for value, typ in [(sym.Float(1), "c"), (sym.Float("0.5"), "e"), (sym.Rational(1, 3), "e")]:
        coords = replace(res.coords, types=(typ,), omega_J_tilde=sym.ImmutableMatrix([[value]]))
        data = cq.to_spec(replace(res, coords=coords))
        assert data["cosines"][0]["omega"] == [float(value)]
        assert abs(float(cq.load_spec(data).cosines[0][1][0]) - float(value)) < 1e-15


def test_voltage_sign_and_flux_offset_round_trip():
    cj, cg, voltage = sym.symbols("C_J C_g V", positive=True)
    circuit = cq.Circuit([
        cq.Branch("J", 1, 0, C=cj, EJ=1), cq.Branch("Cg", 1, 2, C=cg),
        cq.Branch("V", 0, 2, V=voltage),
    ], ground=0)
    names, _ = cq.tree_to_G(circuit, cq.default_tree(circuit))
    res = cq.quantize(circuit, cq.Loops.from_tree(circuit, cq.default_tree(circuit), dict.fromkeys(names, 0)))
    loaded = cq.load_spec(cq.to_spec(res))
    expected = sym.Symbol("n_g1", real=True) - cg * voltage / (2 * cq.units.e)
    assert sym.simplify(loaded.n_g[0] - real_symbols(cq.subs_constants(expected))) == 0
    assert sym.simplify(loaded.hamiltonian() - real_symbols(cq.subs_constants(res.H))) == 0

    # Put the flux on the junction explicitly, rather than the inductor.
    base = fluxonium()
    custom = cq.quantize(base.circuit, base.loops, B=sym.ImmutableMatrix([[-1], [0]]))
    loaded = cq.load_spec(cq.to_spec(custom))
    assert loaded.cosines[0][2] == -sym.Symbol("varphi_ext", real=True)
    assert loaded.flux_linear[0] == 0
    assert sym.simplify(loaded.hamiltonian() - real_symbols(cq.subs_constants(custom.H))) == 0


def test_time_dependent_flux_requires_supported_expression_or_substitution():
    base = fluxonium()
    flux = sym.Function("Phi")(base.circuit.t)
    loops = cq.Loops.from_tree(base.circuit, cq.default_tree(base.circuit), {"L": flux})
    res = cq.quantize(base.circuit, loops)
    with pytest.raises(ValueError, match="flux_linear"):
        cq.to_spec(res)
    phase = sym.Symbol("f", real=True)
    data = cq.to_spec(res, subs={flux: cq.phase_to_weber(phase)})
    assert cq.load_spec(data).flux_linear[0] == sym.Symbol("E_L", real=True) * phase


def test_numeric_strings_and_constant_offset(spec):
    baseline = cq.load_spec(spec).hamiltonian()
    spec["constant"] = "1.25e-20 + 7 // 13"
    loaded = cq.load_spec(spec)
    expected = sym.Float("1.25e-20") + sym.Rational(7, 13)
    assert loaded.constant == expected
    assert sym.simplify(loaded.hamiltonian() - baseline - expected) == 0


def test_nested_validation_and_no_input_aliasing(spec):
    loaded = cq.load_spec(spec)
    spec["E_C"][0][0] = "999"
    spec["modes"][0]["name"] = "other"
    assert loaded.E_C[0] == sym.Symbol("E_C", real=True)
    assert loaded.modes[0][0] == "varphi_1"
    spec["cosines"][0]["typo"] = 1
    with pytest.raises(ValueError, match="typo"):
        cq.load_spec(spec)

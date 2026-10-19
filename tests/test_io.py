"""Hamiltonian interchange: exact small cases, circuit coverage, and hostile input."""

from dataclasses import FrozenInstanceError, replace
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

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


def test_substituted_rational_coefficient_finishes():
    # Julia integration's two-zero-pi E_C[theta1, theta1] used to hang in
    # cancel after substitution. Isolate it so a regression fails in 20 s
    # rather than hanging the suite. The expected result is exact.
    script = """
from dataclasses import replace
import sympy as sym
import circuit_quantization as cq
x_B1, x_J1, x_B2, x_J2, x_c = sym.symbols('x_B1 x_J1 x_B2 x_J2 x_c', positive=True)
E_CS1, E_CS2, E_CJ1, E_CJ2, g = sym.symbols('E_CS1 E_CS2 E_CJ1 E_CJ2 g', positive=True)
e00 = x_B1*x_J1*(x_B2*x_J2 + 4*x_B2*x_c + 4*x_J2*x_c) / (2*(
    x_B1*x_B2*x_J1 + x_B1*x_B2*x_J2 + 4*x_B1*x_B2*x_c
    + x_B1*x_J1*x_J2 + 4*x_B1*x_J2*x_c + x_B2*x_J1*x_J2
    + 4*x_B2*x_J1*x_c + 4*x_J1*x_J2*x_c))
subs = {
    x_c: (16*E_CS1*E_CS2 - g**2) / (16*g),
    x_J1: (-16*E_CJ1*E_CS1*E_CS2 + E_CJ1*g**2) / (8*E_CJ1*g - 16*E_CS1*E_CS2 + g**2),
    x_B1: (16*E_CJ1*E_CS1*E_CS2 - E_CJ1*g**2) / (16*E_CJ1*E_CS2 + 4*E_CJ1*g - 16*E_CS1*E_CS2 + g**2),
}
swap = {E_CS1: E_CS2, E_CS2: E_CS1, E_CJ1: E_CJ2}
subs[x_J2], subs[x_B2] = subs[x_J1].xreplace(swap), subs[x_B1].xreplace(swap)
circuit = cq.Circuit([cq.Branch('J', 1, 0, C=1, EJ=1)], ground=0)
res = cq.quantize(circuit, cq.Loops.from_branch_dict(circuit, {}))
res = replace(res, ham=replace(res.ham, E_C=sym.ImmutableMatrix([[e00]])))
loaded = cq.load_spec(cq.to_spec(res, subs=subs))
assert loaded.E_C[0] == sym.Symbol('E_CS1', real=True) / 2
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(cq.__file__).parents[1]), *sys.path]))
    subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True, timeout=20)


def test_custom_mode_names_and_sector_renaming(tmp_path):
    # Compact sectors use their retained-mode indices, including intervening
    # harmonic modes: n_g1 and n_g3, not compact-only indices 1 and 2.
    circuit = cq.Circuit([
        cq.Branch("J1", 1, 0, C=1, EJ=1), cq.Branch("J2", 3, 0, C=1, EJ=1),
        cq.Branch("C", 2, 0, C=1), cq.Branch("L", 2, 0, L=1),
    ], ground=0)
    res = cq.quantize(circuit, cq.Loops.from_circuit(circuit, zero_capacitive=True),
                      Z_full=sym.eye(3), types=("c", "h", "c"))
    ng1, ng3, ng_theta1, ng_theta2 = sym.symbols("n_g1 n_g3 n_g_theta1 n_g_theta2", real=True)
    assert res.ham.n_g == sym.ImmutableMatrix([ng1, 0, ng3])
    names = ("theta1", "resonator", "theta2")
    data = cq.to_spec(res, mode_names=names)
    assert [m["name"] for m in data["modes"]] == list(names)
    assert cq.load_spec(data).hamiltonian() == cq.load_spec(cq.to_spec(res)).hamiltonian()
    path = tmp_path / "named.json"
    cq.save_spec(res, path, mode_names=names, subs={ng1: ng_theta1, ng3: ng_theta2})
    loaded = cq.load_spec(path)
    assert loaded.n_g == sym.ImmutableMatrix([ng_theta1, 0, ng_theta2])
    assert loaded.hamiltonian() == cq.load_spec(data).hamiltonian().subs({ng1: ng_theta1, ng3: ng_theta2})


@pytest.mark.parametrize("names", [[], ["one", "two"], "phi", [""], [1], {"phi"}, {"phi": 1}])
def test_invalid_mode_names(names):
    with pytest.raises(ValueError, match="mode_names"):
        cq.to_spec(fluxonium(), mode_names=names)


def test_duplicate_mode_names():
    circuit = fluxonium_pair()
    res = cq.quantize(circuit, cq.Loops.from_circuit(circuit, zero_capacitive=True))
    with pytest.raises(ValueError, match="mode_names"):
        cq.to_spec(res, mode_names=["phi", "phi"])


def test_voltage_bias_on_noncompact_mode_round_trips():
    cj, cg, voltage = sym.symbols("C_J C_g V", positive=True)
    circuit = cq.Circuit([
        cq.Branch("J", 1, 0, C=cj, EJ=1), cq.Branch("L", 1, 0, L=1),
        cq.Branch("Cg", 1, 2, C=cg), cq.Branch("V", 0, 2, V=voltage),
    ], ground=0)
    res = cq.quantize(circuit, cq.Loops.from_circuit(circuit, zero_capacitive=True))
    assert res.coords.types == ("e",)
    loaded = cq.load_spec(cq.to_spec(res))
    expected = real_symbols(cq.subs_constants(-cg * voltage / (2 * cq.units.e)))
    assert sym.simplify(loaded.n_g[0] - expected) == 0
    assert not any(s.name.startswith("n_g") for s in loaded.parameters)
    assert sym.simplify(loaded.hamiltonian() - real_symbols(cq.subs_constants(res.H))) == 0


@pytest.fixture
def two_mode_spec(spec):
    spec.update(parameters=["a", "b", "g", "x"],
                modes=[{"name": "p", "type": "e"}, {"name": "q", "type": "e"}],
                E_C=[["a", "0"], ["0", "a"]], E_L=[["a", "0"], ["0", "a"]],
                flux_linear=["0", "0"], n_g=["0", "0"], cosines=[])
    return spec


@pytest.mark.parametrize("key", ["E_C", "E_L"])
@pytest.mark.parametrize("left,right", [("a", "b"), ("2*x", "x"), ("0", "1e-30")])
def test_nonsymmetric_matrix_is_rejected(two_mode_spec, key, left, right):
    two_mode_spec[key][0][1], two_mode_spec[key][1][0] = left, right
    with pytest.raises(ValueError, match="symmetric") as error:
        cq.load_spec(two_mode_spec)
    assert f"{key}[0,1]" in str(error.value)
    assert f"{key}[1,0]" in str(error.value)
    assert repr(left) in str(error.value) and repr(right) in str(error.value)


@pytest.mark.parametrize("left,right", [
    ("abs(x)", "x"), ("abs(x)^2", "x^2"), ("abs(-1)", "1"),
    ("0*sign(x)", "0"), ("abs(x)+x", "x+abs(x)"),
])
def test_nonanalytic_symmetry_pair_is_rejected_before_simplification(two_mode_spec, left, right):
    two_mode_spec["E_C"][0][1], two_mode_spec["E_C"][1][0] = left, right
    with pytest.raises(ValueError, match="not real-analytic"):
        cq.load_spec(two_mode_spec)


@pytest.mark.parametrize("left,right", [
    ("abs(x)", "abs(x)"), ("(abs(x))", "abs( x )"),
    ("sign(x) .* x", "sign(x) * x"), ("g/8", "0.125*g"),
    ("2*x", "x+x"), ("(x+1)^2", "x^2+2*x+1"),
    ("sin(x)^2+cos(x)^2", "1"), ("sqrt(x)", "x^0.5"),
])
def test_symmetric_functions_load(two_mode_spec, left, right):
    two_mode_spec["E_C"][0][1], two_mode_spec["E_C"][1][0] = left, right
    cq.load_spec(two_mode_spec)


@pytest.mark.parametrize("left,right", [
    ("sqrt(-a)", "sqrt(-b)"), ("sqrt(-x)^2", "-x"),
    ("log(-x)-log(-x)", "0"), ("exp(1000*x)", "exp(1001*x)"),
])
def test_symmetry_function_domain_failures_are_rejected(two_mode_spec, left, right):
    two_mode_spec["E_L"][0][1], two_mode_spec["E_L"][1][0] = left, right
    with pytest.raises(ValueError, match="cannot be evaluated"):
        cq.load_spec(two_mode_spec)


def test_export_rejects_nonsymmetric_matrix():
    circuit = fluxonium_pair()
    res = cq.quantize(circuit, cq.Loops.from_circuit(circuit, zero_capacitive=True))
    ec = sym.ImmutableMatrix([[1, 2], [3, 1]])
    with pytest.raises(ValueError, match="E_C.*symmetric"):
        cq.to_spec(replace(res, ham=replace(res.ham, E_C=ec)))


@pytest.mark.parametrize("alias,canonical", [
    ("µ", "μ"), ("ɛ", "ε"), ("ℏ", "ħ"), ("K", "K"),
    ("Ω", "Ω"), ("Å", "Å"), ("E_µ", "E_μ"),
])
def test_julia_identifier_aliases_are_rejected(spec, alias, canonical):
    spec["parameters"] += [alias, canonical]
    with pytest.raises(ValueError, match="canonical") as error:
        cq.load_spec(spec)
    assert canonical in str(error.value)
    res = fluxonium()
    res = replace(res, lag=replace(res.lag, EJ=(sym.Symbol(alias, real=True),)))
    with pytest.raises(ValueError, match="canonical"):
        cq.to_spec(res)


def test_alias_in_expression_is_rejected(spec):
    spec["parameters"].append("μ")
    spec["constant"] = "µ"
    with pytest.raises(ValueError, match="canonical"):
        cq.load_spec(spec)


def test_canonical_unicode_names_and_euler_constant_remain_distinct(spec):
    # Julia uses NFC plus a small custom map, not blanket NFKC folding.
    # Mathematical mu remains distinct; Euler's ℯ must not become reserved e.
    spec["parameters"] += ["μ", "ε", "ħ", "K", "Ω", "Å", "𝜇"]
    spec["constant"] = "μ + ε + ħ + K + Ω + Å + 𝜇 + ℯ"
    loaded = cq.load_spec(spec)
    assert len(loaded.constant.free_symbols) == 7
    assert loaded.constant.has(sym.E)
    res = fluxonium()
    res = replace(res, lag=replace(res.lag, EJ=(loaded.constant,)))
    assert cq.load_spec(cq.to_spec(res)).cosines[0][0] == loaded.constant


def test_large_integer_literals_remain_exact(spec):
    value = 44173801000000000000000000000000
    spec["constant"] = f"{value} + 1"
    assert cq.load_spec(spec).constant == value + 1
    res = fluxonium()
    res = replace(res, lag=replace(res.lag, EJ=(sym.Integer(value),)))
    assert cq.load_spec(cq.to_spec(res)).cosines[0][0] == value

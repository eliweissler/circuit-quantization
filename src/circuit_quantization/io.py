"""Save and load symbolic Hamiltonians in the version 1 kronfree-spec format.

The JSON contains Hamiltonian coefficients in GHz, not a serialized Result.
Loading constructs SymPy expressions without running quantization or evaluating
Python/Julia source code; parameter symbols have real=True assumptions.
See docs/spec-format.md for the interchange contract and unsupported features.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import math
from os import PathLike
from pathlib import Path
import random
import re
import unicodedata

import sympy as sym
from sympy.printing.julia import JuliaCodePrinter

from .units import subs_constants


_FUNCTIONS = {
    "sqrt": sym.sqrt, "exp": sym.exp, "cos": sym.cos, "sin": sym.sin,
    "tan": sym.tan, "cosh": sym.cosh, "sinh": sym.sinh, "tanh": sym.tanh,
    "log": sym.log, "abs": sym.Abs, "sign": sym.sign, "conj": sym.conjugate,
    "real": sym.re, "imag": sym.im,
}
_CONSTANTS = {"pi": sym.pi, "π": sym.pi, "ℯ": sym.E}
_KEYWORDS = set("baremodule begin break catch const continue do else elseif end export false finally for function global if import let local macro module quote return struct true try using while where abstract primitive mutable type in isa outer public".split())
_RESERVED = set(_FUNCTIONS) | set(_CONSTANTS) | _KEYWORDS | {"im", "e", "cis", "cispi", "Inf", "NaN"}
_IDENTIFIER = re.compile(r"(?:[^\W\d]|_)\w*", re.UNICODE)
_TOKEN = re.compile(
    r"(?P<number>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"|(?P<name>(?:[^\W\d]|_)\w*)|(?P<op>\.[+*/^\-]|//|[+*/^(),\-])",
    re.UNICODE,
)
_PRECEDENCE = {"+": 10, "-": 10, "*": 20, "/": 20, "//": 25, "^": 40}
# Julia 1.13 Base.Unicode._julia_charmap, applied before NFC (not NFKC).
# Only the letter mappings can occur in our portable identifier subset.
_JULIA_CHARMAP = str.maketrans({"ɛ": "ε", "µ": "μ", "·": "⋅", "·": "⋅", "−": "-", "ℏ": "ħ"})


class _SpecPrinter(JuliaCodePrinter):
    def _print_Float(self, expr):
        # Default SymPy printing uses 15 decimal digits for a binary64 Float,
        # insufficient to recover all 53 bits; retain two guard digits.
        return str(expr.evalf(math.ceil(expr._prec * math.log10(2)) + 2))

    def _print_Exp1(self, expr):
        # SymPy's default e spelling is reserved but not accepted by Julia's
        # coefficient loader; its supported constant spelling is ℯ.
        return "ℯ"


def _fail(field, message):
    raise ValueError(f"spec: {field}: {message}")


def _name(name, field):
    # Use a portable subset of Julia identifiers, including Greek letters.
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) or name in _RESERVED or not name.strip("_"):
        _fail(field, f"invalid or reserved parameter name {name!r}")
    _canonical_name(name, field)


def _canonical_name(name, field):
    canonical = unicodedata.normalize("NFC", name.translate(_JULIA_CHARMAP))
    if name != canonical:
        _fail(field, f"identifier {name!r} is not canonical in Julia; use {canonical!r}")


def _tokens(source, field):
    tokens = []
    pos = 0
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue
        match = _TOKEN.match(source, pos)
        if match is None:
            _fail(field, f"unsupported expression syntax near {source[pos:pos + 24]!r}")
        if match.lastgroup == "name":
            _canonical_name(match.group(), field)
        tokens.append((match.lastgroup, match.group()))
        pos = match.end()
    return tokens


def _real_expression(expr, field):
    if expr.has(sym.I, sym.oo, -sym.oo, sym.zoo, sym.nan) or expr.is_real is False:
        _fail(field, "expression must be real and finite")
    return expr


class _ExpressionParser:
    """Small scalar Julia grammar; no eval, sympify(string), or parse_expr.

    Julia's // binds more tightly than * and /; ^ is right associative and
    binds more tightly than unary signs. Dotted arithmetic is scalar here.
    """

    def __init__(self, source, symbols, field):
        self.tokens = _tokens(source, field) + [("end", "")]
        self.symbols = symbols
        self.field = field
        self.index = 0
        self.syntax = None

    def peek(self):
        kind, token = self.tokens[self.index]
        return token[1:] if kind == "op" and token.startswith(".") else token

    def take(self, expected):
        if self.peek() != expected:
            _fail(self.field, f"expected {expected!r}, got {self.peek()!r}")
        self.index += 1

    def expression(self, minimum=0):
        kind, _ = self.tokens[self.index]
        token = self.peek()
        self.index += 1
        if token in ("+", "-"):
            value = self.expression(30)
            syntax = (token, self.syntax)
            if token == "-":
                value = -value
        elif token == "(":
            value = self.expression()
            syntax = self.syntax
            self.take(")")
        elif kind == "number":
            # Preserve all printed decimal digits (and at least binary64 precision).
            if any(c in token for c in ".eE"):
                mantissa = re.split("[eE]", token)[0]
                digits = len(re.sub(r"[^0-9]", "", mantissa))
                value = sym.Float(token, max(15, digits))
            else:
                value = sym.Integer(token)
            syntax = ("number", value)
        elif kind == "name":
            if self.peek() == "(":
                if token not in _FUNCTIONS:
                    _fail(self.field, f"function {token!r} is not allowed")
                self.take("(")
                args = [self.expression()]
                arg_syntax = [self.syntax]
                while self.peek() == ",":
                    self.take(",")
                    args.append(self.expression())
                    arg_syntax.append(self.syntax)
                self.take(")")
                syntax = (token, *arg_syntax)
                if token == "log" and len(args) == 2:
                    value = sym.log(args[1], args[0])  # Julia log(base, x).
                elif len(args) == 1:
                    value = _FUNCTIONS[token](args[0])
                else:
                    _fail(self.field, f"wrong number of arguments for {token}")
            elif token in _CONSTANTS:
                value = _CONSTANTS[token]
                syntax = ("name", token)
            elif token in self.symbols:
                value = self.symbols[token]
                syntax = ("name", token)
            else:
                _fail(self.field, f"undeclared or forbidden symbol {token!r}")
        else:
            _fail(self.field, f"expected a number, parameter, or function; got {token!r}")
        value = _real_expression(value, self.field)
        while self.peek() in _PRECEDENCE and _PRECEDENCE[self.peek()] >= minimum:
            operator = self.peek()
            self.index += 1
            right = self.expression(_PRECEDENCE[operator] + (operator != "^"))
            syntax = (operator, syntax, self.syntax)
            if operator == "+":
                value += right
            elif operator == "-":
                value -= right
            elif operator == "*":
                value *= right
            elif operator in ("/", "//"):
                if operator == "//" and (value.is_Rational is not True or right.is_Rational is not True):
                    _fail(self.field, "// requires rational numeric operands; use / for symbolic division")
                value /= right
            else:
                value **= right
            value = _real_expression(value, self.field)
        self.syntax = syntax
        return value


def _expression(source, symbols, field, *, with_syntax=False):
    if not isinstance(source, str) or not source.strip():
        _fail(field, "expected a nonempty expression string")
    try:
        parser = _ExpressionParser(source, symbols, field)
        result = parser.expression()
        parser.take("")
        return (result, parser.syntax) if with_syntax else result
    except (TypeError, ZeroDivisionError, OverflowError, RecursionError) as error:
        _fail(field, f"invalid expression ({type(error).__name__})")


def _check_symmetric(rows, syntax, symbols, field):
    """Mirror Julia's syntactic gate and three-point relative comparison.

    Inspect and evaluate the source, not simplified SymPy expressions: abs(x)^2
    simplifies to x^2 and sqrt(-x)^2 to -x, hiding nonanalyticity/domain failures.
    All allowed names are real and complex constructs are already forbidden,
    so abs/sign are the only possible nonanalytic calls in this grammar;
    real sqrt/log/power domain failures are caught during sampled evaluation.
    """
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            left, right = rows[i][j], rows[j][i]
            if left == right or syntax[i][j] == syntax[j][i]:
                continue
            pair = f"{field}[{i},{j}] = {left!r} and {field}[{j},{i}] = {right!r}"
            tokens = _tokens(left, field) + _tokens(right, field)
            if any(kind == "name" and name in ("abs", "sign") for kind, name in tokens):
                _fail(field, f"must be symmetric, but {pair} are not identical and not real-analytic")
            deps = sorted({name for kind, name in tokens if kind == "name" and name in symbols})
            # Same seed, interval, count and tolerance as Julia _test_points;
            # Python's RNG stream is independent of Julia's implementation.
            rng = random.Random(0x6B66)
            for _ in range(3):
                point = {name: sym.Float(0.5 + 1.5 * rng.random()) for name in deps}
                try:
                    a = float(_expression(left, point, field))
                    b = float(_expression(right, point, field))
                    if not (math.isfinite(a) and math.isfinite(b)):
                        raise ValueError("nonfinite value")
                except (ValueError, TypeError, OverflowError) as error:
                    _fail(field, f"must be symmetric, but {pair} cannot be evaluated at a test point ({error})")
                if abs(a - b) > 1e-12 * max(abs(a), abs(b)):
                    _fail(field, f"must be symmetric, but {pair} differ")


def _object(value, required, optional, field):
    if not isinstance(value, dict):
        _fail(field, "expected an object")
    for key in value:
        if not isinstance(key, str) or (key not in required | optional and not key.startswith("_")):
            _fail(field, f"unknown key {key!r}")
    for key in sorted(required - value.keys()):
        _fail(field, f"missing required key {key!r}")


def _array(value, size, field):
    if not isinstance(value, list) or (size is not None and len(value) != size):
        _fail(field, "expected an array" + (f" of length {size}" if size is not None else ""))
    return value


@dataclass(frozen=True)
class HamiltonianSpec:
    """Immutable Hamiltonian coefficients, with real parameter symbols.

    :param parameters: Declared symbols in file order, each with real=True.
    :param modes: Pairs of descriptive mode name and type (c, e, h).
    :param E_C: Charging-energy matrix in GHz.
    :param E_L: Inductive-energy matrix in GHz.
    :param flux_linear: Column of linear phase coefficients in GHz.
    :param n_g: Column of offset charges in Cooper-pair units.
    :param cosines: Tuples (energy, numeric omega tuple, phase offset).
    :param constant: Additive energy in GHz.
    """

    parameters: tuple[sym.Symbol, ...]
    modes: tuple[tuple[str, str], ...]
    E_C: sym.ImmutableMatrix
    E_L: sym.ImmutableMatrix
    flux_linear: sym.ImmutableMatrix
    n_g: sym.ImmutableMatrix
    cosines: tuple[tuple[sym.Expr, tuple[sym.Expr, ...], sym.Expr], ...]
    constant: sym.Expr

    def hamiltonian(self) -> sym.Expr:
        """Rebuild H/h in GHz without quantizing a circuit.

        :returns: SymPy expression in real varphi_1...varphi_N and n_1...n_N,
            indexed by mode order; mode names are descriptive labels.
        """
        size = len(self.modes)
        phase = sym.ImmutableMatrix(sym.symbols(f"varphi_1:{size + 1}", real=True))
        charge = sym.ImmutableMatrix(sym.symbols(f"n_1:{size + 1}", real=True)) - self.n_g
        result = 4 * (charge.T * self.E_C * charge)[0]
        result += (phase.T * self.E_L * phase)[0] / 2 + (phase.T * self.flux_linear)[0]
        result -= sum(energy * sym.cos(sum(w * p for w, p in zip(omega, phase)) + offset)
                      for energy, omega, offset in self.cosines)
        return result + self.constant


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate key {key!r}")
        result[key] = value
    return result


def load_spec(source) -> HamiltonianSpec:
    """Read and validate a Hamiltonian dictionary, JSON string, or file path.

    :param source: JSON-ready dict, JSON text, or str/PathLike file path.
        Strings beginning with a JSON value delimiter are treated as JSON;
        use pathlib.Path for a filename that would otherwise be ambiguous.
    :returns: Immutable HamiltonianSpec, independent of the input dictionary.
    :raises ValueError: Invalid format, expression, or unsupported feature.
    :raises OSError: A source file cannot be read.
    """
    if isinstance(source, PathLike):
        source = Path(source).read_text(encoding="utf-8")
    elif isinstance(source, str) and source.strip() and source.lstrip()[0] not in '{["-0123456789' and source.strip() not in ("null", "true", "false"):
        source = Path(source).read_text(encoding="utf-8")
    if isinstance(source, str):
        try:
            source = json.loads(source, object_pairs_hook=_unique_object,
                                parse_constant=lambda value: _fail("JSON", f"nonfinite number {value}"))
        except json.JSONDecodeError as error:
            _fail("JSON", str(error))
    required = {"format", "version", "parameters", "modes", "E_C", "E_L", "flux_linear", "n_g", "cosines"}
    _object(source, required, {"units", "constant", "charge_cosines"}, "document")
    if source["format"] != "kronfree-spec":
        _fail("format", 'expected "kronfree-spec"')
    if type(source["version"]) is not int or source["version"] != 1:
        _fail("version", "expected integer 1")
    if source.get("units", "GHz") != "GHz":
        _fail("units", 'expected "GHz"')
    if _array(source.get("charge_cosines", []), None, "charge_cosines"):
        _fail("charge_cosines", "phase-slip terms are not supported")
    modes = []
    for i, mode in enumerate(_array(source["modes"], None, "modes")):
        field = f"modes[{i}]"
        _object(mode, {"name", "type"}, set(), field)
        if not isinstance(mode["name"], str) or not mode["name"]:
            _fail(field, "name must be a nonempty string")
        if mode["type"] not in ("c", "e", "h"):
            _fail(field, "type must be c, e, or h")
        modes.append((mode["name"], mode["type"]))
    size = len(modes)
    if not size or len({name for name, _ in modes}) != size:
        _fail("modes", "mode names must be unique and modes must be nonempty")
    coordinates = {f"{base}_{i}" for base in ("varphi", "n") for i in range(1, size + 1)}
    symbols = {}
    for name in _array(source["parameters"], None, "parameters"):
        _name(name, "parameters")
        if name in symbols or name in coordinates:
            _fail("parameters", f"duplicate or coordinate parameter name {name!r}")
        symbols[name] = sym.Symbol(name, real=True)

    def matrix(key):
        rows = _array(source[key], size, key)
        parsed = [[
            _expression(value, symbols, f"{key}[{i},{j}]", with_syntax=True)
            for j, value in enumerate(_array(row, size, f"{key}[{i}]"))
        ] for i, row in enumerate(rows)]
        _check_symmetric(rows, [[syntax for _, syntax in row] for row in parsed], symbols, key)
        return sym.ImmutableMatrix([[value for value, _ in row] for row in parsed])

    def vector(key):
        return sym.ImmutableMatrix(size, 1, [
            _expression(value, symbols, f"{key}[{i}]")
            for i, value in enumerate(_array(source[key], size, key))
        ])

    ec, el, flux, ng = matrix("E_C"), matrix("E_L"), vector("flux_linear"), vector("n_g")
    cosines = []
    for i, cosine in enumerate(_array(source["cosines"], None, "cosines")):
        field = f"cosines[{i}]"
        _object(cosine, {"energy", "omega", "offset"}, set(), field)
        omega = []
        for j, value in enumerate(_array(cosine["omega"], size, field + ".omega")):
            if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)):
                _fail(field + ".omega", f"mode {modes[j][0]} requires a finite number")
            if modes[j][1] == "c" and value != int(value):
                _fail(field + ".omega", f"compact mode {modes[j][0]} requires an integer")
            omega.append(sym.Integer(value) if value == int(value) else sym.Float(value))
        cosines.append((_expression(cosine["energy"], symbols, field + ".energy"), tuple(omega),
                        _expression(cosine["offset"], symbols, field + ".offset")))
    return HamiltonianSpec(tuple(symbols.values()), tuple(modes), ec, el, flux, ng, tuple(cosines),
                           _expression(source.get("constant", "0"), symbols, "constant"))


def to_spec(res, *, subs=None, simplify=True, mode_names=None) -> dict:
    """Convert a quantization Result into a JSON-ready Hamiltonian dictionary.

    :param res: Quantization Result; only its retained dynamical modes are saved.
    :param subs: Optional SymPy substitution mapping, applied before exact SI
        physical constants are evaluated; does not mutate res.
    :param bool simplify: Combine rational denominators and cancel before printing
        (default True).
    :param mode_names: Optional sequence of unique, nonempty labels, in retained
        mode order. Defaults to varphi_1...varphi_N; does not rename coordinates
        or compact-sector parameters (use subs for parameter renaming).
    :returns: Version 1 kronfree-spec dictionary with coefficients in GHz.
    :raises ValueError: Unsupported expression, parameter name, or cosine vector.
    """
    size = len(res.coords.types)
    if mode_names is None:
        mode_names = [f"varphi_{i + 1}" for i in range(size)]
    if isinstance(mode_names, (str, bytes)) or not isinstance(mode_names, Sequence):
        _fail("mode_names", "expected a sequence of labels")
    if len(mode_names) != size or any(not isinstance(name, str) or not name for name in mode_names):
        _fail("mode_names", f"expected {size} nonempty string labels")
    if len(set(mode_names)) != size:
        _fail("mode_names", "labels must be unique")
    symbols = {}

    def prepare(expr):
        expr = sym.sympify(expr)
        if subs is not None:
            expr = expr.subs(subs)
        expr = subs_constants(expr)
        # Combine nested denominators before polynomial expansion: cancel alone
        # can take minutes on substituted coupled-circuit charging energies.
        return sym.cancel(sym.together(expr)) if simplify else expr

    def coefficient(expr, field):
        expr = _real_expression(prepare(expr), field)
        by_name = {}
        for symbol in expr.free_symbols:
            _name(symbol.name, field)
            if symbol.is_real is False:
                _fail(field, f"parameter {symbol.name!r} must be real")
            previous = by_name.get(symbol.name, symbols.get(symbol.name))
            if previous is not None and previous != symbol:
                _fail(field, f"distinct symbols share the name {symbol.name!r}")
            by_name[symbol.name] = symbol
        try:
            printed = _SpecPrinter().doprint(expr)
        except (TypeError, ValueError, NotImplementedError) as error:
            _fail(field, f"cannot print expression: {error}")
        for kind, token in _tokens(printed, field):
            if kind == "name" and token in by_name:
                symbols.setdefault(token, by_name[token])
        _expression(printed, {name: sym.Symbol(name, real=True) for name in symbols}, field)
        return printed

    def matrix(values, field):
        return [[coefficient(values[i, j], f"{field}[{i},{j}]") for j in range(values.cols)] for i in range(values.rows)]

    def vector(values, field):
        return [coefficient(value, f"{field}[{i}]") for i, value in enumerate(values)]

    modes = [{"name": name, "type": typ} for name, typ in zip(mode_names, res.coords.types)]
    data = {"format": "kronfree-spec", "version": 1, "units": "GHz", "parameters": [], "modes": modes}
    data["E_C"], data["E_L"] = matrix(res.ham.E_C, "E_C"), matrix(res.ham.E_L, "E_L")
    linear = res.coords.omega_L_tilde * res.ham.E_Lb * res.ham.varphi_b[list(res.lag.L), :]
    data["flux_linear"], data["n_g"] = vector(linear, "flux_linear"), vector(res.ham.n_g, "n_g")
    offsets = res.ham.varphi_b[list(res.lag.J), :]
    cosines = []
    for k, branch in enumerate(res.lag.J):
        field = f"junction {res.circuit.branch_names[branch]}"
        omega = []
        for j, mode in enumerate(modes):
            value = prepare(res.coords.omega_J_tilde[j, k])
            where = f"{field}, mode {mode['name']}"
            if value.free_symbols or value.is_number is not True or value.is_real is not True or value.is_finite is not True:
                _fail(where, "omega must be a finite real number without free symbols")
            integer = value.is_integer is True or (isinstance(value, sym.Float) and (value % 1).is_zero)
            if mode["type"] == "c" and not integer:
                _fail(where, f"omega {value} on a compact mode must be an integer")
            number = int(value) if integer else float(value)
            if isinstance(number, float) and (not math.isfinite(number) or (number == 0 and value != 0)):
                _fail(where, "omega cannot be represented as a finite nonzero JSON number")
            omega.append(number)
        cosines.append({"energy": coefficient(res.lag.EJ[k], field + ".energy"), "omega": omega,
                        "offset": coefficient(offsets[k], field + ".offset")})
    data["cosines"], data["constant"], data["parameters"] = cosines, "0", list(symbols)
    try:
        package_version = version("circuit-quantization")
    except PackageNotFoundError:
        package_version = "unknown"
    data["_generator"] = {"package": "circuit-quantization", "version": package_version,
                          "branches": list(res.circuit.branch_names)}
    load_spec(data)  # Share the importer validation before writing any file.
    return data


def save_spec(res, path, **kwargs) -> dict:
    """Write a quantization Result as UTF-8 JSON and return the dictionary.

    :param res: Quantization Result to export.
    :param path: Destination str or PathLike, overwritten only after validation.
    :param kwargs: Forwarded to to_spec (subs, simplify, and mode_names).
    :returns: The JSON-ready dictionary written to path.
    """
    data = to_spec(res, **kwargs)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return data

import hashlib
from pathlib import Path
import runpy

import sympy as sp

import circuit_quantization as cq


VENDOR = (
    Path(__file__).parents[1]
    / "src"
    / "circuit_quantization"
    / "optimization"
    / "_vendor"
)
SOURCE_HASHES = {
    "quantize.py": "789e3280adc6aee91e748feaf932cc3c3199152b660a2a51a861b222027888f5",
    "equationset.py": "72fd3ce7b38cc0fcebfffb84977fc2b710e4a11e5c7a53def6544bc30fe0b7cf",
    "singular_interface.py": "c5ab57d229dac8b776dbc43e73c9536519d1a2ebfb7b3ceaa45a885aa024bf78",
    "z3_interface.py": "49a9389176977c4f0f86702d4c70ad5659c4a09d9001f39d0b7e3df272187730",
    "rational_check.py": "fb7c0ab24785f395082582f55d68ec45364e9f181950728ad0075f03cefaee85",
    "utils.py": "0d292d18190cd8bf73f59fe1def073829aaedfd7e72883190975d3d289ea3a50",
    "poly_solver.sing": "6501fa924e5bdc8372f65e8722564386288b4d69e6efa434c977e2a6603c0a89",
}


def _source_namespace(content: bytes) -> bytes:
    return content.replace(
        b"circuit_quantization.optimization._vendor", b"sircuitenum"
    )


def test_solver_copy_differs_only_by_internal_import_namespace():
    for filename, expected in SOURCE_HASHES.items():
        content = (VENDOR / filename).read_bytes()
        normalized = _source_namespace(content) if filename.endswith(".py") else content
        assert hashlib.sha256(normalized).hexdigest() == expected


def test_harmonium_example_uses_published_parameters_and_full_optimizer():
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / "quantize_harmonium.py")
    )
    circuit = namespace["harmonium"]()
    junctions = {branch.name: branch for branch in circuit.branches if branch.kind == "junction"}
    left = junctions["J_k_left"].josephson_energy
    right = junctions["J_k_right"].josephson_energy
    assert sp.simplify((left - right) / ((left + right) / 2)) == sp.Rational(1, 100)
    assert junctions["J_f"].josephson_energy == 50
    assert next(branch for branch in circuit.branches if branch.name == "C_sh").capacitance == sp.Rational(1500, 10**15)

    canonical = cq.quantize(circuit, coordinate_strategy="canonical")
    assert canonical.coordinates.strategy == "canonical"
    harmonic = list(canonical.coordinates.mode_indices[cq.ModeType.HARMONIC])
    assert harmonic == [3]
    assert canonical.coordinates.junction_incidence_transformed[:, harmonic] == sp.zeros(3, 1)

    result = cq.quantize(
        circuit,
        transformation=namespace["PAPER_COORDINATES"],
        mode_types=namespace["PAPER_MODE_TYPES"],
    )
    assert result.coordinates.strategy == "explicit"
    assert result.coordinates.transformation == namespace["PAPER_COORDINATES"]
    assert result.coordinates.mode_indices[cq.ModeType.COMPACT] == (0,)
    assert result.coordinates.mode_indices[cq.ModeType.EXTENDED] == (1, 2, 3)
    assert result.coordinates.mode_indices[cq.ModeType.HARMONIC] == ()
    assert result.coordinates.junction_incidence_transformed == sp.Matrix(
        [[1, -1, -1, -1], [-1, 1, -1, 1], [0, -1, 0, 0]]
    )
    assert result.coordinates.inductor_incidence_transformed == sp.Matrix(
        [[0, 0, 1, 1], [0, 0, 1, -1], [0, 1, 0, 0]]
    )
    assert namespace["KITE_REDUCED_FLUX"] in result.hamiltonian.free_symbols
    assert namespace["FLUXONIUM_REDUCED_FLUX"] in result.hamiltonian.free_symbols
    assert sp.simplify(
        circuit.loop_fluxes[0].value
        - cq.phase_to_weber(namespace["KITE_REDUCED_FLUX"])
    ) == 0
    assert sp.simplify(
        circuit.loop_fluxes[1].value
        - cq.phase_to_weber(namespace["FLUXONIUM_REDUCED_FLUX"])
    ) == 0
    half_flux = namespace["at_half_flux"](result.hamiltonian)
    assert namespace["KITE_REDUCED_FLUX"] not in half_flux.free_symbols
    assert namespace["FLUXONIUM_REDUCED_FLUX"] not in half_flux.free_symbols
    assert result.hamiltonian.has(sp.cos)

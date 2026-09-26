import sympy as sp

import circuit_quantization as cq


def test_canonical_strategy_preserves_quantization_subspaces():
    c_j, c_c, c_r, inductance = sp.symbols("C_J C_c C_r L", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch("J", 0, 1, capacitance=c_j, josephson_energy=5),
            cq.Branch("Cc", 1, 2, capacitance=c_c),
            cq.Branch("L", 0, 2, inductance=inductance),
            cq.Branch("Cr", 0, 2, capacitance=c_r),
        ),
        ground=0,
    )
    matrices = cq.assemble(circuit)
    structural = cq.choose_coordinates(matrices, strategy="structural")
    canonical = cq.choose_coordinates(matrices, strategy="canonical")
    assert canonical.strategy == "canonical"
    assert canonical.mode_indices == structural.mode_indices
    assert canonical.transformation.rank() == structural.transformation.rank()
    assert sp.Matrix.hstack(
        canonical.transformation, structural.transformation
    ).rank() == structural.transformation.rank()
    assert sp.simplify(
        canonical.transformation.T
        * matrices.capacitance_internal
        * canonical.full_transformation[:, canonical.dynamical_count :]
    ) == sp.zeros(
        canonical.dynamical_count,
        canonical.full_transformation.cols - canonical.dynamical_count,
    )


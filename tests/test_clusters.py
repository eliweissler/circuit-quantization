import pytest
import sympy as sym

from circuit_quantization import (
    cluster_cost, decouple, hierarchical_decoupling, junction_clusters, junction_inversions, lagrangian, mode_basis,
    transform,
)
from circuits import HARMONIUM_Z, KITE_Z, ZERO_PI_Z, harmonium, kite, rhombus, transmon_fluxonium, zero_pi

NOTES_BASES = {  # the notes' published coordinates, which mix a harmonic direction into the extended block
    "harmonium": (harmonium, HARMONIUM_Z, ("c", "e", "e", "e")),
    "kite": (kite, KITE_Z, ("c", "e", "e")),
    "zero_pi": (zero_pi, ZERO_PI_Z, ("e", "c", "h", "cy")),
}


def notes_coords(name):
    build, Z, types = NOTES_BASES[name]
    lag = lagrangian(build())
    return lag, decouple(lag, Z, types)


def default_coords(build):
    lag = lagrangian(build())
    return lag, decouple(lag, *mode_basis(lag))


@pytest.mark.parametrize("name", sorted(NOTES_BASES))
@pytest.mark.parametrize("types", [("c",), ("e",), ("c", "e")])
def test_every_inversion_is_a_valid_S1_and_hierarchical_cost_matches_rebuilt_coordinates(name, types):
    lag, coords = notes_coords(name)
    inversions = list(junction_inversions(coords, types))
    assert inversions
    for S in inversions:
        transform(lag, coords, S)  # raises on disallowed mixing or a non-unimodular S_cc
    S, cost = hierarchical_decoupling(coords, types)
    assert cluster_cost(transform(lag, coords, S)) == cost


def test_compact_inversion_leaves_each_aligned_junction_on_one_compact_mode():
    lag, coords = notes_coords("harmonium")
    c = [j for j, t in enumerate(coords.types) if t == "c"]
    for S in junction_inversions(coords, ("c",)):
        W = (coords.omega_J_tilde.T * S).applyfunc(sym.cancel)
        single = [k for k in range(W.rows) if [j for j in range(W.cols) if W[k, j] != 0] in ([j] for j in c)]
        assert len(single) >= len(c)


def test_extended_inversion_turns_a_junction_free_extended_direction_harmonic():
    lag, coords = notes_coords("harmonium")
    assert junction_clusters(coords) == [(0, 1, 2, 3)]
    S, cost = hierarchical_decoupling(coords, ("c", "e"))
    new = transform(lag, coords, S)
    assert new.types == ("c", "e", "e", "h")
    assert cost == (40 * 100, 100, 20)
    assert cluster_cost(new, {"c": 2, "e": 2, "h": 2}) == (4, 2, 2)


def test_junction_clusters_by_hand():
    assert junction_clusters(default_coords(transmon_fluxonium)[1]) == [(0,), (1,)]  # coupled only by C_c
    assert junction_clusters(notes_coords("zero_pi")[1]) == [(0, 1), (2,)]  # theta, phi share both junctions
    assert junction_clusters(default_coords(rhombus)[1]) == [(0, 1, 2)]  # four junctions round three islands


@pytest.mark.parametrize("build", [harmonium, kite, zero_pi, rhombus])
def test_default_bases_are_already_optimal(build):
    _, coords = default_coords(build)
    S, cost = hierarchical_decoupling(coords)
    assert cost == cluster_cost(coords)


def test_junction_inversions_rejects_unknown_types():
    _, coords = default_coords(kite)
    with pytest.raises(ValueError, match="subset"):
        next(junction_inversions(coords, ("h",)))

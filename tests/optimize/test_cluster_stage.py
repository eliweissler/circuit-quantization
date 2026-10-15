import pytest
import sympy as sym

from circuit_quantization import cluster_cost, decouple, hierarchical_decoupling, lagrangian, mode_basis, transform
from circuit_quantization.optimize import (
    SIRCUITENUM_BLOCKS, Seed, Stage, by_clusters, choose_S1, cluster_score, instance, junction_seeds,
)
from circuits import HARMONIUM_Z, KITE_Z, harmonium, kite, zero_pi_asymmetric

CLUSTERS = (Stage(rows=("wJ",), order=by_clusters()),)


def choose_clusters(lag, coords):
    return choose_S1(lag, coords, seeds=junction_seeds(), blocks=SIRCUITENUM_BLOCKS, stages=CLUSTERS,
                     score=cluster_score())


def test_by_clusters_keeps_a_junction_in_its_compact_modes_cluster_and_orders_by_cost():
    # one junction on compact modes 0, 1 and extended mode 2: 0 and 1 cannot be split
    seed = Seed({"compact": [0, 1], "extended": [2], "harmonic": []}, sym.Matrix([[1, 1, 1]]))
    patterns = by_clusters()([("wJ", 0, 2)], seed=seed)
    costs = [cost for cost, _ in patterns]
    assert costs == sorted(costs)
    assert patterns[0] == ((40 * 40, 100), (0,))  # {0, 1}, {2}: the junction leaves mode 2
    assert all(cost != (100, 40, 40) for cost in costs)  # never {0}, {1}, {2}


def test_by_clusters_needs_the_seed():
    with pytest.raises(ValueError, match="seed"):
        by_clusters()([("wJ", 0, 0)])


@pytest.mark.parametrize("build, Z, types, expected", [
    (harmonium, HARMONIUM_Z, ("c", "e", "e", "e"), (4000, 100, 20)),
    (kite, KITE_Z, ("c", "e", "e"), (4000, 20)),
])
def test_stages_split_the_notes_bases_and_instances_realize_the_cost(build, Z, types, expected):
    lag = lagrangian(build())
    coords = decouple(lag, Z, types)
    assert cluster_cost(coords) == (40 * 100 ** (len(types) - 1),)  # one cluster to start
    S1s, cost = choose_clusters(lag, coords)
    assert cost == expected
    for S in S1s:
        assert cluster_cost(transform(lag, coords, instance(lag, coords, S))) == cost


@pytest.mark.parametrize("build", [harmonium, zero_pi_asymmetric])
def test_stages_never_lose_to_the_inversions(build):
    lag = lagrangian(build())
    coords = decouple(lag, *mode_basis(lag))
    _, inversions = hierarchical_decoupling(coords)
    _, cost = choose_clusters(lag, coords)
    assert cost <= inversions

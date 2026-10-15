# Adapted from sircuitenum/quantize.py choose_Z (GPL-3.0), sircuitenum commit b2c57e8.
# The ranking body follows choose_Z; the circuit preparation is replaced by
# circuit_quantization's decoupled Coordinates (Z0 = I on the retained modes), the
# compact alignments become the default seeds, the decoupling steps become stages,
# and the z3 instance step becomes choose_instance.
"""Choose the residual transformation S_1 (Sec III.C): seeds, stages, score, then instances."""

import inspect
from dataclasses import dataclass
from functools import cached_property, partial

import numpy as np
import sympy as sym

from ..modes import _check_residual, _cluster_cost, _clusters, junction_inversions, mode_types
from ..units import Phi0, e, h, h_GHz, hbar
from .equationset import extract_denom
from .patterns import Seed
from .stages import SIRCUITENUM, SIRCUITENUM_BLOCKS, Target, resolve, trans_blocks, zero_keys
from .transforms import (
    H_hash, _find_Z_instance, _find_Z_instance_deterministic, _find_Z_min_cost,
    _incidence_square_nonzero_str, _nonzero_entries_str,
    _remove_permutation_equivalent_transformations, _sub_equal_LC, _var_col_perms,
    compact_alignment_transformation, incidence_to_square, secondary_decouple,
)

ORDER = ("c", "e", "h")  # sircuitenum's column order: compact, extended, harmonic


@dataclass(frozen=True)
class Candidate:
    """A candidate S_1 as a score sees it; matrices in (compact, extended, harmonic) order.

    instance is False while ranking symbolic candidates, whose free entries are set
    deterministically with all C and all L equal, and True for concrete instances.
    EJ holds one placeholder per junction (the junction structure is topological,
    as in sircuitenum).
    """

    S: sym.Matrix  # residual transformation of the retained modes
    C_inv: sym.Matrix  # (S^T C~ S)^{-1}
    L: sym.Matrix  # S^T L~* S
    wJ_T: sym.Matrix  # omega_J~^T S, one row per junction
    EJ: tuple
    var_types: dict
    instance: bool
    targets: tuple = ()  # every stage's matrices (stages.Target)

    @cached_property
    def zeros(self):
        """Keys (name, i, j) of every stage's terms that vanish for this S, as the solver keys them."""
        return zero_keys(self.targets, self.S, self.var_types["compact"])


def h_hash(candidate):
    """sircuitenum's H_hash, the default score; with extra_nl for instances, as choose_Z.

    Needs integer junction coefficients on compact and extended modes for instances;
    rank candidates without them, such as normal-mode seeds, with a custom score.
    """
    c = candidate
    n_nl = len(c.var_types["compact"]) + len(c.var_types["extended"])
    if c.instance and not all(sym.sympify(x).is_integer for x in c.wJ_T[:, :n_nl]):
        raise ValueError("h_hash needs integer junction coefficients on compact and extended modes; "
                         "rank this candidate with a custom score")
    return H_hash(c.C_inv, c.L, c.wJ_T, c.EJ, var_types=c.var_types, extra_nl=c.instance)[0]


def _with_seed(order, seed):
    """The stage order, given the seed when it takes one (a seed parameter, as by_clusters)."""
    return partial(order, seed=seed) if "seed" in inspect.signature(order).parameters else order


def _ordering(types):
    """P selecting columns in (compact, extended, harmonic) order, and sircuitenum's var_types."""
    columns = [j for t in ORDER for j, kind in enumerate(types) if kind == t]
    P = sym.eye(len(columns))[:, columns]
    var_types = {name: [i for i, j in enumerate(columns) if types[j] == t]
                 for name, t in (("compact", "c"), ("extended", "e"), ("harmonic", "h"))}
    var_types.update(free=[], frozen=[], sigma=[])
    return P, var_types


def _junctions(lag, coords, P):
    """Junction incidence in P's order and one placeholder E_J per junction (sircuitenum's gen_w)."""
    # One distinct E_J per junction, as choose_Z's gen_w(add_elem_number(circuit)): the
    # junction structure is topological, so equal junctions never cancel a coupling.
    EJ = list(sym.symbols(f"E_J1:{len(lag.EJ) + 1}", positive=True)) or [0]
    wJ = P.T * coords.omega_J_tilde if lag.EJ else sym.zeros(P.cols, 1)
    return wJ, EJ


def _seeded(lag, coords, S):
    """A seed or family S_1 (coords.types order), as rows (c, e, h) of coords and columns
    (c, e, h) of its own re-inferred types, with those columns' P and var_types."""
    P, _ = _ordering(coords.types)
    P_new, var_types = _ordering(mode_types(lag, coords.Z * sym.Matrix(S)))
    return P.T * sym.Matrix(S) * P_new, var_types, P_new


def candidate(lag, coords, S1, *, instance=True, targets=()):
    """The Candidate a score sees for S_1 in coords.types order, e.g. to rank instances."""
    Z, var_types, _ = _seeded(lag, coords, S1)
    P, _ = _ordering(coords.types)
    wJ, EJ = _junctions(lag, coords, P)
    C = sym.simplify(Z.T * (P.T * coords.C_tilde * P) * Z)
    return Candidate(sym.ImmutableMatrix(Z), C.inv() if C.shape[0] else C,
                     sym.simplify(Z.T * (P.T * coords.Lstar_tilde * P) * Z),
                     sym.simplify(wJ.T * Z), tuple(EJ), var_types, instance, targets)


def junction_seeds(types=("c", "e"), *, clean=True):
    """Seeds for choose_S1: the identity and every junction inversion (modes.junction_inversions).

    :param types: Blocks to invert, ("c",), ("e",), or ("c", "e").
    :param bool clean: Remove the extended components of the compact-aligned junctions.
    :returns: (lag, coords) -> list of S_1 in coords.types order.
    """
    def seeds(lag, coords):
        return [sym.eye(len(coords.types)), *junction_inversions(coords, types, clean=clean)]
    return seeds


def cluster_score(dims=None):
    """Score for choose_S1: the candidate's junction cluster dimensions, largest first (cluster_cost).

    :param dims: Basis size per mode type, as in cluster_cost.
    :returns: Candidate -> tuple, lower is better.
    """
    def score(candidate):
        types = [None] * candidate.wJ_T.cols
        for kind, t in (("compact", "c"), ("extended", "e"), ("harmonic", "h")):
            for j in candidate.var_types[kind]:
                types[j] = t
        return _cluster_cost(_clusters(candidate.wJ_T, len(types)), types, dims)
    return score


def compact_alignments(lag, coords):
    """sircuitenum's compact alignment candidates, the default seeds, as S_1 in coords.types order."""
    P, var_types = _ordering(coords.types)
    wJ, _ = _junctions(lag, coords, P)
    return [P * Z * P.T for Z in compact_alignment_transformation(sym.simplify(wJ.T), len(var_types["compact"]))]


def choose_S1(lag, coords, *, seeds=compact_alignments, blocks=SIRCUITENUM_BLOCKS, stages=SIRCUITENUM,
              score=h_hash, debug=False):
    """Residual transformations S_1 tied for the lowest score, with their free entries, and that score.

    Each seed is refined by the stages in sequence: the first opens the S_1 blocks
    in blocks as free entries and solves for those that zero its couplings
    (Singular); each later stage solves among the entries the previous one left
    free. The results are ranked by score; free entries stay symbolic, for
    choose_instance or instance. The defaults reproduce sircuitenum's choose_Z:
    compact alignments, junction couplings then inductive and charging couplings,
    ranked by H_hash. With stages=() the seeds themselves are ranked.
    Each S_1 is in coords.types column order; apply it with transform or Result.transform.

    :param lag: Node Lagrangian matrices.
    :param coords: Decoupled retained coordinates, e.g. from quantize or decouple.
    :param seeds: (lag, coords) -> iterable of S_1 in coords.types order, each a valid
        residual transformation. Defaults to compact_alignments.
    :param blocks: (old type, new type) blocks of S_1 the first stage opens, from
        stages.ALLOWED_BLOCKS. Defaults to sircuitenum's SIRCUITENUM_BLOCKS.
    :param stages: Decoupling stages run in sequence, each Stage naming built-in or custom
        matrices and the search order over their couplings; see
        circuit_quantization.optimize.stages. Defaults to SIRCUITENUM.
    :param score: Ranks candidates, lowest first: called with a Candidate, returns
        anything sortable. Defaults to h_hash, sircuitenum's ranking.
    :param bool debug: Print sircuitenum's intermediate candidates.
    :returns: Pair (list of S_1 tied for the lowest score, that score).
    """
    P, _ = _ordering(coords.types)
    cMat = P.T * coords.C_tilde * P
    lMat = P.T * coords.Lstar_tilde * P
    wJ, EJ = _junctions(lag, coords, P)  # retained-mode analogue of the node incidence wJ
    jMat = incidence_to_square(wJ, EJ) if lag.EJ else sym.zeros(P.cols, P.cols)
    n_dyn = P.cols
    builtins = {
        "J": Target("J", jMat, "flux", jMat, False) if lag.EJ else None,
        "L": Target("L", lMat, "flux", lMat, False),
        "Cinv": Target("Cinv", P.T * coords.C_tilde_inv * P, "charge", cMat, True),
        "wJ": Target("wJ", wJ.T, "rows", wJ.T, False) if lag.EJ else None,
    }
    resolved = [(stage, *resolve(stage, builtins, P)) for stage in stages]
    active = [(stage, square, rows) for stage, square, rows in resolved if square or rows]
    all_targets = tuple(t for _, square, rows in resolved for t in square + rows)
    free_blocks = trans_blocks(blocks)

    # Step 1: Seeds, each validated and re-expressed in its own column types
    seeded = []
    for S in seeds(lag, coords):
        _check_residual(coords.types, sym.Matrix(S))
        seeded.append(_seeded(lag, coords, S))
    if debug:
        print(f"Generated {len(seeded)} seeds.")
        for i, (Z, _, _) in enumerate(seeded):
            print(f"Seed {i+1}:")
            sym.pprint(Z)

    # Step 2: Minimize intermode coupling, one stage at a time; each branch of a
    # stage seeds the next (choose_Z: junction stage, then the linear stage).
    lowest_hash = ""
    lowest_Z = []  # (Z in rows c,e,h of coords and columns c,e,h of its own types, var_types, P_new)
    for Z, var_types, P_new in seeded:
        seed = Seed(var_types, (wJ.T * Z).applyfunc(sym.cancel))
        Zs_in, vals = [None], [""]
        for k, (stage, square, rows) in enumerate(active):
            last = k == len(active) - 1
            next_Z, next_vals = [], []
            for Z_in in Zs_in:
                out = secondary_decouple(var_types, [Z.transpose()*t.solver*Z for t in square],
                                         [t.invert for t in square], Z_in=Z_in, debug=debug,
                                         trans_blocks=free_blocks, names=[t.name for t in square],
                                         row_mats=[t.solver*Z for t in rows], row_names=[t.name for t in rows],
                                         pattern_order=_with_seed(stage.order, seed), return_tiebreaker=last)
                Zs, vs = out if last else (out, [""]*len(out))
                next_Z += Zs
                next_vals += vs
            Zs_in, vals = next_Z, next_vals
            if debug:
                print(f"Stage {k + 1}: {len(Zs_in)} candidate transformations.")
        for Z2, val in zip(Zs_in, vals):
            entry = (Z if Z2 is None else Z*Z2, var_types, P_new)
            if score is not h_hash:  # sircuitenum's tiebreaker prefilter belongs to its H_hash ranking
                lowest_Z.append(entry)
            elif val < lowest_hash or lowest_hash == "":
                lowest_Z = [entry]
                lowest_hash = val
            elif val == lowest_hash:
                lowest_Z += [entry]

    if debug:
        print(f"Found {len(lowest_Z)} transformations with lowest hash {lowest_hash}.")
        for i, (Z, _, _) in enumerate(lowest_Z):
            print(f"Lowest hash transformation {i+1}:")
            sym.pprint(Z)
    if not lowest_Z:
        return [], None
    # Filter out any bonkers transformations and sort by string length
    str_len = [len(str(Z)) for Z, _, _ in lowest_Z]
    order = np.argsort(str_len)
    min_len = str_len[order[0]]
    lowest_Z = [lowest_Z[i] for i in order if str_len[i] <= 50*min_len]

    # Collapse column-permutation/sign-equivalent symbolic transformations
    # before determinant instantiation, matrix inversion, and H_hash.  These
    # operations dominate runtime when compact alignment emits a large orbit.
    # Seeds that changed column types are compared only with seeds of the same types.
    kept = []
    for key in dict.fromkeys(tuple(map(tuple, vt.values())) for _, vt, _ in lowest_Z):
        group = [entry for entry in lowest_Z if tuple(map(tuple, entry[1].values())) == key]
        col_perms = list(_var_col_perms(group[0][1], dyn_only=True))
        _, unique_indices = _remove_permutation_equivalent_transformations([Z for Z, _, _ in group], perms=col_perms)
        kept += [group[index] for index in unique_indices]
    lowest_Z = kept

    # If there are multiple Z with the same lowest hash
    # then see if they separate with equal L/C values
    Z_final = []
    hash_final = tuple()
    structural_hash_cache = {}
    for Z, var_types, P_new in lowest_Z:
        Z_equal = sym.simplify(_sub_equal_LC(Z))
        var_list = [x for x in Z.free_symbols if "Z" in str(x)]
        Z_hash = _find_Z_instance_deterministic(Z_equal, var_list, nonzero=extract_denom(Z_equal))
        cTrans = sym.simplify(Z_hash.transpose()*cMat*Z_hash)[:n_dyn, :n_dyn]
        if cTrans.shape[0] > 1:
            cTransInv = sym.inv_quick(cTrans)
        else:
            cTransInv = cTrans.inv()
        lTrans = sym.simplify(Z_hash.transpose()*lMat*Z_hash)
        wJtTrans = sym.simplify(wJ.transpose()*Z_hash)
        # The non-instance H hash depends only on these three zero/nonzero
        # support patterns.  Many distinct symbolic alignments instantiate to
        # the same pattern, so avoid repeating all mode permutations for an
        # already evaluated structure.
        support_key = (
            _nonzero_entries_str(cTransInv),
            _nonzero_entries_str(lTrans),
            _incidence_square_nonzero_str(wJtTrans.transpose(), EJ),
            tuple(map(tuple, var_types.values())),
        )
        candidate_ = Candidate(Z_hash, cTransInv, lTrans, wJtTrans, tuple(EJ), var_types, False, all_targets)
        if score is not h_hash:  # the support cache holds only for the pattern-based default
            val = score(candidate_)
        else:
            if support_key not in structural_hash_cache:
                structural_hash_cache[support_key] = h_hash(candidate_)
            val = structural_hash_cache[support_key]
        if hash_final == tuple() or val < hash_final:
            hash_final = val
            Z_final = [(Z, P_new)]
        elif val == hash_final:
            Z_final.append((Z, P_new))
    if debug:
        print(f"Chosen Z transformations have H_hash = {hash_final} with {len(Z_final)} equivalent transformations.")
        for Zf, _ in Z_final:
            sym.pprint(Zf)
            sym.pprint(wJ.transpose()*Zf)

    return [sym.ImmutableMatrix(P * Zf * P_new.T) for Zf, P_new in Z_final], hash_final


def _free_entries(lag, coords, S):
    """Symbols of S that are not circuit parameters or physical constants."""
    parameters = set().union(*(sym.sympify(m).free_symbols for m in
                               (coords.C_tilde, coords.Lstar_tilde, coords.omega_J_tilde, *lag.EJ)))
    return sym.Matrix(S).free_symbols - parameters - {Phi0, e, h, h_GHz, hbar}


def choose_instance(lag, coords, S1, *, max_weight=None, debug=False):
    """sircuitenum's integer instance of one S_1 family: z3's minimum junction weight.

    The weight is the sum of |junction coefficients| on compact and extended modes,
    which must come out integer, as in sircuitenum's parameter-independent families;
    families with non-integer extended coefficients, such as normal-mode seeds, are
    not meant for this (use instance). An S_1 with no free entries is returned as is;
    entries z3 leaves free, being in no junction coefficient, are set as by instance.
    Pass the best weight so far as max_weight to skip families that cannot reach it.

    :param lag: Node Lagrangian matrices.
    :param coords: The coordinates S_1 was chosen for.
    :param S1: One S_1 family in coords.types order, e.g. from choose_S1.
    :param max_weight: Return None unless an instance of at most this weight exists.
    :returns: Pair (S_1 instance in coords.types order, its weight), or None.
    :raises ValueError: No integer instance exists.
    """
    P, _ = _ordering(coords.types)
    Zf, var_types, P_new = _seeded(lag, coords, S1)
    wJ, _ = _junctions(lag, coords, P)
    n_nl = len(var_types["compact"]) + len(var_types["extended"])

    def weight(Z):
        return sum(abs(x) for x in sym.simplify(wJ.T * Z)[:, :n_nl])

    var_list = sorted(_free_entries(lag, coords, S1), key=str)
    if not var_list:
        return sym.ImmutableMatrix(S1), weight(Zf)
    # Don't bother trying if we already know the cost can't be beat
    if max_weight is not None and _find_Z_min_cost(Zf, var_list, var_types, wJ, nonzero=extract_denom(Zf)) > max_weight:
        return None
    # Include apriori solution to speed up search
    ans = _find_Z_instance(Zf, var_list, var_types=var_types, wJ=wJ, nonzero=extract_denom(Zf),
                           apriori_sol=max_weight, debug=debug)
    # Can't beat best cost
    if ans == []:
        return None
    if ans is None:
        raise ValueError("Could not find instance of Z transformation")
    S = sym.ImmutableMatrix(P * ans * P_new.T)
    # sircuitenum fixes only entries in a junction coefficient or det S_1; set the rest,
    # e.g. a harmonic-into-extended entry, which cannot change the junction weight
    if _free_entries(lag, coords, S):
        S = instance(lag, coords, S)
    return S, weight(ans)


def instance(lag, coords, S1):
    """One valid instance of an S_1 family: its free entries set deterministically, no optimization.

    Free entries, in name order, each take the first of 0, 1, -1, 2, -2, ... that
    keeps det S_1 from vanishing identically and every entry finite. Junction
    coefficients need not come out integer.

    :param lag: Node Lagrangian matrices.
    :param coords: The coordinates S_1 was chosen for.
    :param S1: One S_1 family in coords.types order.
    :returns: S_1 with no free entries, in coords.types order.
    :raises ValueError: No small values keep S_1 invertible.
    """
    S = sym.Matrix(S1)
    for x in sorted(_free_entries(lag, coords, S), key=str):
        for k in range(41):
            value = (k + 1) // 2 * (1 if k % 2 else -1)  # 0, 1, -1, 2, -2, ...
            trial = S.subs(x, value)
            if not trial.has(sym.zoo, sym.oo, sym.nan) and sym.cancel(trial.det()) != 0:
                S = trial
                break
        else:
            raise ValueError(f"No small value of {x} keeps S_1 invertible")
    S = S.applyfunc(sym.cancel)
    _check_residual(coords.types, S)
    return sym.ImmutableMatrix(S)

# Adapted from sircuitenum/quantize.py choose_Z (GPL-3.0), sircuitenum commit b2c57e8.
# The body from "Step 1" on is verbatim; the circuit preparation is replaced by
# circuit_quantization's decoupled Coordinates, so Z0 = I on the retained modes.
"""Choose the residual transformation S_1 (Sec III.C) that minimizes H_hash, as sircuitenum's choose_Z."""

from dataclasses import dataclass

import numpy as np
import sympy as sym

from .equationset import extract_denom
from .patterns import most_zeros_first
from .transforms import (
    H_hash, _det_fast, _find_Z_instance, _find_Z_instance_deterministic, _find_Z_min_cost,
    _incidence_square_nonzero_str, _nonzero_entries_str,
    _remove_permutation_equivalent_transformations, _sub_equal_LC, _var_col_perms,
    compact_alignment_transformation, incidence_to_square, secondary_decouple,
)

ORDER = ("c", "e", "h")  # sircuitenum's column order: compact, extended, harmonic


@dataclass(frozen=True)
class Candidate:
    """A candidate S_1 as a score sees it; matrices in (compact, extended, harmonic) order.

    instance is False while ranking symbolic candidates, whose free entries are set
    deterministically with all C and all L equal, and True for the final integer
    instances. EJ holds one placeholder per junction (the junction structure is
    topological, as in sircuitenum).
    """

    S: sym.Matrix  # residual transformation of the retained modes
    C_inv: sym.Matrix  # (S^T C~ S)^{-1}
    L: sym.Matrix  # S^T L~* S
    wJ_T: sym.Matrix  # omega_J~^T S, one row per junction
    EJ: tuple
    var_types: dict
    instance: bool


def h_hash(candidate):
    """sircuitenum's H_hash, the default score; with extra_nl for instances, as choose_Z."""
    c = candidate
    return H_hash(c.C_inv, c.L, c.wJ_T, c.EJ, var_types=c.var_types, extra_nl=c.instance)[0]


def choose_S1(lag, coords, *, pattern_order=most_zeros_first, score=h_hash, return_instance=True,
              minimize_nl=True, order=("L", "C"), debug=False):
    """Residual transformations S_1 tied for the lowest sircuitenum H_hash, and that hash.

    Ports sircuitenum's choose_Z to the retained modes: enumerate compact alignments,
    solve for S_1 entries that decouple the junction, then inductive and capacitive
    couplings (Singular), rank by H_hash, and pick integer instances (z3).
    Each S_1 is in coords.types column order; apply it with transform or Result.transform.

    :param lag: Node Lagrangian matrices.
    :param coords: Decoupled retained coordinates, e.g. from quantize or decouple.
    :param pattern_order: Search order for the couplings to zero in each stage, given keys
        (name, i, j) with name "J" for the junction stage and "L"/"C" for the linear one;
        see circuit_quantization.optimize.patterns. Defaults to most_zeros_first.
    :param bool return_instance: Substitute exact rational values for free S_1 entries.
    :param bool minimize_nl: Include junction couplings when decoupling.
    :param order: Linear decoupling priority, "L" and/or "C".
    :param bool debug: Print sircuitenum's intermediate candidates.
    :param score: Ranks candidates, lowest first: called with a Candidate, returns
        anything sortable. Defaults to h_hash, sircuitenum's ranking.
    :returns: Pair (list of S_1 matrices tied for the lowest score, that score).
    """
    # Retained columns in (compact, extended, harmonic) order: Z_re = coords.Z P.
    columns = [j for t in ORDER for j, kind in enumerate(coords.types) if kind == t]
    P = sym.eye(len(columns))[:, columns]
    var_types = {name: [i for i, j in enumerate(columns) if coords.types[j] == t]
                 for name, t in (("compact", "c"), ("extended", "e"), ("harmonic", "h"))}
    var_types.update(free=[], frozen=[], sigma=[])
    cMat = P.T * coords.C_tilde * P
    lMat = P.T * coords.Lstar_tilde * P
    wJ = P.T * coords.omega_J_tilde  # retained-mode analogue of the node incidence wJ
    # One distinct E_J per junction, as choose_Z's gen_w(add_elem_number(circuit)): the
    # junction structure is topological, so equal junctions never cancel a coupling.
    EJ = list(sym.symbols(f"E_J1:{len(lag.EJ) + 1}", positive=True)) or [0]
    if not lag.EJ:
        wJ = sym.zeros(len(columns), 1)
    elem_counts = {"J": len(lag.EJ)}
    jMat = incidence_to_square(wJ, EJ) if elem_counts["J"] > 0 else sym.zeros(len(columns), len(columns))
    Z0 = sym.eye(len(columns))
    n_dyn = len(columns)

    # Step 1: Enumerate different choices of compact variable
    wJT_trans = sym.simplify(wJ.transpose()*Z0)
    Z1 = [Z0*Z for Z in compact_alignment_transformation(wJT_trans, len(var_types["compact"]))]
    if debug:
        print(f"Generated {len(Z1)} possible compact variable alignments.")
        for i, Z in enumerate(Z1):
            print(f"Compact alignment {i+1}:")
            sym.pprint(Z)

    # Step 2: Minimize intermode coupling
    lowest_hash = ""
    lowest_Z = []
    for Z in Z1:
        if sym.simplify(_det_fast(Z)) == 0:
            raise ValueError("Invalid Transformation (Not Invertible)")
        # 2a: Decouple nonlinear degrees of freedom
        if elem_counts["J"] > 0 and minimize_nl:
            Z2_a = secondary_decouple(var_types,[Z.transpose()*jMat*Z], [False], debug=debug,
                                      names=["J"], pattern_order=pattern_order)
        else:
            Z2_a = [None]

        # 2b: Decouple linear degrees of freedom
        for Z_in in Z2_a:
            lin_coupling_terms = []
            to_invert = []
            for elem in order:
                if elem not in ["L", "C"]:
                    raise ValueError("Order must be a list containing 'L' and 'C' in some order")
                elif elem == "L":
                    lin_coupling_terms.append(Z.transpose()*lMat*Z)
                    to_invert.append(False)
                elif elem == "C":
                    lin_coupling_terms.append(Z.transpose()*cMat*Z)
                    to_invert.append(True)
            if debug:
                print("Decoupling linear coupling terms with secondary transformation...")
            Z2_b, vals = secondary_decouple(var_types, lin_coupling_terms,
                                            to_invert, Z_in=Z_in,
                                            return_tiebreaker=True, debug=debug,
                                            names=list(order), pattern_order=pattern_order)
            if debug:
                print(f"Found {len(Z2_b)} possible secondary transformations for linear decoupling.")
                for i, Z2 in enumerate(Z2_b):
                    print(f"Secondary transformation {i+1} with tiebreaker value {vals[i]}:")
                    sym.pprint(Z2)
            for Z2_bi, val in zip(Z2_b, vals):
                Z_tot = Z*Z2_bi
                if val < lowest_hash or lowest_hash == "":
                    lowest_Z = [Z_tot]
                    lowest_hash = val
                elif val == lowest_hash:
                    lowest_Z += [Z_tot]
    
    if debug:
        print(f"Found {len(lowest_Z)} transformations with lowest hash {lowest_hash}.")
        for i, Z in enumerate(lowest_Z):
            print(f"Lowest hash transformation {i+1}:")
            sym.pprint(Z)
    # Filter out any bonkers transformations and sort by string length
    str_len = [len(str(Z)) for Z in lowest_Z]
    order = np.argsort(str_len)
    min_len = str_len[order[0]]
    lowest_Z = [lowest_Z[i] for i in order if str_len[i] <= 50*min_len]

    # Collapse column-permutation/sign-equivalent symbolic transformations
    # before determinant instantiation, matrix inversion, and H_hash.  These
    # operations dominate runtime when compact alignment emits a large orbit.
    col_perms = [
        permutation + tuple(range(n_dyn, Z0.shape[1]))
        for permutation in _var_col_perms(var_types, dyn_only=True)
    ]
    _, unique_indices = _remove_permutation_equivalent_transformations(
        lowest_Z, perms=col_perms
    )
    lowest_Z = [lowest_Z[index] for index in unique_indices]

    # If there are multiple Z with the same lowest hash
    # then see if they separate with equal L/C values
    Z_final = []
    hash_final = tuple()
    structural_hash_cache = {}
    for Z in lowest_Z:
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
        )
        candidate = Candidate(Z_hash, cTransInv, lTrans, wJtTrans, tuple(EJ), var_types, instance=False)
        if score is not h_hash:  # the support cache holds only for the pattern-based default
            val = score(candidate)
        else:
            if support_key not in structural_hash_cache:
                structural_hash_cache[support_key] = h_hash(candidate)
            val = structural_hash_cache[support_key]
        if hash_final == tuple() or val < hash_final:
            hash_final = val
            Z_final = [Z]
        elif val == hash_final:
            Z_final.append(Z)
    if debug:
        print(f"Chosen Z transformations have H_hash = {hash_final} with {len(Z_final)} equivalent transformations.")
        for Zf in Z_final:
            sym.pprint(Zf)
            sym.pprint(wJ.transpose()*Zf)

    # Get a specific instance of the transformation
    if return_instance:
        # Nonzero terms -- det is already done in find_Z_instance
        # so collect all the denominators
        Z_final_instance = []
        hash_final_instance = tuple()

        min_costs = []
        for Zf in Z_final:
            var_list = [x for x in Zf.free_symbols if "Z" in str(x)]
            min_costs.append(_find_Z_min_cost(Zf, var_list, var_types, wJ, nonzero=extract_denom(Zf)))
        best_cost = None

        for Zf, min_cost in zip(Z_final, min_costs):
            if debug:
                print("Searching for instance of Z transformation...", min_cost, best_cost)
                print("Z form:")
                sym.pprint(Zf)
            # Don't bother trying if we already know the cost can't be beat
            # (valid only for h_hash, whose ranking leads with this instance cost)
            if score is h_hash and not best_cost is None:
                if best_cost < min_cost:
                    continue
            var_list = [x for x in Zf.free_symbols if "Z" in str(x)]
            ans = None
            # Include apriori solution to speed up search
            ans = _find_Z_instance(Zf, var_list, var_types=var_types,
                                    wJ=wJ, nonzero=extract_denom(Zf),
                                    apriori_sol=best_cost if score is h_hash else None,
                                    debug=debug)
            # Can't beat best cost
            if ans == []:
                continue
            if ans is None:
                raise ValueError("Could not find instance of Z transformation")
            cTrans = sym.simplify(ans.transpose()*cMat*ans)[:n_dyn, :n_dyn]
            if cTrans.shape[0] > 1:
                cTransInv = sym.inv_quick(cTrans)
            else:
                cTransInv = cTrans.inv()
            lTrans = sym.simplify(ans.transpose()*lMat*ans)
            wJtTrans = sym.simplify(wJ.transpose()*ans)

            hash_val = score(Candidate(ans, cTransInv, lTrans, wJtTrans, tuple(EJ), var_types, instance=True))
            if hash_final_instance == tuple() or hash_val < hash_final_instance:
                hash_final_instance = hash_val
                Z_final_instance = [ans]
                best_cost = hash_val[2] if score is h_hash else None
            elif hash_val == hash_final_instance:
                Z_final_instance.append(ans)

        Z_final = Z_final_instance
        hash_final = hash_final_instance

    _, idx = _remove_permutation_equivalent_transformations(Z_final, perms=col_perms)
    Z_final = [Z_final[i] for i in idx]

    if debug:
        print(f"Final chosen Z transformations have H_hash = {hash_final} with {len(Z_final)} unique transformations.")
        for Zf in Z_final:
            sym.pprint(Zf)
            sym.pprint(wJ.transpose()*Zf)

    return [_to_coords_order(Zf, P) for Zf in Z_final], hash_final


def _to_coords_order(S, P):
    """S_1 in coords.types order from S in (compact, extended, harmonic) order."""
    return sym.ImmutableMatrix(P * S * P.T)

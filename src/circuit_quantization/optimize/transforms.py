# Vendored from sircuitenum/quantize.py (GPL-3.0), sircuitenum commit b2c57e8.
# Only the transformation-level helpers are kept: compact alignment, secondary
# decoupling, H_hash, and parameter instances. Circuit construction and choose_Z
# are replaced by circuit_quantization (see optimize/__init__.py).
"""Residual-freedom search: secondary decoupling and H_hash ranking (from sircuitenum)."""

import itertools
import functools
import time
from copy import deepcopy

from dataclasses import dataclass, field
from typing import Union, Sequence, Iterable, Mapping, Optional, Tuple, Any

import sympy as sym
import numpy as np
import networkx as nx

from sympy import collect, expand_mul, Mul, Dummy
from sympy.core.add import Add

from . import _utils as utils
from .equationset import maximally_compatible_sol, extract_denom, eq_indep_of_vars, _unique_products, _eq_as_numer_denom
from .z3_interface import find_rational_vars_integer_results, _calc_min_cost, _heuristic_upper_bound


PERIODIC_CHARGE = "n"
PERIODIC_PHASE = "θ"
EXTENDED_CHARGE = "q"
EXTENDED_PHASE = "φ"
NODE_CHARGE = "q"
NODE_PHASE = "ϕ"
EXT_CHARGE = "n_g"
EXT_PHASE = "_{ext}"


def _independent_from(A, B):
    """
    Returns the maximum set of vectors in A such that 
    their span has no overlap with the span of B.
    Assumes A and B are sets of linearly independent column vectors.

    Args:
        A (list[sym.Matrix] or sym.Matrix): Set of linearly independent column vectors.
        B (list[sym.Matrix] or sym.Matrix): Set of linearly independent column vectors.

    Returns:
        list[sym.Matrix]: Basis vectors in span(A) not in span(B).
    """
    # Convert to list of column vectors if needed
    if isinstance(A, sym.Matrix):
        A = [A[:, j] for j in range(A.shape[1])]
    if isinstance(B, sym.Matrix):
        B = [B[:, j] for j in range(B.shape[1])]

    # If B is empty, all of A is independent
    if not B:
        return [v for v in A]

    # Stack B and A, get linearly independent columns
    mat = sym.Matrix.hstack(*B, *A)
    li_cols = _linearly_indep_cols(mat)
    # Only keep those columns that come from A
    offset = len(B)
    result = []
    for idx in li_cols:
        if idx >= offset:
            result.append(mat[:, idx])
    return result


def _vec_space_overlap(vecs1:Union[list[sym.Matrix], sym.Matrix],
                       vecs2:Union[list[sym.Matrix], sym.Matrix],
                       support:Union[list[sym.Matrix], sym.Matrix] = [],
                       return_decomp:bool = False
                       ) -> list[sym.Matrix]:
    """
    Calculates the overlap of two linearly independent sets of
    vectors vecs1, vecs2 using NULL(vecs1, -vecs2)

    Args:
        vecs1 (Union[list[sym.Matrix], sym.Matrix]): set of column vectors
        vecs2 (Union[list[sym.Matrix], sym.Matrix]): set of column vectors
        support (Union[list[sym.Matrix], sym.Matrix]): set of vectors that can be
                                                       included to make the overlap work,
                                                       if used in addition to the vectors
                                                       from the individual sets

    Returns:
        list[sym.Matrix]: list of vectors that span the overlap space.
    """
    # Convert matrices to list
    if isinstance(vecs1, sym.Matrix):
        vecs1 = [vecs1[:, j] for j in range(vecs1.shape[1])]
    if isinstance(vecs2, sym.Matrix):
        vecs2 = [vecs2[:, j] for j in range(vecs2.shape[1])]
    if isinstance(support, sym.Matrix):
        support = [support[:, j] for j in range(support.shape[1])]

    # If either one is empty, return no overlap
    if len(vecs1) == 0 or len(vecs2) == 0:
        if return_decomp:
            if len(support) > 0:
                return [], [], [], []
            else:
                return [], [], []
        else:
            return []

    # Assert vector sets are linearly independent and same length
    assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs1))) == len(vecs1)
    assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs2))) == len(vecs2)
    assert len(_linearly_indep_cols(sym.Matrix.hstack(*support))) == len(support)
    assert vecs1[0].shape[0] == vecs2[0].shape[0]
    if len(support) > 0:
        assert support[0].shape[0] == vecs1[0].shape[0]

    # Calculate the overlap of the two vector spaces
    divide = (len(vecs1), len(support) + len(vecs1))
    ns = sym.Matrix.hstack(*[v for v in vecs1],
                           *[-v for v in support],
                           *[-v for v in vecs2]).nullspace()
    
    # Gather the entries
    in_v1 = []
    in_v2 = []
    in_sup = []
    overlap_vecs = []
    mat = sym.Matrix.hstack(*vecs1)
    for vec in ns:
        v1_entry = vec[:divide[0], :]
        sup_entry = vec[divide[0]:divide[1], :]
        v2_entry = vec[divide[1]:, :]
        if not(v1_entry.is_zero_matrix or v2_entry.is_zero_matrix):
            in_v1.append(v1_entry)
            in_v2.append(v2_entry)
            in_sup.append(sup_entry)
            overlap_vecs.append(mat*v1_entry)

    if return_decomp:
        if len(support) == 0:
            return overlap_vecs, in_v1, in_v2
        else:
            return overlap_vecs, in_v1, in_v2, in_sup
    else:
        return overlap_vecs


def _linearly_indep_cols(X):
    rref, pivot_cols = X.rref()
    return pivot_cols


def _linearly_indep_rows(X):
    return _linearly_indep_cols(X.transpose())


def _linearly_indep_col_sets(X):
    pivot_cols = _linearly_indep_cols(X)
    rank = len(pivot_cols)
    if rank == X.shape[1]:
        return [pivot_cols]

    # Every independent column set is a rank-sized basis.  The previous
    # implementation discovered these by trying all n! column permutations
    # and rerunning RREF.  Enumerating the n-choose-r candidate bases directly
    # is equivalent and avoids factorial work for junction-rich circuits.
    return [
        cols
        for cols in itertools.combinations(range(X.shape[1]), rank)
        if len(_linearly_indep_cols(X[:, cols])) == rank
    ]


def _linearly_indep_row_sets(X):
    return _linearly_indep_col_sets(X.transpose())


def _det_fast(M):
    """
    Hardcoded determinant for small sympy matrices (up to 4x4)

    Falls back to M.det() for larger
    """
    n = M.shape[0]
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    if n == 1:
        return M[0,0]

    elif n == 2:
        return M[0,0]*M[1,1] - M[0,1]*M[1,0]

    elif n == 3:
        return (M[0,0]*(M[1,1]*M[2,2] - M[1,2]*M[2,1])
              - M[0,1]*(M[1,0]*M[2,2] - M[1,2]*M[2,0])
              + M[0,2]*(M[1,0]*M[2,1] - M[1,1]*M[2,0]))
    elif n == 4:
        return (
            M[0,0]*(M[1,1]*(M[2,2]*M[3,3] - M[2,3]*M[3,2])
                   -M[1,2]*(M[2,1]*M[3,3] - M[2,3]*M[3,1])
                   +M[1,3]*(M[2,1]*M[3,2] - M[2,2]*M[3,1]))
          - M[0,1]*(M[1,0]*(M[2,2]*M[3,3] - M[2,3]*M[3,2])
                   -M[1,2]*(M[2,0]*M[3,3] - M[2,3]*M[3,0])
                   +M[1,3]*(M[2,0]*M[3,2] - M[2,2]*M[3,0]))
          + M[0,2]*(M[1,0]*(M[2,1]*M[3,3] - M[2,3]*M[3,1])
                   -M[1,1]*(M[2,0]*M[3,3] - M[2,3]*M[3,0])
                   +M[1,3]*(M[2,0]*M[3,1] - M[2,1]*M[3,0]))
          - M[0,3]*(M[1,0]*(M[2,1]*M[3,2] - M[2,2]*M[3,1])
                   -M[1,1]*(M[2,0]*M[3,2] - M[2,2]*M[3,0])
                   +M[1,2]*(M[2,0]*M[3,1] - M[2,1]*M[3,0]))
        )
    else:
        # The default Bareiss path can spend minutes expanding rational
        # expressions for the 5x5 symbolic transformations encountered during
        # H-class enumeration.  DomainMatrix Gaussian elimination preserves a
        # compact exact rational form and is dramatically faster for these
        # matrices.
        return M.det(method="domain-ge")


def _equiv_cols(c1, c2, shifts=None):
    """
    Are two transformation columns equivalent up to a change
    in sign and and given shifts?

    Args:
        c1 (sym.Matrix): column 1
        c2 (sym.Matrix): column 2
        shifts (list[sym.Matrix], optional): Shifts. Defaults to global 1.

    Returns:
        bool: True if equivalent up to sign and shifts. False otherwise.
    """
    if shifts is None:
        # Just the global shift
        shifts = [sym.ones(c1.shape[0], 1)]
    for sign in [1, -1]:
        # Can you make the difference using the specified shifts?
        diff = sign*c1 - c2
        if diff.is_zero_matrix:
            return True
        if not shifts is None:
            if len(shifts) > 0:
                li_cols = _linearly_indep_cols(sym.Matrix.hstack(diff, *shifts))
                if len(li_cols) == len(shifts):
                    return True
    return False


def _equal_up_to_column_shift_and_sign(A: sym.Matrix, B: sym.Matrix, shifts: Union[sym.Matrix,list[sym.Matrix]] = None):
    """
    Assumes that the columns in A and B are linearly independent

    Args:
        A (sym.Matrix): Transformation matrix A
        B (sym.Matrix): Transformation matrix B
        shifts (list[sym.Matrix], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if A.shape != B.shape:
        return False
    if isinstance(shifts, sym.Matrix):
        shifts = [shifts[:, i] for i in range(shifts.shape[1])]
    if shifts is None:
        # Just the global shift
        shifts = [sym.ones(A.shape[0], 1)]
    elif len(shifts) > 0:
        # Get linearly independent set of shifts
        li_shifts = _linearly_indep_cols(sym.Matrix.hstack(*shifts))
        shifts = [shifts[i] for i in li_shifts]

    rows, cols = A.shape
    for j in range(cols):
        if not _equiv_cols(A[:, j], B[:, j], shifts):
            return False
    return True


def _find_equiv_cols(c1, cList, shifts: Union[sym.Matrix, list[sym.Matrix]]=None):
    if isinstance(shifts, sym.Matrix):
        shifts = [shifts[:, i] for i in range(shifts.shape[1])]
    return [i for i in range(len(cList)) if _equal_up_to_column_shift_and_sign(c1, cList[i], shifts)]


def _find_equiv_mats(m1, mList, shifts=None):
    return [i for i in range(len(mList)) if _equal_up_to_column_swaps_and_shift_and_sign(m1, mList[i], shifts)]


def _equal_up_to_column_swaps_and_shift_and_sign(A: sym.Matrix, B: sym.Matrix, shifts: Union[sym.Matrix, list[sym.Matrix]] = None):
    """
 
    Args:
        A (sym.Matrix): _description_
        B (sym.Matrix): _description_

    Returns:
        _type_: _description_
    """

    if A.shape != B.shape:
        return False
    
    # Count unique columns and number of occurances
    colsA = []
    countsA = []
    colsB = []
    countsB = []
    for j in range(A.shape[1]):
        for mat, cols, counts in zip([A, B],
                                     [colsA, colsB],
                                     [countsA, countsB]):
            c = mat[:, j]
            equiv = _find_equiv_cols(c, cols, shifts)
            if len(equiv) == 0:
                cols.append(c)
                counts.append(1)
            else:
                counts[equiv[0]] += 1
    
    # Make sure there are equal sets of equal columns
    # Function that returns equivalent columns in an iterable
    for cA in colsA:
        # Identify equivalent columns
        eq_cols = _find_equiv_cols(cA, colsB, shifts)
        if len(eq_cols) > 1:
            raise ValueError("Columns are not unique")
        elif len(eq_cols) != 1:
            return False
    return True


def _nonzero_entries_str(X: Union[sym.Matrix, np.ndarray],
                         prepend_sum: bool = True,
                         full_mat: bool = False) -> str:
    """
    For a symmetric matrix X, returns the flattened indices
    of nonzero off-diagonal entries

    Args:
        X (Union[sym.Matrix, np.ndarray]): symmetric matrix
        prepend_sum (bool): prepends the sum of the binary string

    Returns:
        str: binary string 10101001 with 1 being the indices
             of nonzero entries and 0 being the indices of
             zero entries. optionally prepend the sum of the
             binary string 4-10101001
    """
    if isinstance(X, np.ndarray):
        X = sym.Matrix(X)
    all_vars = list(X.free_symbols)
    nz_idx = []
    n_nz = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if j > i or full_mat:
                terms = _unique_products(X[i,j].as_numer_denom()[0])
                if any(coeff != 0 for coeff in terms.values()):
                    nz_idx.append("1")
                    n_nz += 1
                else:
                    nz_idx.append("0")

    bin_str = "".join(nz_idx)
    if prepend_sum:
        bin_str = str(n_nz) + "-" + bin_str
    return bin_str 


def _incidence_square_nonzero_str(w: sym.Matrix, vals: list) -> str:
    """Return the off-diagonal support of ``w diag(vals) w.T`` cheaply.

    With distinct symbolic coefficients, terms from different incidence
    columns cannot cancel.  The support can therefore be read directly from
    pairs of nonzero incidence entries, avoiding construction and expansion of
    a symbolic weighted Gram matrix.  Repeated coefficients use the original
    exact symbolic path because cancellation is then possible.
    """
    if len(set(vals)) != len(vals):
        return _nonzero_entries_str(incidence_to_square(w, vals))

    bits = []
    count = 0
    for row in range(w.rows):
        for column in range(row + 1, w.rows):
            nonzero = any(
                sym.simplify(w[row, edge] * w[column, edge]) != 0
                for edge in range(w.cols)
            )
            bits.append("1" if nonzero else "0")
            count += int(nonzero)
    return f"{count}-" + "".join(bits)


def _remove_permutation_equivalent_transformations(Z_list, perms=[]):
    """Filter transformations that are column permutations or sign-flips of each other. Considers
    subsets of columns based on variable types. Substitutes dummy variables to allow for equality
    checks even when symbolic variables differ.

    For each transformation matrix in ``Z_list``, generates column permutations
    based on perms and checks if it's equivalent (up to column shifts
    and sign flips) to any previously kept transformation.

    Parameters
    ----------
    Z_list : list[sym.Matrix]
        List of transformation matrices to filter.
    perms : list[tuple[int]]
        List of column index permutations to consider for equivalence.

    Returns
    -------
    tuple[list[sym.Matrix], list[str]]
        Filtered lists of transformation matrices and their nonzero strings.
    """
    # If perms is empty, consider all permutations of columns
    if not perms:
        perms = list(itertools.permutations(range(Z_list[0].shape[1])))

    def normalized_permutation(Z, perm):
        Z_perm = Z[:, perm]
        substitutions = {}
        variable_count = 1
        for column in range(Z_perm.shape[1]):
            for row in range(Z_perm.shape[0]):
                for variable in sorted(Z_perm[row, column].free_symbols, key=str):
                    if variable not in substitutions:
                        substitutions[variable] = sym.Symbol(
                            f"v_{variable_count}", real=True
                        )
                        variable_count += 1
        return Z_perm.subs(substitutions)

    def orbit_fingerprint(Z):
        permutation_keys = []
        for perm in perms:
            Z_perm = normalized_permutation(Z, perm)
            column_keys = []
            for column in range(Z_perm.shape[1]):
                positive = tuple(sym.srepr(value) for value in Z_perm[:, column])
                negative = tuple(sym.srepr(-value) for value in Z_perm[:, column])
                column_keys.append(min(positive, negative))
            permutation_keys.append(tuple(column_keys))
        return min(permutation_keys)

    unique_Z = []
    idx_keep = []
    seen_fingerprints = set()
    for k, Z_k in enumerate(Z_list):
        fingerprint = orbit_fingerprint(Z_k)
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_Z.append(normalized_permutation(Z_k, perms[0]))
            idx_keep.append(k)

    return unique_Z, idx_keep


def _sort_wT(wT: Union[sym.Matrix, np.ndarray]):
    if not isinstance(wT, np.ndarray):
        wT = np.array(wT)
    keys = []
    for i in range(wT.shape[0]):
        keys.append(str(sum(wT[i, :] != 0)) +"_"+"-".join(wT[i, :].nonzero()[0].astype(str)))

    return wT[np.argsort(keys)]


def _maximize_wT(wT):
    """
    Need to sort before sending in for accurate row accounting
    Assumes all integer entries in wT

    Args:
        wT (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Make sure all integer values
    # nz_min = np.min(np.abs(wT[np.nonzero(wT)]))
    # wT = (wT/nz_min).astype(int)


    # Sort to put rows with identical nz entries
    # next to each other
    wT = _sort_wT(wT)

    # Identify rows with identical nonzero entries
    nz_entries = [tuple(wT[i].nonzero()[0]) for i in range(wT.shape[0])]
    swap_rows = []
    segment = [0]
    for i in range(1, wT.shape[0]):
        if nz_entries[i] == nz_entries[i-1]:
            segment.append(i)
        else:
            swap_rows.append(segment)
            segment = [i]
    swap_rows.append(segment)

    # For a fixed set of column signs, the row signs that maximize the total
    # sum are separable: each row must make its own signed row sum positive.
    # A zero-sum row may take either sign, so choose its lexicographically
    # larger representation. Rows with identical supports can then be sorted
    # directly. This is equivalent to enumerating all row signs and row
    # permutations, leaving only the much smaller column-sign search.
    best_w = None
    best_val = -np.inf
    best_key = None

    # Which columns to invert
    for cols in itertools.product([1, -1], repeat=wT.shape[1]):
        col_vec = np.array(cols).flatten()
        column_signed = wT * col_vec[np.newaxis, :]
        row_signs = []
        signed_rows = []
        for row in column_signed:
            row_sum = int(np.sum(row))
            if row_sum > 0:
                row_sign = 1
            elif row_sum < 0:
                row_sign = -1
            else:
                positive_key = tuple(int(value) for value in row)
                negative_key = tuple(-value for value in positive_key)
                row_sign = 1 if positive_key >= negative_key else -1
            row_signs.append(row_sign)
            signed_rows.append(row_sign * row)

        row_order = []
        for row_segment in swap_rows:
            row_order.extend(sorted(
                row_segment,
                key=lambda row_index: tuple(
                    int(value) for value in signed_rows[row_index]
                ),
                reverse=True,
            ))
        wT_perm = np.array(signed_rows)[row_order, :].astype(int)
        val = int(np.sum(wT_perm))
        key = tuple(int(value) for value in wT_perm.flatten())
        if val > best_val or (val == best_val and key > best_key):
            best_val = val
            best_key = key
            best_w = (
                wT_perm.copy(),
                np.array(row_signs).copy(),
                col_vec.copy(),
                row_order,
            )

        
    return best_w[0], best_key, best_w[1:]


def _var_col_perms(var_types,
                   dyn_modes=["compact", "extended", "harmonic"],
                   nd_modes=["free", "frozen", "sigma"],
                   dyn_only=False):
    
    mode_perms = []
    for dyn_mode in dyn_modes:
        if dyn_mode in var_types:
            mode_perms.append(tuple(itertools.permutations(var_types[dyn_mode], len(var_types[dyn_mode]))))
    
    if not dyn_only:
        for nd_mode in nd_modes:
            if nd_mode in var_types:
                mode_perms.append((tuple(var_types[nd_mode]),))

    perms = itertools.product(*mode_perms)
    return [tuple(itertools.chain.from_iterable(perm)) for perm in perms]


def _sub_equal_LC(X):
    # All C the same, different from Cj
    C = sym.symbols("C", positive=True, real=True)
    Cj = sym.symbols("C_J", positive=True, real=True)
    for x in X.free_symbols:
        if "c" in str(x).lower() and "j" not in str(x).lower():
            X = X.subs(x, C)
        if "c" in str(x).lower() and "j" in str(x).lower():
            X = X.subs(x, Cj)
    # All L the same
    L = sym.symbols("L", positive=True, real=True)
    for x in X.free_symbols:
        if "l" in str(x).lower():
            X = X.subs(x, L)
    return X


# Deterministic symbolic instantiation helper
def _find_Z_instance_deterministic(Z: sym.Matrix, var_list: list[sym.Symbol],
                                   max_tries=5, nonzero=[], return_mapping=False):
    """
    Deterministically substitute symbolic parameters in Z with
    rational values guaranteeing (if possible) a nonzero det.
    """
    


    
    ordered = sorted([v for v in Z.free_symbols if v in var_list], key=str)
    if not ordered:
        return Z
    
    # Identify a valid assignment that yields a non-singular Z
    nonzero = list(set(x for x in [sym.simplify(x) for x in nonzero + [_det_fast(Z)]] if len(x.free_symbols) > 0))
    nonzero = [x for x in nonzero if any(v in x.free_symbols for v in ordered)]
    _, res = _heuristic_upper_bound(ordered, nonzero, [], var_list, debug=False)
    all_Z = []
    for subs in res:
        all_Z.append(Z.subs(subs["variables"]))
    # Choose the transformation that maximizes the row stacked Z
    # purely arbitrarily to break ties and behave deterministically
    return max(all_Z, key=lambda Z: tuple(str(x) for x in tuple(Z)))


def _fixed_cost_plus_integer_cost(Z: Union[sym.Matrix, sym.Expr], var_list: list[sym.Expr],
                                 var_types: dict, wJ: sym.Matrix, nonzero=[]):
    n_nl = len(var_types.get("compact", [])) + len(var_types.get("extended", []))

    # Variables to substitute concrete values in for
    det = sym.simplify(_det_fast(Z), rational=True)
    if det not in nonzero:
        nonzero = nonzero + [det]

    # Identify a valid assignment that yields a non-singular Z
    # with all integer entries in wJ^T*Z
    wJT_trans = wJ.transpose()*Z
    
    # Identify nonzero entries in wJ^T*Z
    nz_entries = {}
    fixed_cost = 0
    for i in range(wJT_trans.shape[0]):
        for j in var_types.get("compact", []) + var_types.get("extended", []):
            wJval = sym.simplify(wJT_trans[i,j])
            if wJval != 0:
                if len(wJval.free_symbols) == 0:
                    if sym.core.numbers.int_valued(wJval):
                        fixed_cost += abs(int(wJval))
                    else:
                        raise ValueError("Fixed Non Integer Present")
                else:
                    nz_entries[(i,j)] = wJval
    if len(nz_entries) == 0:
        return fixed_cost, [], [], [], []
    
    # Pass to z3 to solve for integer assignments
    # while ensuring det(Z) != 0 and minimizing sum of abs values
    nz_idx = sorted(nz_entries.keys(), key=lambda x: (x[0], x[1]))
    integer_constraints = [nz_entries[(i,j)] for (i,j) in nz_idx]
    nonzero_constraints = nonzero
    var_list = set()
    for expr in integer_constraints + nonzero_constraints:
        var_list = var_list.union(expr.free_symbols)
    var_list = sorted(var_list, key=str)
    return fixed_cost, integer_constraints, nonzero_constraints, var_list, nz_idx


def _find_Z_min_cost(Z: Union[sym.Matrix, sym.Expr], var_list: list[sym.Expr],
                     var_types: dict, wJ: sym.Matrix, nonzero=[]):
    fixed_cost, integer_constraints, nonzero_constraints, var_list, _ = _fixed_cost_plus_integer_cost(Z, var_list,
                                                                                               var_types, wJ, nonzero)
    if integer_constraints == []:
        return fixed_cost
    variable_cost = _calc_min_cost(integer_constraints,nonzero_constraints,[],var_list)[0]
    return fixed_cost + variable_cost


def _complete_graph_wjt_cost_certificate(Z: sym.Matrix, wJ: sym.Matrix,
                                         n_nl: int) -> Optional[int]:
    """Return an exact L1 lower bound for a recognized complete-graph basis.

    If ``wJ`` is the signed incidence matrix of the simple complete graph K_n,
    its kernel is the one-dimensional common mode, the final columns of ``Z``
    explicitly span that kernel, and all other columns are nonlinear dynamical
    modes, then every dynamical column has integer edge differences with L1
    norm at least the edge connectivity of K_n, namely ``n - 1``.  Invertibility
    prevents any such column from lying in the common-mode kernel.  The bound
    ``(n - 1)^2`` is achievable only by singleton-cut columns; every basis made
    from those columns belongs to one canonical row/column-sign orbit.

    ``None`` is returned unless every structural precondition is established
    exactly.  This deliberately narrow certificate is preferable to applying a
    plausible graph bound to a transformation whose nullspace layout is not
    known.
    """
    if not isinstance(Z, sym.MatrixBase) or not isinstance(wJ, sym.MatrixBase):
        return None
    n_vertices = wJ.rows
    if n_vertices < 2 or Z.rows != n_vertices or Z.cols != n_vertices:
        return None
    if n_nl != n_vertices - 1:
        return None

    expected_pairs = set(itertools.combinations(range(n_vertices), 2))
    incidence_pairs = []
    for column in range(wJ.cols):
        nonzero = [row for row in range(n_vertices) if wJ[row, column] != 0]
        if len(nonzero) != 2:
            return None
        values = {sym.simplify(wJ[row, column]) for row in nonzero}
        if values != {sym.Integer(-1), sym.Integer(1)}:
            return None
        incidence_pairs.append(tuple(sorted(nonzero)))
    if len(incidence_pairs) != len(expected_pairs):
        return None
    if set(incidence_pairs) != expected_pairs:
        return None

    kernel_columns = Z[:, n_nl:]
    if kernel_columns.cols != 1 or kernel_columns.rank() != 1:
        return None
    kernel_image = wJ.transpose() * kernel_columns
    if any(sym.simplify(value) != 0 for value in kernel_image):
        return None

    return (n_vertices - 1) ** 2


def _decomposed_complete_graph_wjt_cost_certificate(
        Z: sym.Matrix, wJ: sym.Matrix, n_nl: int) -> Optional[int]:
    """Certify L1 cost for a direct sum of complete quotient-cut spaces.

    A symbolic column may force some junction-edge differences to zero.  Those
    zero edges contract vertices.  If the remaining quotient is a simple
    complete graph K_m, any nonconstant integer potential on that column costs
    at least ``m - 1``.  This routine accepts a collection of such column
    spaces only when each K_m has exactly ``m - 1`` columns and their potential
    spaces form a direct decomposition modulo the common mode.  Under those
    conditions the minimum bases are products of singleton-cut bases and form
    one canonical row/column-sign orbit.

    The recognition rules are intentionally strict; unrecognized layouts fall
    back to the general Z3 search.
    """
    complete_bound = _complete_graph_wjt_cost_certificate(Z, wJ, n_nl)
    if complete_bound is not None:
        return complete_bound

    if not isinstance(Z, sym.MatrixBase) or not isinstance(wJ, sym.MatrixBase):
        return None
    n_vertices = wJ.rows
    if (n_vertices < 2 or Z.rows != n_vertices or Z.cols != n_vertices
            or n_nl != n_vertices - 1):
        return None

    edges = []
    for column in range(wJ.cols):
        nonzero = [row for row in range(n_vertices) if wJ[row, column] != 0]
        if len(nonzero) != 2:
            return None
        values = {sym.simplify(wJ[row, column]) for row in nonzero}
        if values != {sym.Integer(-1), sym.Integer(1)}:
            return None
        edges.append(tuple(sorted(nonzero)))
    if len(set(edges)) != len(edges) or wJ.transpose().rank() != n_vertices - 1:
        return None

    kernel_columns = Z[:, n_nl:]
    if kernel_columns.cols != 1 or kernel_columns.rank() != 1:
        return None
    if any(sym.simplify(value) != 0
           for value in wJ.transpose() * kernel_columns):
        return None

    wJT = sym.simplify(wJ.transpose() * Z[:, :n_nl])
    groups = {}
    total_bound = 0
    for dynamic_column in range(n_nl):
        parent = list(range(n_vertices))

        def find(vertex):
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for edge_index, (left, right) in enumerate(edges):
            if sym.simplify(wJT[edge_index, dynamic_column]) == 0:
                union(left, right)

        roots = [find(vertex) for vertex in range(n_vertices)]
        root_order = {root: index for index, root in enumerate(dict.fromkeys(roots))}
        partition = tuple(root_order[root] for root in roots)
        n_quotient_vertices = len(root_order)
        if n_quotient_vertices < 2:
            return None

        quotient_edges = []
        for edge_index, (left, right) in enumerate(edges):
            left_part, right_part = partition[left], partition[right]
            expression = sym.simplify(wJT[edge_index, dynamic_column])
            if left_part == right_part:
                if expression != 0:
                    return None
                continue
            if expression == 0:
                return None
            quotient_edges.append(tuple(sorted((left_part, right_part))))
        expected_edges = set(itertools.combinations(range(n_quotient_vertices), 2))
        if (len(quotient_edges) != len(expected_edges)
                or set(quotient_edges) != expected_edges):
            return None

        groups.setdefault(partition, []).append(dynamic_column)
        total_bound += n_quotient_vertices - 1

    subspace_columns = [sym.ones(n_vertices, 1)]
    expected_dimension = 1
    for partition, columns in groups.items():
        n_parts = max(partition) + 1
        if len(columns) != n_parts - 1:
            return None
        expected_dimension += n_parts - 1
        for part in range(n_parts - 1):
            subspace_columns.append(sym.Matrix([
                int(vertex_part == part) for vertex_part in partition
            ]))
    if expected_dimension != n_vertices:
        return None
    if sym.Matrix.hstack(*subspace_columns).rank() != n_vertices:
        return None

    return total_bound


def _find_Z_instance(Z: Union[sym.Matrix, sym.Expr], var_list: list[sym.Expr],
                     var_types: dict, wJ: sym.Matrix, nonzero=[],
                     apriori_sol=None, debug=False):
    
    # If no variables, just return
    if len(Z.free_symbols) == 0:
        return Z
    
    n_nl = len(var_types.get("compact", [])) + len(var_types.get("extended", []))

    # Variables to substitute concrete values in for
    det = sym.simplify(_det_fast(Z), rational=True)
    if det not in nonzero:
        nonzero = nonzero + [det]

    # No nonzero entries in wJ or no nonlinear variables
    if wJ.shape[0] == 0 or n_nl == 0:
        return _find_Z_instance_deterministic(Z, var_list, max_tries=100, nonzero=nonzero)

    # Identify a valid assignment that yields a non-singular Z
    # with all integer entries in wJ^T*Z
    ext = var_types.get("extended", [])
    wJT_trans = sym.simplify(wJ.transpose()*Z)
    # Nothing to solve for
    if len(wJT_trans.free_symbols) == 0:
        return _find_Z_instance_deterministic(Z, var_list, max_tries=100, nonzero=nonzero)

    fixed_cost, integer_constraints, nonzero_constraints, var_list, nz_idx = _fixed_cost_plus_integer_cost(Z, var_list,
                                                                                var_types, wJ, nonzero)
    total_cost_certificate = _decomposed_complete_graph_wjt_cost_certificate(
        Z, wJ, n_nl
    )
    variable_cost_certificate = None
    canonical_optimum_is_unique = False
    if total_cost_certificate is not None:
        variable_cost_certificate = max(0, total_cost_certificate - fixed_cost)
        canonical_optimum_is_unique = True
    # Encode symmetry map of row swaps, column sign flips, and row sign flips
    n_terms = len(nz_idx)
    max_row = max(i for (i,j) in nz_idx)
    min_col = min(j for (i,j) in nz_idx)
    max_col = max(j for (i,j) in nz_idx)
    perms = []
    # Make a list of all possible column sign flips,
    # this seems to be the main symmetry in Z3 solutions
    valid_perms = set()
    for row_perm in [range(max_row+1)]:
        new_order = []
        new_nz_idx = [(row_perm[i], j) for (i,j) in nz_idx]
        new_order = sorted(range(n_terms), key=lambda x: (new_nz_idx[x][0], new_nz_idx[x][1]))
        for row_signs in [[1]*(max_row+1)]:
            for col_signs in itertools.product([1, -1], repeat=max_col-min_col+1):
                sign_map = {(i,j): row_signs[i]*col_signs[j - min_col] for (i,j) in nz_idx}
                this_perm = tuple((new_order[k], sign_map[nz_idx[k]]) for k in range(n_terms))
                valid_perms.add(this_perm)
    valid_perms = sorted(valid_perms, key=str)
    def _symmetry_perms(terms):
        perms = []
        for v_perm in valid_perms:
            new_terms = []
            for (idx, sign) in v_perm:
                new_terms.append(sign*terms[idx])
            perms.append(new_terms)
        return perms
    
    # Try to find solutions with increasing search space
    for max_result_range in [4, 10]:
        if debug:
            print("Trying max result range:", max_result_range)
            print("for transformation", Z)
        if apriori_sol is None:
            all_sols = find_rational_vars_integer_results(integer_constraints,
                                                nonzero_constraints, [], var_list,
                                                heuristic_upper=True,
                                                symmetry_map=_symmetry_perms,
                                                max_result_range=max_result_range,
                                                debug=debug,
                                                proven_lower_bound=variable_cost_certificate,
                                                accept_heuristic_optimum=canonical_optimum_is_unique)
        else:
            all_sols = find_rational_vars_integer_results(integer_constraints,
                                                nonzero_constraints, [], var_list,
                                                apriori_sol=apriori_sol-fixed_cost,
                                                heuristic_upper=False,
                                                symmetry_map=_symmetry_perms,
                                                max_result_range=max_result_range,
                                                debug=debug,
                                                proven_lower_bound=variable_cost_certificate)
        if all_sols or apriori_sol is not None:
            break

    # Can't beat apriori solution
    if all_sols == []:
        return []

    # Identify best solution by wJ^T canonicalization
    # Minimize L1 norm, with tiebreaker being max term
    min_key = tuple()
    best_s = None
    for sol in all_sols:
        s = sol["variables"]
        wJT_subs = sym.simplify(wJT_trans.subs(s))
        _, wMax, _ = _maximize_wT(wJT_subs[:, :n_nl])
        n_val = sum(abs(int(x)) for x in wMax)
        max_n_val = max(abs(int(x)) for x in wMax)
        key = (n_val, max_n_val, wMax)
        if key < min_key or min_key == tuple():
            min_key = key
            best_s = s
    if best_s is None:
        return None
    Zsubs = Z.subs(best_s)
    to_sub = [x for x in var_list if x in Zsubs.free_symbols]
    if len(to_sub) > 0:
        Zsubs = _find_Z_instance_deterministic(Zsubs, to_sub, max_tries=100, nonzero=nonzero)

    return Zsubs


def compact_well_aligned(wT: sym.Matrix, n_c: int):

    # Nothing to worry about here
    if n_c == 0:
        return True
    
    wc = wT[:, :n_c]
    rows, cols = wc.shape

    # 1 periodic variable -- is there a 1 or -1 there
    # and all other entries are integers (i.e. same periodicity)
    if n_c == 1:
        nz_entries = [sym.Abs(wc[i,0]) for i in range(rows) if wc[i,0].is_nonzero]
        if len(nz_entries) == 0:
            return False
        elif 1 in nz_entries and all(x.is_integer for x in nz_entries):
            return True
        else:
            return False
        
    # 2 or more periodic variables --
    # Does each compact variable have at least one junction that
    # depends on it alone (among compact variables)
    # And do all the other rows have only integer entries
    else:
        # Select all rows with a single 1/-1 and rest 0's
        e_rows = []
        for i in range(wc.shape[0]):
            nz_entries = [sym.Abs(wc[i,j]) for j in range(cols) if wc[i,j].is_nonzero]
            if len(nz_entries) == 0:
                continue
            if len(nz_entries) == 1 and nz_entries[0] == 1:
                e_rows.append(i)
            elif any(not x.is_integer for x in nz_entries):
                return False
        # Check if all the single entry rows span the
        # number of compact variables: yes->well aligned
        return len(_linearly_indep_rows(wc[e_rows, :])) == n_c


def compact_alignment_transformation(wT:sym.Matrix, n_c:int):

    wc = wT[:, :n_c]
    all_Zc = []

    # Consider unchanged if it works
    if compact_well_aligned(wc, n_c):
        all_Zc.append(sym.eye(n_c))

    # Try all different linearly independent row combinations
    row_sets = _linearly_indep_row_sets(wc)
    for rows in row_sets:
        # Get the appropriate transformation for each set
        Zc = wc[rows, :].pinv()
        # Verify it's well-aligned
        if compact_well_aligned(wc*Zc, n_c):
            if len(_find_equiv_mats(Zc, all_Zc, shifts=[])) == 0:
                all_Zc.append(Zc)

    # Return all possible transformations
    all_Z = []
    for Zc in all_Zc:
        Z = sym.eye(wT.shape[1])
        Z[:n_c, :n_c] = Zc
        all_Z.append(Z)

    return all_Z


def H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types, try_perms = True, extra_nl:bool=False):
    """
    Produces a hash that represents in the specified basis

    1) a = [nCompact][nExtended][nHarmonic]
    2) b = number of nonzero entries in L + number of nonzero entries in C
    3) c = positions of nonzero inductive intermode coupling (= 0 -> 0, != 0 -> 1)
    4) d = base 2 positions of nonzero capacitive intermode coupling (= 0 -> 0, != 0 -> 1)
    5) e = wJ_trans in canonical form, in base 3 (-1 -> 0, 0 -> 1, 1 -> 2)
    Swapping columns to maximize:

    hash = a_b_c_d_e (ex: 3_511_012)

    Keeps the variable transformation that minimizes the hash,
    after trying all possible permutations of the columns
    corresponding to different variable types.

    Args:
        circuit (_type_): _description_
        edges (_type_): _description_
        symmetric (bool, optional): _description_. Defaults to True.
        numerical (bool, optional):
        eps (float, optional): 
    """
    
    # Record number of modes
    n_nodes = len(functools.reduce(lambda a,b: a+b, var_types.values(), []))
    ext_var = var_types.get("extended", [])
    n_ext = len(ext_var)
    comp_var = var_types.get("compact", [])
    n_comp = len(comp_var)
    harm_var = var_types.get("harmonic", [])
    n_harm = len(harm_var)
    mode_str = f"{n_comp}{n_ext}{n_harm}"
    n_nd = n_nodes - n_ext - n_comp - n_harm
    n_dyn = n_nodes - n_nd
    n_nl = n_ext + n_comp

    # Truncate to just the dynamical modes
    L_tilde = lTrans[:n_dyn, :n_dyn]
    cTransInv = cTransInv[:n_dyn, :n_dyn]
    wJ = wJtTrans[:, :n_dyn]
    
    # All different ways to arrange column permutations in Z
    # that preserve variable types
    # Try all permutations of dynamical modes only
    # Note: column shift in Z is row and column shift in L, C
    # For permutation P and Z, the transformed L2 is
    # as L2 = (Z*P)^T * L * (Z*P) = P^T * Z^T * L * Z * P
    # C2inv is
    # C2inv = [(Z*P)^T * C * (Z*P)]^-1 = (Z*P)^-1 * C^-1 * (Z*P)^T^-1
    # = P^-1 * Z^-1 * C^-1 * Z^T^-1 * P^T^-1
    # Recall that P^-1 = P^T for permutation matrices
    # For wJ^T, we have
    # wJt2^T = wJ^T * (Z*P) = (wJ^T*Z) * P
    if try_perms:
        perms = _var_col_perms(var_types, dyn_only=True)
        perms = [p[:n_dyn] for p in perms]
    else:
        perms = [tuple(range(n_dyn))]

    lowest_hash = ""
    lowest_perm = tuple(range(n_nodes))
    for perm in perms:
        
        L_key = (_nonzero_entries_str(lTrans[perm, perm]),)
        C_key =  (_nonzero_entries_str(cTransInv[perm, perm]),)
        w_key = (_incidence_square_nonzero_str(
            wJtTrans[:, perm].transpose(), EJ
        ),)
        if extra_nl and n_nl > 0:
            wT_tilde, wT_key_full, _ = _maximize_wT(_sort_wT(wJtTrans[:, perm][:, :n_nl]))
            n_val = sum(abs(x) for x in wT_key_full)
            max_n_val = max(abs(x) for x in wT_key_full)
            w_key += (n_val, max_n_val, wT_key_full)


        # Make the hash string
        Z_hash = (mode_str,) + w_key + (int(L_key[0][0])+int(C_key[0][0]),) + L_key + C_key
        if lowest_hash == "" or Z_hash < lowest_hash:
            lowest_hash = Z_hash
            lowest_perm = perm + tuple(range(n_dyn, n_nodes))

    return lowest_hash, lowest_perm


def incidence_to_square(w, vals):
    mat = sym.Matrix(np.zeros((w.shape[0], w.shape[0])))
    for j in range(w.shape[1]):
        mat += vals[j]*w[:, j]*w[:, j].transpose()
    return mat


def _generate_block_transformation(var_types: dict[str, list[int]], trans_blocks: list[list[int]], prefix="Z",
                                   return_full=True):
    
    # Record mode types
    comp = var_types.get("compact", [])
    n_comp = len(comp)
    ext = var_types.get("extended", [])
    n_ext = len(ext)
    harm = var_types.get("harmonic", [])
    n_harm = len(harm)
    n_d = n_comp + n_ext + n_harm
    n_nd = len(var_types.get("free", []) + var_types.get("frozen", []) + var_types.get("sigma", []))
    mode_vec = [comp, ext, harm]

    # Create symbolic variables for transformation
    var_list = []
    Z_blocks = []
    Z = sym.eye(n_d+n_nd)
    var_list = []
    for block_i, block_j in itertools.product(range(3), range(3)):
        if trans_blocks[block_i][block_j]:
            for i in mode_vec[block_i]:
                for j in mode_vec[block_j]:
                    Z[i, j] = sym.Symbol(f"{prefix}{i}{j}")
                    var_list.append(Z[i,j])

    if return_full:
        return Z, var_list
    else:
        return Z[:n_d, :n_d], var_list


def secondary_decouple(var_types: dict[str, list[int]], mats: list[sym.Matrix], invert: list[bool], **kwargs):
    """
    Performs a secondary decoupling transformation of
    the form
    [ I   Z_ce   0     0 ]
    [ 0   Z_e    0     0 ]
    [ 0   Z_eh  Z_hh   0 ]
    [ 0    0     0     I ]
    that maximizes the number of nonzero off diagonal elements in the
    transformed susceptance and inverse capacitance matrices.
    In case of a tie in number, prefers fewer inductive terms,
    in line with H_hash.

    The form of the transformation can be adjusted via the trans_blocks kwarg.

    Args:
        var_types (dict[str, list[int]]): Dictionary mapping mode type to a list
                                          of indices that are that mode.
        mats (list[sym.Matrix]): List of matrices to decouple
        invert (list[bool]): List of whether to invert each matrix in mats
        return_instance: bool
            If there are free parameters in the final transformation,
            substitute in actual values to get a concrete instance.
            Defaults to False.
    kwargs:
        Z_in (sym.Matrix): Possible initial secondary transformation
        trans_blocks (list[list[int]]): Blocks of the transformation
                                        rows/columns correspond to
                                        [compact, extended, harmonic]
                                        non-dynamical modes are not transformed
        prefix (str): Prefix for symbolic variables in the transformation.
        param_symbols (list[str]): List of strings to identify
                                   system parameters in the matrices.
                                   Defaults to ["C", "L", "J"].
        tiebreaker_fn (callable): Function that takes a set of keys
                                   and returns a sortable item
                                   when choosing transformations.
        names (list[str]): circuit_quantization: a label for each matrix in mats,
                           used in the term keys (name, i, j). Defaults to "0", "1", ...
        row_mats (list[sym.Matrix]): circuit_quantization: rectangular matrices M with one
                           column per mode, such as the junction incidence omega_J^T,
                           transformed as M Z rather than by congruence. Their entries
                           (k, j) in noncompact columns j are decoupling terms, after
                           those of mats; compact columns are fixed by the alignment.
        row_names (list[str]): circuit_quantization: labels for row_mats in the term keys.
                           Defaults to "R0", "R1", ...
        pattern_order (callable): circuit_quantization: takes the term keys, one
                                  (name, i, j) per off-diagonal entry in mats order,
                                  and returns (cost, term indices) candidates for
                                  maximally_compatible_sol in non-decreasing cost.
                                  Defaults to largest subsets first.

    Returns:
        sym.Matrix: Secondary transformation
    """
    ### process kwargs
    # Blocks of the transformation
    # rows/columns correspond to [compact, extended, harmonic]
    # non-dynamical modes are not transformed
    trans_blocks = kwargs.get("trans_blocks", [[0, 1, 0],
                                               [0, 1, 0],
                                               [0, 1, 1]])
    return_tiebreaker = kwargs.get("return_tiebreaker", False)
    Z_in = kwargs.get("Z_in", None)
    prefix = kwargs.get("prefix", "Z")
    param_symbols = kwargs.get("param_symbols", ["C", "L", "J"])

    debug = kwargs.get("debug", False)
    names = kwargs.get("names", [str(k) for k in range(len(mats))])
    row_mats = kwargs.get("row_mats", [])
    row_names = kwargs.get("row_names", [f"R{k}" for k in range(len(row_mats))])
    pattern_order = kwargs.get("pattern_order", None)

    # Record mode types
    comp = var_types.get("compact", [])
    n_comp = len(comp)
    ext = var_types.get("extended", [])
    n_ext = len(ext)
    harm = var_types.get("harmonic", [])
    n_harm = len(harm)
    n_d = n_comp + n_harm + n_ext
    mode_vec = [comp, ext, harm]
    tiebreaker_fn = kwargs.get("tiebreaker_fn", lambda keys: "".join(["0" if i in keys else "1" for i in range(n_d*(n_d-1)//2)]))


    # Generate Possible Transformation
    if Z_in is None:
        Z, var_list = _generate_block_transformation(var_types, trans_blocks, prefix=prefix)
    else:
        Z = Z_in
        var_list = [x for x in Z.free_symbols if prefix in str(x)]
    # Sort variable list for consistent substitution
    var_list.sort(key=lambda s: s.name)

    # Transform matrices and invert as needed
    to_decouple = []
    for mat, inv in zip(mats, invert):
        transformed = Z[:n_d, :n_d].transpose()*mat[:n_d, :n_d]*Z[:n_d, :n_d]
        if inv:
            if transformed.shape[0] > 1:
                to_decouple.append(sym.inv_quick(transformed))
            else:
                to_decouple.append(transformed.inv())
        else:
            to_decouple.append(transformed)
    
    # Symmetric coupling -> only consider upper triangle
    pairs = [(i, j) for (i, j) in itertools.product(range(n_d), repeat=2) if i < j]
    targets = [(name, trMat, pairs) for name, trMat in zip(names, to_decouple)]
    # circuit_quantization: rectangular matrices, entry by entry in noncompact columns
    noncompact = [j for j in range(n_d) if j not in comp]
    for name, mat in zip(row_names, row_mats):
        trMat = mat[:, :n_d]*Z[:n_d, :n_d]
        targets.append((name, trMat, [(k, j) for k in range(trMat.rows) for j in noncompact]))
    coupling = []
    term_keys = []  # (name, i, j) for each entry of coupling
    zero_keys = []
    count = 0

    for name, trMat, entries in targets:
        for (i, j) in entries:
            term_keys.append((name, i, j))
            vars_in_eq = [v for v in var_list if v in trMat[i, j].free_symbols]
            # If it takes more than 1 second to determine equations for decoupling, then
            # assume it's unsolvable and move on to avoid long computation times
            ans = utils.run_with_timeout(eq_indep_of_vars, (trMat[i, j], vars_in_eq), timeout=1/60)
            count += 1
            if ans is None:
                coupling.append([])
            else:
                eqs, denom = ans
                if not eqs:
                    zero_keys.append(count)
                coupling.append(eqs)

    # Nothing to decouple
    if all(eq == [] for eq in coupling):
       if return_tiebreaker:
           return [Z], [tiebreaker_fn(zero_keys)]
       else:
           return [Z]
    
    # We want to keep det(Z) nonzero. Consider least common multiple
    # of denominators in Z to clear fractions Z*LCM = Z' (integer matrix)
    # Z^-1 = 1/det(Z) * adj(Z) ->
    # Z^-1 = (Z'/LCM)^-1 = LCM*Z'^-1 = LCM/det(Z') * adj(Z')
    # So we want to keep det(Z)/LCM nonzero, as well as LCM itself
    # Similarly, (Z.T C Z) ^-1 = Z.T^-1 C^-1 Z^-1
    # = LCM^2/det(Z')^2 * adj(Z').T C^-1 adj(Z')
    # C is invertible, we only need to keep det(Z)/LCM nonzero
    Z_common_denom = sym.lcm([elem.as_numer_denom()[1] for elem in Z[:] if elem != 0])
    Z_poly = sym.cancel(Z*Z_common_denom)
    if debug:
        print("Calling maximally compatible sol with coupling:")
    # Flatten the list of possible substitutions
    all_keys, all_subs = maximally_compatible_sol(coupling,
                            nonzero_constraints=[_det_fast(Z_poly), Z_common_denom],
                            rational_only=True, stop_at_first=kwargs.get("stop_at_first", True),
                            debug=debug, get_all_sols=True,
                            patterns=None if pattern_order is None else pattern_order(term_keys))
    # Cannot decouple anything
    if not all_keys:
        if return_tiebreaker:
            return [Z], [tiebreaker_fn(zero_keys)]
        else:
            return [Z]

    all_keys = list(itertools.chain.from_iterable([[all_keys[i]]*len(all_subs[i]) for i in range(len(all_keys))]))
    all_subs = list(itertools.chain.from_iterable(all_subs))

    # Remove transformations that are column permutations or sign-flips of each other
    perms = _var_col_perms(var_types, dyn_only=True)
    perms = [p + tuple(range(n_d, Z.shape[0])) for p in perms]
    unique_Z = []
    unique_keys = []
    unique_subs = []
    # With the unique transformations, sort by tiebreaker function
    key_order = np.argsort([tiebreaker_fn(keys) for keys in all_keys])
    all_Z = [Z.subs(all_subs[i]) for i in key_order]
    _, idx_kept = _remove_permutation_equivalent_transformations(all_Z, perms)
    unique_Z = [all_Z[j] for j in idx_kept]

    if return_tiebreaker:
        return unique_Z, [tiebreaker_fn(all_keys[key_order[j]]) for j in idx_kept]
    else:
        return unique_Z

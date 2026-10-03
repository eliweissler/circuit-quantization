# Ported from sircuitenum/tests/test_quantize.py (GPL-3.0), sircuitenum commit b2c57e8.
# The tests of vendored functions are verbatim. Tests that built inputs with
# sircuitenum's circuit helpers read those inputs from sircuitenum_fixtures.json;
# test_choose_Z becomes test_choose_S1_reproduces_choose_Z, through circuit_quantization.
import functools
import itertools
import time

import numpy as np
import sympy as sym

import circuit_quantization as cq
from circuit_quantization.optimize import transforms as quantize
from sircuitenum_circuits import FIXTURES, fixture, to_circuit

num_map = {"0": -4, "1": -3, "2": -2, "3": -1,"4": 0, "5": 1, "6": 2, "7": 3, "8": 4}

inv_num_map = {v: k for k, v in num_map.items()}

def new_to_old(k):
    old_form = ""
    for c in k:
        if isinstance(c, str):
            old_form += c + "_"
        elif isinstance(c, tuple):
            old_form += "".join(inv_num_map[cr] for cr in c) + "_"
        elif isinstance(c, int):
            old_form += str(c) + "_"
    return old_form[:-1]



def test__independent_from():
    # Basic: A = {[1,0], [0,1]}, B = {[1,1]}
    A = [sym.Matrix([1, 0]), sym.Matrix([0, 1])]
    B = [sym.Matrix([1, 1])]
    result = quantize._independent_from(A, B)
    assert len(result) == 1
    assert any(v == sym.Matrix([1, 0]) for v in result) or any(v == sym.Matrix([0, 1]) for v in result)

    A = [sym.Matrix([1, 1]), sym.Matrix([0, -1])]
    B = [sym.Matrix([1, 0])]
    result = quantize._independent_from(A, B)
    assert len(result) == 1
    assert any(v == sym.Matrix([1, 1]) for v in result) or any(v == sym.Matrix([0, -1]) for v in result)

    # Subset: A = {[1,0], [0,1]}, B = {[1,0]}
    A = [sym.Matrix([1, 0]), sym.Matrix([0, 1])]
    B = [sym.Matrix([1, 0])]
    result = quantize._independent_from(A, B)
    assert len(result) == 1
    assert result[0] == sym.Matrix([0, 1])

    # All in B: A = {[1,0]}, B = {[1,0]}
    A = [sym.Matrix([1, 0])]
    B = [sym.Matrix([1, 0])]
    result = quantize._independent_from(A, B)
    assert result == []

    # Empty B: A = {[1,0], [0,1]}, B = []
    A = [sym.Matrix([1, 0]), sym.Matrix([0, 1])]
    B = []
    result = quantize._independent_from(A, B)
    assert len(result) == 2
    assert any(v == sym.Matrix([1, 0]) for v in result)
    assert any(v == sym.Matrix([0, 1]) for v in result)

    # Symbolic: A = {[x,0], [0,y]}, B = {[x,y]}
    x, y = sym.symbols('x y')
    A = [sym.Matrix([x, 0]), sym.Matrix([0, y])]
    B = [sym.Matrix([x, y])]
    result = quantize._independent_from(A, B)
    assert len(result) == 1
    assert any(v == sym.Matrix([x, 0]) for v in result) or any(v == sym.Matrix([0, y]) for v in result)


def test__vec_space_overlap():

    vecs1 = [sym.Matrix([1, 0, 1]),
             sym.Matrix([1, 1, 0])]
    
    vecs2 = [sym.Matrix([2, 1, 0]),
             sym.Matrix([-1, 0, 0])]

    overlap, in_v1, in_v2 = quantize._vec_space_overlap(vecs1, vecs2, return_decomp=True)

    assert len(overlap) == 1


def test__remove_permutation_equivalent_transformations():
    """Test that identical matrices after column permutation are filtered."""
    # Simple matrices where permutation creates duplicate
    Z1 = sym.Matrix([[1, 0], [0, 1]])
    Z2 = sym.Matrix([[0, 1], [1, 0]])  # Column permutation of Z1
    
    Z_list = [Z1, Z2]
    perms = [(0, 1), (1, 0)]  # Identity and swap permutations
    
    filt_Z, idx_keep = quantize._remove_permutation_equivalent_transformations(Z_list, perms)
    
    # Should only keep one since they're equivalent under permutation
    assert len(filt_Z) == 1
    assert len(idx_keep) == 1
    assert idx_keep[0] == 0  # Should keep the first one


    a, b = sym.symbols("a b", real=True)
    Z1 = sym.Matrix([[a, 0], [0, b]])
    Z2 = sym.Matrix([[a, b], [b, a]])
    Z3 = sym.Matrix([[2*a, 0], [0, 3*b]])
    
    Z_list = [Z1, Z2, Z3]
    perms = [(0, 1), (1, 0)]
    
    filt_Z, idx_keep = quantize._remove_permutation_equivalent_transformations(Z_list, perms)
    
    # Should keep at least one, and filter duplicates if any
    assert len(filt_Z) == 3
    assert len(idx_keep) == len(filt_Z)
    # Indices should be valid
    assert all(0 <= i < 3 for i in idx_keep)


    a = sym.symbols("a", real=True)
    Z1 = sym.Matrix([[a, 0],
                     [0, a]])
    Z2 = sym.Matrix([[0, -b],
                    [-b, 0]])  # Sign flip and column permutation of Z1
    
    Z_list = [Z1, Z2]
    perms = [(0, 1), (1, 0)]
    
    filt_Z, idx_keep = quantize._remove_permutation_equivalent_transformations(Z_list, perms)
    
    # Should filter one as they're equivalent up to sign
    assert len(filt_Z) == 1
    assert len(idx_keep) == len(filt_Z)
    assert idx_keep[0] == 0  # Should keep the first one


def test__linearly_indep_rows_cols():

    x,y,z = sym.symbols("x,y,z")
    test_cases = [
        # Empty matrix
        (sym.Matrix([]), []),
        
        # Single row matrices
        (sym.Matrix([[1, 2, 3]]), [0]),
        (sym.Matrix([[0, 0, 0]]), []),
        
        # Identity matrix
        (sym.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [0, 1, 2]),
        
        # Duplicate rows
        (sym.Matrix([[1, 2, 3], [1, 2, 3], [4, 5, 6]]), [0, 2]),
        
        # Scaled rows
        (sym.Matrix([[1, 2, 3], [2, 4, 6], [3, 5, 7]]), [0, 2]),
        
        # Linear combination
        (sym.Matrix([[1, 0, 1], [0, 1, 1], [1, 1, 2]]), [0, 1]),
        
        # Symbolic matrix
        (sym.Matrix([[x, y, z], 
                    [2*x, 2*y, 2*z], 
                    [x+1, y, z]]), [0, 2]),
        
        # Mixed entries
        (sym.Matrix([[1, x, 3], 
                [2, 2*x, 6], 
                [4, 3*x, 5]]), [0, 2]),
        
        # Zero rows interspersed
        (sym.Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]]), [0, 2]),
        
        # Rectangular matrix with more rows than columns
        (sym.Matrix([[1, 0], [0, 1], [1, 1], [2, 0]]), [0, 1]),
        
        # Rectangular matrix with more columns than rows
        (sym.Matrix([[1, 0, 1, 2], [0, 1, 1, 3]]), [0, 1]),
        
        # Fractions
        (sym.Matrix([[sym.Rational(1,2), sym.Rational(1,3), sym.Rational(1,4)], 
                [sym.Rational(1,1), sym.Rational(2,1), sym.Rational(3,1)], 
                [sym.Rational(1,3), sym.Rational(2,3), sym.Rational(3,3)]]), [0, 1]),
    ]
    
    # Run all test cases
    for matrix, expected in test_cases:
        row_set = quantize._linearly_indep_rows(matrix)
        col_set = quantize._linearly_indep_cols(matrix.transpose())
        assert row_set == tuple(expected)
        assert col_set == tuple(expected)


def test__linearly_indep_row_col_sets():

    x,y,z = sym.symbols("x,y,z")
    test_cases = [
        # Empty matrix
        (sym.Matrix([]), [()]),
        
        # Single row matrices
        (sym.Matrix([[1, 2, 3]]), [(0,)]),
        (sym.Matrix([[0, 0, 0]]), [()]),
        
        # Identity matrix
        (sym.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [(0, 1, 2)]),
        
        # Duplicate rows
        (sym.Matrix([[1, 2, 3], [1, 2, 3], [4, 5, 6]]), [(0, 2), (1, 2)]),
        
        # Scaled rows
        (sym.Matrix([[1, 2, 3], [2, 4, 6], [3, 5, 7]]), [(0, 2), (1, 2)]),
        
        # Linear combination
        (sym.Matrix([[1, 0, 1], [0, 1, 1], [1, 1, 2]]), [(0, 1), (0, 2), (1, 2)]),
        
        # Symbolic matrix
        (sym.Matrix([[x, y, z], 
                    [2*x, 2*y, 2*z], 
                    [x+1, y, z]]), [(0, 2), (1, 2)]),
        
        # Mixed entries
        (sym.Matrix([[1, x, 3], 
                [2, 2*x, 6], 
                [4, 3*x, 5]]), [(0, 2), (1, 2)]),
        
        # Zero rows interspersed
        (sym.Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]]), [(0, 2)]),
        
        # Rectangular matrix with more rows than columns
        (sym.Matrix([[1, 0], [0, 1], [1, 1], [2, 0]]), [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]),
        
        # Rectangular matrix with more columns than rows
        (sym.Matrix([[1, 0, 1, 2], [0, 1, 1, 3]]), [(0, 1)]),
        
        # Fractions
        (sym.Matrix([[sym.Rational(1,2), sym.Rational(1,3), sym.Rational(1,4)], 
                [sym.Rational(1,1), sym.Rational(2,1), sym.Rational(3,1)], 
                [sym.Rational(1,3), sym.Rational(2,3), sym.Rational(3,3)]]), [(0, 1), (0, 2)]),
    ]
    
    # Run all test cases
    for matrix, expected in test_cases:
        row_sets = quantize._linearly_indep_row_sets(matrix)
        col_sets = quantize._linearly_indep_col_sets(matrix.transpose())
        assert len(col_sets) == len(expected)
        assert len(row_sets) == len(expected)
        assert all(x in expected for x in row_sets)
        assert all(x in expected for x in col_sets)


def test__det_fast():

    for i in range(10):
        mat = sym.randMatrix(i, i, min=0, max=5)
        assert mat.det() == quantize._det_fast(mat)

    # Regression case for rational 5x5 transformations: the default SymPy
    # determinant algorithm expands this class of expression very slowly.
    a, b, c, d = sym.symbols("a b c d", nonzero=True)
    mat = sym.diag(a / b, c / d, 1, 1, 1)
    assert sym.cancel(quantize._det_fast(mat) - a*c/(b*d)) == 0


def test__equiv_cols():

    c1 = sym.Matrix([[1], [1]])
    c2 = sym.Matrix([[1], [-1]])
    assert quantize._equiv_cols(c1, c2) == False

    c1 = sym.Matrix([[1], [0]])
    c2 = sym.Matrix([[0], [1]])
    assert quantize._equiv_cols(c1, c2) == True

    c1 = sym.Matrix([[1], [0], [0]])
    c2 = sym.Matrix([[0], [1], [-1]])
    assert quantize._equiv_cols(c1, c2) == False


    c1 = sym.Matrix([[1], [0], [0]])
    c2 = sym.Matrix([[0], [1], [-1]])
    shift = sym.Matrix([[1], [-1], [1]])*100
    assert quantize._equiv_cols(c1, c2, shifts=[shift]) == True


def test__equal_up_to_column_shift_and_sign():

    shift = sym.ones(2, 1)

    # Shift no columns
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # Shift one column
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    matrix2[:, 0] += shift
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # Shift both columns
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    matrix2[:, 0] += shift
    matrix2[:, 1] += -1*shift
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True

    # scale a column
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    matrix2[:, 0]*=2
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == False
    
    # With symbolic entries
    x, y = sym.symbols('x y', positive=True)
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = sym.Matrix([[x-y, 2], [0, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # With fractions
    matrix1 = sym.Matrix([[sym.Rational(1,2), sym.Rational(2,1)], [sym.Rational(3,4), sym.Rational(4,1)]])
    matrix2 = sym.Matrix([[-sym.Rational(1,2), sym.Rational(2,1)], [-sym.Rational(1,4), sym.Rational(4,1)]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # Different columns with symbols, first one is *-1 -> shift away
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = sym.Matrix([[y, 2], [x, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True

    # Different columns with symbols
    matrix1 = sym.Matrix([[x+y, 2], [y, 4]])
    matrix2 = sym.Matrix([[y, 2], [x, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == False

    # Identical columns
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # Negative of column
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[-1, 2], [-3, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # With symbolic entries
    x, y = sym.symbols('x y')
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = sym.Matrix([[-x, 2], [-y, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # With fractions
    matrix1 = sym.Matrix([[sym.Rational(1,2), sym.Rational(2,1)], [sym.Rational(3,4), sym.Rational(4,1)]])
    matrix2 = sym.Matrix([[-sym.Rational(1,2), sym.Rational(2,1)], [-sym.Rational(3,4), sym.Rational(4,1)]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True


def test__find_equiv_cols():

    shift = sym.ones(2, 1)

    # Do nothing
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = matrix1
    assert quantize._find_equiv_cols(matrix1, [matrix2, shift, matrix2+shift], shifts=[shift]) == [0, 2]

    # just shift
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = matrix1 - shift
    assert quantize._find_equiv_cols(matrix1, [matrix2, shift], shifts=[shift]) == [0]

    # shift and sign
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = -matrix1 + shift
    assert quantize._find_equiv_cols(matrix1, [shift, matrix2], shifts=[shift]) == [1]

    # shift and sign and scale
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = -2*matrix1 + shift
    assert quantize._find_equiv_cols(matrix1, [matrix2], shifts=[shift]) == []

    # different shift
    matrix1 = sym.Matrix([[1], [0]])
    shift = sym.Matrix([[0], [1]])
    matrix2 = -matrix1 + shift
    assert quantize._find_equiv_cols(matrix1, [matrix2], shifts=[shift]) == [0]

    # Failed for some reason
    matrix1 = sym.simplify(sym.Matrix([-1/2, -1/2, 1/2, 1/2]), rational=True)
    matrix2 = sym.simplify(sym.Matrix([1/2, 1/2, -1/2, -1/2]), rational=True)
    assert quantize._find_equiv_cols(matrix1, [matrix2]) == [0]

    matrix1 = sym.simplify(sym.Matrix([-1/2, -1/2, 1/2, 1/2]), rational=True)
    matrix2 = sym.simplify(sym.Matrix([1/2, 0, -1/2, -1/2]), rational=True)
    assert quantize._find_equiv_cols(matrix1, [matrix2]) == []


def test__find_equiv_mats():

    shift = sym.ones(2, 1)

    # Do nothing
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = matrix1
    matrix3 = sym.Matrix([[1], [-1]])
    matrix4 = -matrix3
    assert quantize._find_equiv_mats(sym.Matrix.hstack(matrix1, matrix2), [sym.Matrix.hstack(matrix3, matrix4)], shifts=[shift]) == []
    assert quantize._find_equiv_mats(sym.Matrix.hstack(matrix1, matrix3), [sym.Matrix.hstack(matrix3, matrix4),
                                                                    sym.Matrix.hstack(matrix4, matrix1)], shifts=[shift]) == [1]


def test__equal_up_to_column_swaps_and_shift_and_sign():

    shift = sym.ones(2, 1)

    # Shift no columns
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True
    
    # Shift one column
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 0] += shift
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True
    
    # Shift both columns
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 0] += shift
    matrix2[:, 1] += -1*shift
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True

    # scale a column
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 0]*=2
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == False

    # swap columns
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 0] += shift
    matrix2[:, 1] += -1*shift
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True

    # With symbolic entries
    x, y = sym.symbols('x y', positive=True)
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 1]*=-1
    matrix2[:, 1]+=7*shift
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True
    
    # With fractions
    matrix1 = sym.Matrix([[sym.Rational(1,2), sym.Rational(2,1)], [sym.Rational(3,4), sym.Rational(4,1)]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 1]*=2
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == False


def test__nonzero_entries_str():

    test = np.array([[1, 0, 2],
                      [0, 0.01, 0],
                      [2, 0, 3]])
    
    res = quantize._nonzero_entries_str(test)
    assert res == "1-010"

    test = sym.Matrix(np.array([[1, 0.01, 2],
                      [-1, 0.01, -.2],
                      [7.3, 0, 3]]))
    
    res = quantize._nonzero_entries_str(test)
    assert res == "3-111"
    res = quantize._nonzero_entries_str(test, prepend_sum=False)
    assert res == "111"


def test__sort_wT():
     
    test = np.array([[1, 0, 1],
                      [-1, 1, 0],
                      [0, 1, 0]])
    assert np.all(quantize._sort_wT(test) == test[(2, 1, 0), :])

    # rows with same nonzero counts but different nonzero column indices
    mat = sym.Matrix([[0, 1, 0],
                      [1, 0, 0],
                      [0, 1, 1],
                      [0, 0, 0]])
    sorted_mat = quantize._sort_wT(mat)
    # keys are sorted lexicographically; expected order computed from key logic
    expected_order = (3, 1, 0, 2)
    expected = np.array(mat)[expected_order, :]
    assert np.array_equal(np.array(sorted_mat), expected)

    # numpy array input with negative entries should behave the same
    arr = np.array([[0, 0], [0, -1], [1, 0]])
    sorted_arr = quantize._sort_wT(arr)
    # compute expected keys and ordering to assert deterministic behavior
    keys = []
    for i in range(arr.shape[0]):
        keys.append(str(sum(arr[i, :] != 0)) + "_" + "-".join(arr[i, :].nonzero()[0].astype(str)))
    order = np.argsort(keys)
    assert np.array_equal(sorted_arr, arr[order, :])


def test__maximize_wT():

    # Map old format string keys to numerical tuples
    num_map = {"0": -4, "1": -3, "2": -2, "3": -1,"4": 0, "5": 1, "6": 2, "7": 3, "8": 4}
    str_to_num = lambda s: tuple(num_map[c] for c in s)

    test = sym.nsimplify(sym.Matrix([
                        [-1, -1,  0, 0],
                        [-1,  0,  1, 0],
                        [-1,  0,  0, 0],
                        [ 0,  1,  1, 0],
                        [ 0,  1,  0, 0],
                        [ 0,  0, -1, 0]]), rational=True)
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test[:, :-1])
    ans = str_to_num('544454445554545453')
    assert best_key == ans
     
    test = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)

    assert np.all(wT == test)
    assert np.all(row_order == np.arange(3))
    assert np.all(row_vec == np.ones(3))
    assert np.all(col_vec == np.ones(3))
    assert best_key == str_to_num("".join(str(int(x) + 3) for x in "211121112"))

    test = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, -1, 1]])
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    ans = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 1]])
    assert np.all(wT == ans)
    assert np.all(row_order == np.arange(3))
    assert best_key == str_to_num("".join(str(int(x) + 3) for x in "211121122"))


    test = np.array([ [0, 0, 0, -1],
                      [-1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, -1, 1, 0]])
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    ans = np.array([ [0, 0, 0, 1],
                      [1, 1, 0, 0],
                      [1, -1, 0, 0],
                      [0, 1, 1, 0]])
    assert np.all(wT == ans)
    assert np.all(row_order == np.array([0, 2, 1, 3]))
    assert best_key == str_to_num("".join((ans + 4).flatten().astype(str)))

    test = np.array([[-1,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 1,  0,  0,  0, -1],
                    [ 0, -1,  0,  1,  0],
                    [ 0,  1,  0,  1,  0],
                    [ 0, -1,  0,  0, -1],
                    [ 0,  0,  1,  0, -1],
                    [ 0,  0,  0,  1, -1],
                    [ 1, -1, -1,  0,  0],
                    [-1,  0, -1,  1,  1]])
    import time
    t0 = time.time()
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    tf = time.time()
    print(tf-t0)
    ans = np.array([[ 1,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 1,  0,  0,  0,  1],
                    [ 0,  1,  0,  1,  0],
                    [ 0,  1,  0, -1,  0],
                    [ 0,  1,  0,  0,  1],
                    [ 0,  0,  1,  0,  1],
                    [ 0,  0,  0,  1,  1],
                    [ 1,  1, -1,  0,  0],
                    [ 1,  0,  1, -1,  1]])
    assert np.all(wT == ans)
    
    
    test = (np.random.random((17, 5)) > 0.75).astype(int) - (np.random.random((17, 5)) > 0.75).astype(int)
    test = quantize._sort_wT(test)
    t0 = time.time()
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    tf = time.time()
    print(tf-t0)


def test__var_col_perms():

    var_types = {"compact": [0, 1],
                 "extended": [2],
                 "harmonic": [3]}
    perms = quantize._var_col_perms(var_types)
    assert len(perms) == 2
    assert (0,1,2,3) in perms
    assert (1,0,2,3) in perms


def test__sub_equal_LC():

    C1, C2, C3 = sym.symbols("C1, C2, C3")
    CJ1, CJ2 = sym.symbols("CJ1, CJ2")
    L1, L2 = sym.symbols("L1, L2")
    X = sym.Matrix([[C1, L1, C2],
                    [C3, C1, L2],
                    [CJ1, CJ2, CJ1]])
    
    test = quantize._sub_equal_LC(X)

    assert test[0,0] == test[0,2] == test[1,0] == test[1,1]
    assert test[0,1] == test[1,2]
    assert test[2,0] == test[2,1] == test[2,2]


def test__find_Z_deterministic():

    # Solution that only has a single thing
    x,y = sym.symbols("x,y", real=True)
    a,b = sym.symbols("a,b", real=True)
    v = [x,y,a,b]

    Z = sym.Matrix([[x,y],
                    [a,b]])
    
    Zsub = quantize._find_Z_instance_deterministic(Z, v)
    assert Z.det().simplify() != 0
    assert sym.im(Zsub).is_zero_matrix


def test__find_Z_min_cost():

    Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22 = sym.symbols('Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22')
    wJ =  sym.Matrix([[1, 1, 1, 0, 0, 0], [-1, 0, 0, 1, 1, 0], [0, -1, 0, -1, 0, 1], [0, 0, -1, 0, -1, -1]])
    var_list = [Z20, Z10, Z21, Z22]
    nonzero = [1, 2*Z10 + Z20, 4]
    Z = sym.Matrix([[-Z10 - Z20, 3*Z21*(-Z10 + Z20)/(4*(2*Z10 + Z20)) + Z21*(Z10 + 2*Z20)/(4*(2*Z10 + Z20)) - Z21/4, Z22/4, 1/4], [Z10, -Z21*(-Z10 + Z20)/(4*(2*Z10 + Z20)) - 3*Z21*(Z10 + 2*Z20)/(4*(2*Z10 + Z20)) - Z21/4, Z22/4, 1/4], [Z20, -Z21*(-Z10 + Z20)/(4*(2*Z10 + Z20)) + Z21*(Z10 + 2*Z20)/(4*(2*Z10 + Z20)) + 3*Z21/4, Z22/4, 1/4], [0, -Z21*(-Z10 + Z20)/(4*(2*Z10 + Z20)) + Z21*(Z10 + 2*Z20)/(4*(2*Z10 + Z20)) - Z21/4, -3*Z22/4, 1/4]])
    var_types = {'compact': [], 'extended': [0, 1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    assert min_cost == 5

    Z21, Z12, Z01, Z02, Z11, Z22 = sym.symbols('Z21 Z12 Z01 Z02 Z11 Z22')

    #     wJ.transpose()*Z_final[-1] = 
    # Matrix([
    # [-1, Z11/2,    Z22/2, 0],
    # [-1, Z11/2,  3*Z22/2, 0],
    # [ 0,     0,      Z22, 0],
    # [ 1, Z11/2,   -Z22/2, 0],
    # [ 1, Z11/2, -3*Z22/2, 0]]) -> Z11,Z22 = 1 -> min_cost = 4+1+3+2+1+3 = 14 +(4 compact) = 18
    
    wJ = sym.nsimplify(sym.Matrix([
            [ 1,  1,  0,  0,  0],
            [-1,  0,  1,  1,  0],
            [ 0, -1, -1,  0,  1],
            [ 0,  0,  0, -1, -1]]), rational=True)
    Z = sym.nsimplify(sym.Matrix([
            [-1/2,  Z11/2, Z22/2, 1/4],
            [ 1/2,      0,     0, 1/4],
            [ 1/2,      0,  -Z22, 1/4],
            [-1/2, -Z11/2, Z22/2, 1/4]]))
    var_types = {'compact': [0], 'extended': [1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    var_list = [Z11, Z22]
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    #     wJ.transpose()*Z_final[-1] = 
    # Matrix([
    # [-1, Z11/2,    Z22/2, 0],
    # [-1, Z11/2,  3*Z22/2, 0],
    # [ 0,     0,      Z22, 0],
    # [ 1, Z11/2,   -Z22/2, 0],
    # [ 1, Z11/2, -3*Z22/2, 0]]) -> Z11,Z22 = 1 -> min_cost = 4+1+3+2+1+3 = 14 +(4 compact) = 18
    assert min_cost == 18

    wJ =  sym.Matrix([[1, 1, 1, 0, 0, 0], [-1, 0, 0, 1, 1, 0], [0, -1, 0, -1, 0, 1], [0, 0, -1, 0, -1, -1]])
    var_types =  {'compact': [0], 'extended': [1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}

    # Same as test z3 interface, but with 3 entries from compact mode
    Z = sym.Matrix([[-1/4, -Z01/4 - Z21/4, -3*Z02*Z21/(4*(2*Z01 - Z21)) - Z02/4, 1/4], [3/4, 3*Z01/4 - Z21/4, Z02*Z21/(4*(2*Z01 - Z21)) + 3*Z02/4, 1/4], [-1/4, -Z01/4 + 3*Z21/4, Z02*Z21/(4*(2*Z01 - Z21)) - Z02/4, 1/4], [-1/4, -Z01/4 - Z21/4, Z02*Z21/(4*(2*Z01 - Z21)) - Z02/4, 1/4]])
    var_list =  [Z01, Z02, Z21]
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    assert min_cost == 9

    Z =  sym.Matrix([[-1/4, 3*Z11/4, -Z22/2, 1/4], [3/4, -Z11/4, Z22/2, 1/4], [-1/4, -Z11/4, Z22/2, 1/4], [-1/4, -Z11/4, -Z22/2, 1/4]])
    wJT =  sym.Matrix([[-1, Z11, -Z22, 0], [0, Z11, -Z22, 0], [0, Z11, 0, 0], [1, 0, 0, 0], [1, 0, Z22, 0], [0, 0, Z22, 0]])
    var_list = [Z11, Z22]
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    assert min_cost == 10

    Z =  sym.Matrix([[-1/4, -3*Z21/8, 3*Z12/4, 1/4], [3/4, Z21/8, -Z12/4, 1/4], [-1/4, 5*Z21/8, -Z12/4, 1/4], [-1/4, -3*Z21/8, -Z12/4, 1/4]])
    wJT =  sym.Matrix([[-1, -Z21/2, Z12, 0], [0, -Z21, Z12, 0], [0, 0, Z12, 0], [1, -Z21/2, 0, 0], [1, Z21/2, 0, 0], [0, Z21, 0, 0]])
    var_list = [Z21, Z12]
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    assert min_cost == 13


    return


def test__find_Z_instance():

    Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22 = sym.symbols('Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22')
 
    Z = sym.nsimplify(sym.Matrix([
            [-1/2,  Z11/2, Z22/2, 1/4],
            [ 1/2,      0,     0, 1/4],
            [ 1/2,      0,  -Z22, 1/4],
            [-1/2, -Z11/2, Z22/2, 1/4]]))
    wJ = sym.nsimplify(sym.Matrix([
            [ 1,  1,  0,  0,  0],
            [-1,  0,  1,  1,  0],
            [ 0, -1, -1,  0,  1],
            [ 0,  0,  0, -1, -1]]), rational=True)
    var_list = [Z11, Z22]
    var_types = {'compact': [0], 'extended': [1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    Zsub = quantize._find_Z_instance(Z, var_list, wJ=wJ, var_types=var_types)
    ans = sym.nsimplify(sym.Matrix([
                        [-1/2, -1, -1, 1/4],
                        [ 1/2,  0,  0, 1/4],
                        [ 1/2,  0,  2, 1/4],
                        [-1/2,  1, -1, 1/4]]), rational=True)
    assert quantize._equal_up_to_column_shift_and_sign(Zsub, ans)

    # Test apriori_sol that doesn't work
    Z = sym.nsimplify(sym.Matrix([
            [-1/2,  Z11/2, Z22/2, 1/4],
            [ 1/2,      0,     0, 1/4],
            [ 1/2,      0,  -Z22, 1/4],
            [-1/2, -Z11/2, Z22/2, 1/4]]))
    wJ = sym.nsimplify(sym.Matrix([
            [ 1,  1,  0,  0,  0],
            [-1,  0,  1,  1,  0],
            [ 0, -1, -1,  0,  1],
            [ 0,  0,  0, -1, -1]]), rational=True)
    var_list = [Z11, Z22]
    var_types = {'compact': [0], 'extended': [1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    Zsub = quantize._find_Z_instance(Z, var_list, wJ=wJ, var_types=var_types, apriori_sol=4)
    assert Zsub == []

    # solution with one free variable, which is divided by 2 somewhere
    Z11 = sym.symbols("Z11")
    Z = sym.nsimplify(sym.Matrix([
    [ 1/3,  Z11/2, 1/3],
    [ 1/3, -Z11/2, 1/3],
    [-2/3,      0, 1/3]]), rational=True)
    wJ = sym.Matrix([
    [ 1,  1,  0],
    [-1,  0,  1],
    [ 0, -1, -1]])
    var_list = [Z11]
    var_types = {'compact': [0], 'extended': [1], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [2]}
    nonzero = [2, sym.simplify(-Z11/3, rational=True)]
    Zsub = quantize._find_Z_instance(Z, var_list, wJ=wJ, var_types=var_types)
    ans = sym.nsimplify(sym.Matrix([
                                [ 1/3,  1, 1/3],
                                [ 1/3, -1, 1/3],
                                [-2/3,  0, 1/3]]), rational=True)
    assert quantize._equal_up_to_column_shift_and_sign(Zsub, ans)

    # Solution that only has a single thing
    x,y = sym.symbols("x,y")
    a,b = sym.symbols("a,b")
    v = [x,y,a,b]

    Z = sym.Matrix([[x,1],
                    [y,1]])
    wJ = sym.Matrix([[1],
                     [-1]])
    Zsub = quantize._find_Z_instance(Z, v, wJ=wJ, var_types={"extended":[0], "sigma":[1]})
    
    assert sym.im(Zsub).is_zero_matrix
    wJT_trans = wJ.transpose()*Zsub
    assert all(sym.simplify(x) in range(-4,5) for x in wJT_trans)


def test_complete_graph_wjt_cost_certificate():
    """K_n incidence bases have a certified (n-1)^2 minimum L1 cost."""
    for n_vertices in (3, 4, 5):
        edges = list(itertools.combinations(range(n_vertices), 2))
        wJ = sym.zeros(n_vertices, len(edges))
        for column, (left, right) in enumerate(edges):
            wJ[left, column] = 1
            wJ[right, column] = -1

        dynamic = [sym.eye(n_vertices)[:, column]
                   for column in range(n_vertices - 1)]
        common_mode = sym.ones(n_vertices, 1)
        Z = sym.Matrix.hstack(*dynamic, common_mode)
        assert quantize._complete_graph_wjt_cost_certificate(
            Z, wJ, n_vertices - 1
        ) == (n_vertices - 1) ** 2

        # A missing edge is not a complete-graph certificate.
        assert quantize._complete_graph_wjt_cost_certificate(
            Z, wJ[:, :-1], n_vertices - 1
        ) is None

        # Nor is a final column that fails to span the common-mode kernel.
        bad_Z = Z.copy()
        bad_Z[:, -1] = sym.eye(n_vertices)[:, -1]
        assert quantize._complete_graph_wjt_cost_certificate(
            bad_Z, wJ, n_vertices - 1
        ) is None


def test_complete_graph_minimum_bases_have_one_canonical_wjt_orbit():
    """All singleton-cut K_n bases canonicalize to one nonlinear key."""
    for n_vertices in (3, 4, 5):
        edges = list(itertools.combinations(range(n_vertices), 2))
        singleton_cuts = [
            sym.Matrix([int((left == vertex) - (right == vertex))
                        for left, right in edges])
            for vertex in range(n_vertices)
        ]
        keys = {
            quantize._maximize_wT(sym.Matrix.hstack(
                *(singleton_cuts[index] for index in basis)
            ))[1]
            for basis in itertools.combinations(range(n_vertices), n_vertices - 1)
        }
        assert len(keys) == 1


def test_maximize_wT_matches_exhaustive_row_sign_search():
    """The separable row optimizer preserves the former exhaustive result."""
    def exhaustive(wT):
        wT = quantize._sort_wT(wT)
        supports = [tuple(wT[row].nonzero()[0]) for row in range(wT.shape[0])]
        segments = []
        segment = [0]
        for row in range(1, wT.shape[0]):
            if supports[row] == supports[row - 1]:
                segment.append(row)
            else:
                segments.append(segment)
                segment = [row]
        segments.append(segment)
        permutations = list(itertools.product(*[
            itertools.permutations(rows, len(rows)) for rows in segments
        ]))
        best = None
        best_value = -np.inf
        best_key = None
        for columns in itertools.product([1, -1], repeat=wT.shape[1]):
            column_signs = np.array(columns)
            for rows in itertools.product([1, -1], repeat=wT.shape[0]):
                row_signs = np.array(rows)
                modified = row_signs[:, None] * wT * column_signs[None, :]
                value = np.sum(modified)
                if value >= best_value:
                    for row_permutation in permutations:
                        order = list(itertools.chain.from_iterable(row_permutation))
                        candidate = modified[order, :].astype(int)
                        key = tuple(int(x) for x in candidate.flatten())
                        if value > best_value or (value == best_value and key > best_key):
                            best_value, best_key, best = value, key, candidate.copy()
        return best, best_key

    rng = np.random.default_rng(73)
    matrices = [
        rng.integers(-2, 3, size=(rows, columns))
        for rows, columns in ((2, 2), (3, 2), (4, 3))
        for _ in range(8)
    ]
    for matrix in matrices:
        expected_matrix, expected_key = exhaustive(matrix)
        actual_matrix, actual_key, _ = quantize._maximize_wT(matrix)
        assert np.array_equal(actual_matrix, expected_matrix)
        assert actual_key == expected_key


def test_incidence_square_nonzero_str_matches_symbolic_construction():
    rng = np.random.default_rng(101)
    for rows, columns in ((3, 4), (5, 7)):
        w = sym.Matrix(rng.integers(-1, 2, size=(rows, columns)))
        values = list(sym.symbols(f"e0:{columns}", nonzero=True))
        expected = quantize._nonzero_entries_str(
            quantize.incidence_to_square(w, values)
        )
        assert quantize._incidence_square_nonzero_str(w, values) == expected

    # Repeated coefficients deliberately retain the cancellation-aware path.
    shared = sym.Symbol("shared", nonzero=True)
    w = sym.Matrix([[1, 1], [1, -1]])
    values = [shared, shared]
    assert quantize._incidence_square_nonzero_str(w, values) == "0-0"


def test_decomposed_complete_graph_wjt_cost_certificate():
    """A bridge K2 plus an independent K4 cut space has exact cost 1+9."""
    n_vertices = 5
    edges = [
        (0, 1),
        (0, 2), (0, 3), (0, 4),
        (2, 3), (2, 4), (3, 4),
    ]
    wJ = sym.zeros(n_vertices, len(edges))
    for column, (left, right) in enumerate(edges):
        wJ[left, column] = 1
        wJ[right, column] = -1
    eye = sym.eye(n_vertices)
    bridge_scale = sym.Symbol("bridge_scale", nonzero=True)
    block_symbols = sym.symbols("block_0:9")
    block = sym.Matrix(3, 3, block_symbols)
    block_columns = [
        sym.Matrix.vstack(sym.zeros(2, 1), block[:, column])
        for column in range(3)
    ]
    Z = sym.Matrix.hstack(bridge_scale * eye[:, 1], *block_columns,
                          sym.ones(n_vertices, 1))
    assert quantize._decomposed_complete_graph_wjt_cost_certificate(
        Z, wJ, 4
    ) == 10


def test_compact_well_aligned():

    wT = sym.nsimplify(sym.Matrix([[1, 0, 1, 1],
                                    [0, -1, 0, 1],
                                    [0, 1, 0, 1],
                                    [1, 0, -1, 1]]), rational=True)
    assert quantize.compact_well_aligned(wT, 2)


    wT = sym.nsimplify(sym.Matrix([[1, 0, 1, 1],
                                    [0, -1, 0, 1],
                                    [0, 1, 0, 1],
                                    [1, 0, -1, 1]]), rational=True)
    assert not quantize.compact_well_aligned(wT, 3)

    wT = sym.nsimplify(sym.Matrix([[1, 1, 1],
                        [2/3, 1/3, 1/10],
                        [40, 90, 200]]), rational=True)
    assert not quantize.compact_well_aligned(wT, 2)


def test_incidence_to_square():

    wTest = sym.Matrix([[1, 0, 1, 1],
                        [0, -1/2, 0, 1],
                        [0, 1/2, 0, 1],
                        [1, 0, -1, 1.0]])
    ans = functools.reduce(lambda a,b: a+b, [wTest[:,i]*wTest[:,i].transpose() for i in range(4)], sym.zeros(4,4))
    wSquare = quantize.incidence_to_square(wTest, [1,1,1,1])
    assert (wSquare - ans).is_zero_matrix


# --- stand-ins for sircuitenum's circuit helpers -------------------------------------------


def node_space(circuit, edges):
    """gen_cap_mat, gen_ind_mat, gen_w and EJ for a sircuitenum circuit, from circuit_quantization."""
    lag = cq.lagrangian(to_circuit(circuit, edges))
    return sym.Matrix(lag.C_int), sym.Matrix(lag.Lstar_int), sym.Matrix(lag.omega_J_int), tuple(lag.EJ)


def sircuitenum_basis(circuit, edges):
    """sircuitenum's var_trans_basis (Z0, var_types), stored in sircuitenum_fixtures.json."""
    for key, data in FIXTURES.items():
        if data["circuit"] == [list(e) for e in circuit] and data["edges"] == [list(e) for e in edges]:
            stored = fixture(key)
            return stored["Z0"], stored["var_types"]
    raise KeyError(f"no sircuitenum fixture for {circuit} on {edges}; run make_sircuitenum_fixtures.py")


def renumber_nodes(edges):
    """Relabel nodes 0, 1, ... in sorted order, as sircuitenum.utils.renumber_nodes."""
    order = {node: i for i, node in enumerate(sorted({n for edge in edges for n in edge}))}
    return [tuple(order[n] for n in edge) for edge in edges]


def add_elem_number(circuit):
    """Number each element type in order of appearance, as sircuitenum.utils.add_elem_number."""
    counts = {}
    numbered = []
    for labels in circuit:
        edge = []
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
            edge.append(f"{label}_{counts[label]}")
        numbered.append(tuple(edge))
    return numbered


def swap_nodes(edges, a, b):
    """Exchange node labels a and b, as sircuitenum.utils.swap_nodes."""
    swap = {a: b, b: a}
    return [tuple(swap.get(n, n) for n in edge) for edge in edges]


def test_compact_alignment_transformation():

    edges = [(1, 2), (2, 3), (3, 4), (4,1)]
    edges = renumber_nodes(edges)
    circuit = [("J",), ("J","C"), ("J",), ("L",)]
    Z = sym.nsimplify((sym.Matrix([[1, 0, 1, 1],
                                    [0, -1/2, 0, 1],
                                    [0, 1/2, 0, 1],
                                    [1, 0, -1, 1]])), rational=True)
    wj = node_space(circuit, edges)[2]
    wT = wj.transpose()*Z
    n_c = 2
    for Z2 in quantize.compact_alignment_transformation(wT, n_c):
        assert quantize.compact_well_aligned(wT*Z2, n_c)

    
    for n_c in [1, 2, 3]:
        wT = sym.nsimplify(sym.Matrix([[1, 1, 1],
                        [2/3, 1/3, 1/10],
                        [40, 90, 200]]), rational=True)
        for Z2 in quantize.compact_alignment_transformation(wT, n_c):
            assert quantize.compact_well_aligned(wT*Z2, n_c)


def test_H_hash():

   
    # Misbehaving case from earlier version
    L, C, C_J = sym.symbols("L, C, C_J", positive=True)
    J_1, J_2, J_3, J_4, J_5, J_6 = sym.symbols("J_1, J_2, J_3, J_4, J_5, J_6", positive=True)
    Z = sym.Matrix([[-1, -1, -1/4, 1/4], [2, 0, -1/4, 1/4], [-1, 1, -1/4, 1/4], [0, 0, 3/4, 1/4]])
    cTransInv = sym.Matrix([[1/(6*C + 24*C_J), 0, 0], [0, 1/(2*C + 8*C_J), 0], [0, 0, 1/(3*C + 3*C_J)]])
    lTrans = sym.Matrix([[24/L, 0, 0, 0], [0, 8/L, 0, 0], [0, 0, 3/L, 0], [0, 0, 0, 0]])
    wJtTrans = sym.Matrix([[-3, -1, 0, 0], [0, -2, 0, 0], [-1, -1, -1, 0], [3, -1, 0, 0], [2, 0, -1, 0], [-1, 1, -1, 0]])
    EJ = (J_1, J_2, J_3, J_4, J_5, J_6)
    var_types = {'compact': [], 'extended': [0, 1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(hashes[0]) == '030_3-111_0_0-000_0-000'
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types, extra_nl=True)
    assert new_to_old(hashes[0]).replace("-","").replace("_","") == '030_3-111-19-3-446564475473535355_0_0-000_0-000'.replace("-","").replace("_","")

    # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    _, var_types = sircuitenum_basis(circuit, edges)
    cMat = node_space(circuit, edges)[0]
    lMat = node_space(circuit, edges)[1]
    Z = sym.Matrix([[1, 1], [0, 1]])
    wJ, EJ = node_space(circuit, edges)[2:]
    # hashes = quantize.H_hash(Z, var_types, cMat, lMat, wJ)
    cTransInv = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(hashes[0]) == "100_0-_0_0-_0-"
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1],
                                                 [0, 1]]), 
                                     [Z[:, hashes[1]]]) == [0]

    # Zero-pi
    circuit = [("J",),("J",), ("L",), ("L",), ("C",), ("C",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    _, var_types = sircuitenum_basis(circuit, edges)
    cMat = node_space(circuit, edges)[0]
    lMat = node_space(circuit, edges)[1]
    # wJ = node_space(circuit, edges)[2]
    Z = sym.Matrix([[1, 1, 1, 1],
                    [0, 0, 1, 1],
                    [0, 1, 0, 1],
                    [1, 0, 0, 1]])
    circuit = [("J1",),("J2",), ("L",), ("L",), ("C",), ("C",)]
    wJ, EJ = node_space(circuit, edges)[2:]
    # hashes = quantize.H_hash(Z, var_types, cMat, lMat, wJ)
    cTransInv = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(hashes[0]) == '111_1-100_0_0-000_0-000'
    assert quantize._find_equiv_mats(Z, [Z[:, hashes[1]]]) == [0]
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types, extra_nl=True)
    assert new_to_old(hashes[0]).replace("-","").replace("_","") == '111_1-100-4-1-5553_0_0-000_0-000'.replace("-","").replace("_","")
    assert quantize._find_equiv_mats(Z, [Z[:, hashes[1]]]) == [0]

    circuit = [("J1",),("J2",), ("L1",), ("L2",), ("C1",), ("C2",)]
    cMat = node_space(circuit, edges)[0]
    lMat = node_space(circuit, edges)[1]
    wJ, EJ = node_space(circuit, edges)[2:]
    # hashes = quantize.H_hash(Z, var_types, cMat, lMat, wJ)
    cTransInv = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    hash, _ = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(hash) == '111_1-100_4_1-001_3-111'


def test_secondary_decouple():

    # Nothing to decouple 
    circuit = [("L", "C"), ("L", "C"), ("L",), ("J",)]
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    cMat = node_space(circuit, edges)[0]
    lMat = node_space(circuit, edges)[1]
    wJ, EJ = node_space(circuit, edges)[2:]
    jMat = quantize.incidence_to_square(wJ, EJ)

    circuit = add_elem_number(circuit)
    Z0, var_types = sircuitenum_basis(circuit, edges)
    Z1 = quantize.secondary_decouple(var_types, [Z0.transpose()*jMat*Z0], [False])


    # Debugging example
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L_1", "C_1"), ("L_2", "C_2"), ("L_3",), ("J_1",)]
    cMat = node_space(circuit, edges)[0]
    lMat = node_space(circuit, edges)[1]
    edges = renumber_nodes(edges)
    wJ, EJ = node_space(circuit, edges)[2:]
    Z0, var_types = sircuitenum_basis(circuit, edges)
    Z1 = quantize.secondary_decouple(var_types, [Z0.transpose()*lMat*Z0, Z0.transpose()*cMat*Z0], [False, True])
    cT = []
    lT = []
    wT = []
    vals = []
    Z_perm = []
    for Z in Z1:
        Ztot = Z0*Z
        cInvTrans = sym.simplify((Ztot.transpose()*cMat*Ztot)[:-1, :-1].inv())
        lTrans = sym.simplify(Ztot.transpose()*lMat*Ztot)
        wJtTrans = sym.simplify(wJ.transpose()*Ztot)
        cT.append(cInvTrans)
        lT.append(lTrans)
        wT.append(wJtTrans)
        Z = quantize._find_Z_instance_deterministic(Ztot, (Ztot).free_symbols)
        Z = sym.simplify(Z)
        val, Zp = quantize.H_hash(cInvTrans, lTrans, wJtTrans, EJ, var_types)
        vals.append(val)
        Z_perm.append(Zp)
   
    assert new_to_old(vals[0]).replace("-","").replace("_", "") == '012_0-000_3_2-011_1-001'.replace("-","").replace("_", "")

    # Make sure it's not nan
    circuit, edges = ([('J',), ('J', 'L'), ('C', 'J', 'L')], [(0, 1), (0, 2), (1, 2)])
    Z0, var_types = sircuitenum_basis(circuit, edges)
    assert var_types["extended"] == [0,1]
    assert var_types["sigma"] == [2]
    assert Z0.det() != 0
    for i in range(3):
        for j in range(3):
            assert Z[i,j].is_finite
    cMat = node_space(circuit, edges)[0]
    lMat = node_space(circuit, edges)[1]
    wJ = node_space(circuit, edges)[2]
    Z2 = quantize.secondary_decouple(var_types, [Z0.transpose()*lMat*Z0, Z0.transpose()*cMat*Z0], [False, True])
    for Z in Z2:
        for i in range(3):
            for j in range(3):
                assert Z[i,j].is_finite or Z[i,j].is_finite is None


    circuit = [("C", "L1"), ("J", "L2")]
    edges = [(1, 2), (1, 3)]
    edges = renumber_nodes(edges)
    cMat = node_space(circuit, edges)[0]
    lMat = node_space(circuit, edges)[1]
    wJ = node_space(circuit, edges)[2]
    Z0 = sym.simplify(sym.Matrix([[1/2, 0, 1],
                                [0, 1, 1],
                                [-1/2, 0, 1]]), rational=True)
    var_types =  {"compact": [],
                "extended": [0],
                "harmonic": [1],
                "sigma": [2]}
    Z2 = quantize.secondary_decouple(var_types, [Z0.transpose()*lMat*Z0, Z0.transpose()*cMat*Z0], [False, True])[0]
    Z_comp = quantize._find_Z_instance_deterministic(Z2, Z2.free_symbols)
    Z_ans = sym.nsimplify(sym.Matrix([[2, 0, 0],[1, 1, 0],[0, 0, -1]]), rational=True)
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(Z_comp,
                                                                 Z_ans,
                                                                 shifts=[sym.ones(3,1)])
    Z = sym.simplify(Z0*Z_comp)
    cInvTrans = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    # cTransInv, lTrans, wJtTrans, EJ, var_types
    val, Z_perm = quantize.H_hash(cInvTrans, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(val).replace("-","").replace("_", "") == "011_0-0_0_0-0_0-0".replace("-","").replace("_", "")


# --- circuit_quantization: sircuitenum's choose_Z through choose_S1 ------------------------


def test_node_space_matrices_match_sircuitenum():
    # node_space stands in for gen_cap_mat, gen_ind_mat and gen_w; check it on every stored circuit.
    for key in FIXTURES:
        stored = fixture(key)
        cMat, lMat, wJ, EJ = node_space(stored["circuit"], stored["edges"])
        names = {str(s): s for s in cMat.free_symbols | lMat.free_symbols | set(EJ)}
        rename = {s: names[str(s)] for s in stored["cMat"].free_symbols | stored["lMat"].free_symbols | set(stored["EJ"])}
        assert sym.simplify(cMat - stored["cMat"].subs(rename)) == sym.zeros(*cMat.shape), key
        assert sym.simplify(lMat - stored["lMat"].subs(rename)) == sym.zeros(*lMat.shape), key
        assert wJ == stored["wJ"] and EJ == tuple(e.subs(rename) for e in stored["EJ"]), key


def choose(circuit, edges, return_instance=True):
    """sircuitenum's choose_Z(circuit, edges) through circuit_quantization and choose_S1."""
    from circuit_quantization.optimize import choose_S1
    lag = cq.lagrangian(to_circuit(circuit, edges))
    coords = cq.decouple(lag, *cq.mode_basis(lag), check=False)
    S1s, h_class = choose_S1(lag, coords, return_instance=return_instance)
    return lag, coords, S1s, h_class


def test_choose_S1_reproduces_choose_Z():
    # The circuits and expected H_hash classes of sircuitenum's test_choose_Z.
    squash = lambda h: new_to_old(h).replace("_", "").replace("-", "")  # noqa: E731

    # All compact
    circuit = [('C_1',), ('C_2',), ('J_1',), ('C_3', 'J_2'), ('C_4',), ('C_5',)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    *_, val = choose(circuit, edges)
    assert squash(val) == "200_0-0-2-1-5445_1_0-0_1-1".replace("_", "").replace("-", "")

    # Zero-pi
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    lag, coords, S1s, val = choose(circuit, edges)
    Z = cq.transform(lag, coords, S1s[0]).Z_full
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1], [0, 0], [0, 1], [1, 0]]), [Z[:, :2]]) == [0]
    assert squash(val) == '111_1-100-4-1-5553_0_0-000_0-000'.replace("_", "").replace("-", "")

    # Junction triangle: three tied compact alignments
    circuit, edges = [('J1',), ('J2',), ('J3',)], [(0, 1), (0, 2), (1, 2)]
    lag, coords, S1s, val = choose(circuit, edges, return_instance=False)
    assert len(S1s) == 3
    Z = [cq.transform(lag, coords, S).Z_full for S in S1s]
    assert coords.types_full == ("c", "c", "cy")
    for Zi in [sym.nsimplify(sym.Matrix([[2/3, 1/3, 1/3], [-1/3, -2/3, 1/3], [-1/3, 1/3, 1/3]]), rational=True),
               sym.nsimplify(sym.Matrix([[2/3, 1/3, 1/3], [-1/3, 1/3, 1/3], [-1/3, -2/3, 1/3]]), rational=True),
               sym.nsimplify(sym.Matrix([[1/3, 1/3, 1/3], [1/3, -2/3, 1/3], [-2/3, 1/3, 1/3]]), rational=True)]:
        # The sigma column is defined only up to the uniform node shift.
        assert len(quantize._find_equiv_mats(Zi, Z, shifts=[sym.ones(3, 1)])) > 0
    assert squash(val) == "200_1-1_1_0-0_1-1".replace("_", "").replace("-", "")

    # Inductor chain with one junction
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)]
    *_, val = choose(circuit, edges, return_instance=False)
    assert squash(val) == "012_0-000_3_2-011_1-001".replace("_", "").replace("-", "")

    # Every node labelling of one circuit gives the same junction structure
    circuit = add_elem_number([("J", "L"), ("C", "J"), ("C", "J", "L")])
    edges_og = ((0, 1), (0, 2), (1, 2))
    edges_all = [edges_og] + [tuple(swap_nodes(edges_og, *p)) for p in itertools.combinations([0, 1, 2], r=2)]
    res = {}
    for edges in edges_all:
        lag, coords, S1s, val = choose(circuit, edges)
        new = cq.transform(lag, coords, S1s[0])
        assert new.types_full == ("e", "e", "cy")
        wJ = node_space(circuit, edges)[2]
        res[edges] = frozenset({quantize._maximize_wT((wJ.transpose() * new.Z_full)[:, :2])[1]})
    assert len(set(res.values())) == 1

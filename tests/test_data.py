import numpy as np
from numpy import ma
from typing import List, Tuple

from scipy.sparse import coo_matrix
from monstermatch.data import _get_data, to_typed_list, remap, data, masked_array_from_ratings, Rating, \
    nan_masked_from_ratings, generate_w_h, generate_array_data_file
from sklearn.decomposition import NMF


def test_correctly_prepare():
    _column_1 = _get_data()
    # Convert to something typed
    _column_2 = to_typed_list(_column_1)
    assert len(_column_2) < len(_column_1)
    _column_3 = remap(_column_2, len(_column_1))
    # Make sure we didn't miss anything
    _assert_1 = list((a.index, a.monster_type.value) in enumerate(_column_1) for a in _column_2)
    assert len(_column_3) == len(_column_1)
    assert all(_assert_1)


def test_ratings_valid_format():
    def coo_matrix_wrapper(vals: List[Tuple[int, int, float]], rows: int, columns: int) -> coo_matrix:
        row, column, _vals = list(zip(*vals))
        return coo_matrix((_vals, (row, column)), shape=(rows, columns))

    _matrix_1 = coo_matrix_wrapper([(0, 3, 1.0), (5, 3, 0.5)], rows=100, columns=10).tolil()
    assert _matrix_1[0, 3] == 1.0
    assert _matrix_1[5, 3] == 0.5
    assert _matrix_1[4, 4] == 0.0

    _matrix_2 = masked_array_from_ratings([Rating(0, 3, 1.0), Rating(5, 3, 0.5)], rows=100, columns=10)
    assert _matrix_2[0, 3] == 1.0
    assert _matrix_2[5, 3] == 0.5
    assert _matrix_2[4, 4] is ma.masked

    _matrix_3 = nan_masked_from_ratings([Rating(0, 3, 1.0), Rating(5, 3, 0.5)], rows=100, columns=10)
    assert _matrix_3[0, 3] == 1.0
    assert _matrix_3[5, 3] == 0.5
    assert np.isnan(_matrix_3[4, 4])


def test_nmf_on_masked_ratings():
    _matrix_1 = nan_masked_from_ratings([Rating(0, 3, 1.0), Rating(5, 3, 0.5), Rating(5, 8, 1.0)], rows=10, columns=10)
    _nmf = NMF(solver='mu', init='random', n_components=2)
    W = _nmf.fit_transform(_matrix_1)
    X = _nmf.inverse_transform(W)
    assert np.shape(X) == np.shape(_matrix_1)
    recommended = np.argsort(X[0])
    assert recommended[-1] == 8
    assert recommended[-2] == 3


def test_perfect_separation_of_latents():
    latents = data()
    W, H, mapping = generate_w_h(latents, n_users=100)
    nmf = NMF(solver='mu', init='custom', n_components=3)
    nmf.components_ = H
    nmf.n_components_ = H.shape[0]
    X = nmf.inverse_transform(W)


def test_print_csharp():
    generate_array_data_file()

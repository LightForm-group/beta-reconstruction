import numpy as np
from defdap.quat import Quat

from beta_reconstruction.crystal_relations import (
    unq_hex_syms, unq_cub_syms, burg_trans
)


def test_unq_hex_syms_type():
    assert type(unq_hex_syms) is list
    assert len(unq_hex_syms) == 6
    assert all([type(sym) is Quat for sym in unq_hex_syms])


def test_unq_hex_syms_value():
    expected_comps = [
        np.array([1.,         0.,         0.,  0.]),
        np.array([0.5,        0.,         0.,  0.8660254]),
        np.array([0.8660254,  0.,         0.,  0.5]),
        np.array([0.,         0.,         1.,  0.]),
        np.array([0.,         0.8660254, -0.5, 0.]),
        np.array([0.,        -0.8660254, -0.5, 0.])
    ]

    assert all([np.allclose(sym.quatCoef, row) for sym, row
                in zip(unq_hex_syms, expected_comps)])


def test_unq_cub_syms_type():
    assert type(unq_cub_syms) is list
    assert len(unq_cub_syms) == 12
    assert all([type(sym) is Quat for sym in unq_cub_syms])


def test_unq_cub_syms_value():
    expected_comps = [
        np.array([1.,          0.,          0.,          0.]),
        np.array([0.70710678,  0.,          0.,          0.70710678]),
        np.array([0.70710678,  0.,          0.,         -0.70710678]),
        np.array([0.70710678,  0.70710678,  0.,          0.]),
        np.array([0.5,         0.5,         0.5,        -0.5]),
        np.array([0.5,         0.5,         0.5,         0.5]),
        np.array([0.,          0.70710678,  0.,          0.70710678]),
        np.array([0.,          0.,         -0.70710678,  0.70710678]),
        np.array([0.70710678,  0.,          0.70710678,  0.]),
        np.array([0.,          0.,          0.,          1.]),
        np.array([0.5,        -0.5,         0.5,        -0.5]),
        np.array([0.5,         0.5,        -0.5,         0.5])
    ]

    assert all([np.allclose(sym.quatCoef, row) for sym, row
                in zip(unq_cub_syms, expected_comps)])


def test_burg_trans_type():
    assert type(burg_trans) is Quat


def test_burg_trans_value():
    expected_comps = np.array([0.30028953, 0.24033652, 0.66501004, 0.64017670])

    assert np.allclose(burg_trans.quatCoef, expected_comps)

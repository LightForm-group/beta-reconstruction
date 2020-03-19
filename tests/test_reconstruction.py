import pytest

import beta_reconstruction.reconstruction as recon

import numpy as np
from defdap.quat import Quat


@pytest.fixture
def ori_single_valid():
    return Quat(0.22484510, 0.45464871, -0.70807342, 0.49129550)


@pytest.fixture
def ori_list_valid():
    return [
        Quat(0.22484510, 0.45464871, -0.70807342, 0.49129550),
        Quat(0.36520321, 0.25903472, -0.40342268, 0.79798357)
    ]


def test_calculate_beta_oris_return_type(ori_single_valid):
    beta_oris = recon.calc_beta_oris(ori_single_valid)

    assert type(beta_oris) is list
    assert len(beta_oris) == 6
    assert all([type(ori) is Quat for ori in beta_oris])


def test_calculate_beta_oris_calc(ori_single_valid):
    beta_oris = recon.calc_beta_oris(ori_single_valid)

    expected_comps = [
        np.array([0.11460994,  0.97057328,  0.10987647, -0.18105038]),
        np.array([0.71894262,  0.52597293, -0.12611599,  0.43654181]),
        np.array([0.48125180,  0.86403135, -0.00937589,  0.14750805]),
        np.array([0.23608204, -0.12857975,  0.96217918,  0.04408791]),
        np.array([0.39243229, -0.10727847, -0.59866151, -0.68999466]),
        np.array([0.62851433, -0.23585823,  0.36351768, -0.64590675])
    ]

    assert all([np.allclose(quat.quatCoef, row) for quat, row
                in zip(beta_oris, expected_comps)])


def test_construct_quat_comps_return_type(ori_list_valid):
    quat_comps = recon.construct_quat_comps(ori_list_valid)

    assert type(quat_comps) is np.ndarray
    assert quat_comps.shape == (4, len(ori_list_valid))


def test_construct_quat_comps_calc(ori_list_valid):
    quat_comps = recon.construct_quat_comps(ori_list_valid)

    expected_comps = np.array([
        [0.2248451, 0.45464871, -0.70807342, 0.4912955],
        [0.36520321, 0.25903472, -0.40342268, 0.79798357]
    ]).T

    assert np.allclose(quat_comps, expected_comps)

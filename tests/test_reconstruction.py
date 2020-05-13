from typing import List

import pytest
import numpy as np
from defdap.quat import Quat

import beta_reconstruction.reconstruction as recon


@pytest.fixture
def ori_single_valid() -> Quat:
    """A single sample quaternion representing the orientation of a grain."""
    return Quat(0.22484510, 0.45464871, -0.70807342, 0.49129550)


@pytest.fixture
def ori_single_valid_2() -> Quat:
    """A single sample quaternion representing the orientation of a grain."""
    return Quat(0.11939881, -0.36445855, -0.67237386, -0.63310922)


@pytest.fixture
def ori_single_valid_3() -> Quat:
    """A single sample quaternion representing the orientation of a grain."""
    return Quat(0.8730071, -0.41360125, -0.02295757, -0.25742097)


@pytest.fixture
def ori_quat_list_valid() -> List[Quat]:
    """A list of sample quaternion representing the orientations of grains."""
    return [
        Quat(0.22484510, 0.45464871, -0.70807342, 0.49129550),
        Quat(0.36520321, 0.25903472, -0.40342268, 0.79798357)
    ]


@pytest.fixture
def unq_cub_sym_comps():
    """The quaternion values of the 12 unique symmetries for cubic systems."""
    return np.array([
        [1., 0., 0., 0.],
        [0.70710678, 0., 0., 0.70710678],
        [0.70710678, 0., 0., -0.70710678],
        [0.70710678, 0.70710678, 0., 0.],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0., 0.70710678, 0., 0.70710678],
        [0., 0., -0.70710678, 0.70710678],
        [0.70710678, 0., 0.70710678, 0.],
        [0., 0., 0., 1.],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, 0.5]
    ]).T


@pytest.fixture
def beta_oris() -> List[Quat]:
    """The 6 possible beta orientations for the `ori_single_valid` fixture."""
    return [
        Quat(0.11460994,  0.97057328,  0.10987647, -0.18105038),
        Quat(0.71894262,  0.52597293, -0.12611599,  0.43654181),
        Quat(0.4812518,   0.86403135, -0.00937589,  0.14750805),
        Quat(0.23608204, -0.12857975,  0.96217918,  0.04408791),
        Quat(0.39243229, -0.10727847, -0.59866151, -0.68999466),
        Quat(0.62851433, -0.23585823,  0.36351768, -0.64590675)
    ]


class TestCalcBetaOris:
    """Tests for the calculation of possible beta orientations from a given alpha orientation."""
    @staticmethod
    def test_return_type(ori_single_valid: Quat):
        """Test the expected return types from this function."""
        beta_oris = recon.calc_beta_oris(ori_single_valid)

        assert type(beta_oris) is list
        assert len(beta_oris) == 6
        assert all([type(ori) is Quat for ori in beta_oris])

    @staticmethod
    def test_calc(ori_single_valid: Quat):
        """Test with a sample Quaternion to a known good output."""
        beta_oris = recon.calc_beta_oris(ori_single_valid)

        expected_comps = [
            np.array([0.11460994, 0.97057328, 0.10987647, -0.18105038]),
            np.array([0.71894262, 0.52597293, -0.12611599, 0.43654181]),
            np.array([0.48125180, 0.86403135, -0.00937589, 0.14750805]),
            np.array([0.23608204, -0.12857975, 0.96217918, 0.04408791]),
            np.array([0.39243229, -0.10727847, -0.59866151, -0.68999466]),
            np.array([0.62851433, -0.23585823, 0.36351768, -0.64590675])
        ]

        assert all([np.allclose(quat.quatCoef, row) for quat, row
                    in zip(beta_oris, expected_comps)])


class TestConstructQuatComps:
    """Test the method that returns a NumPy array from a list of Quats."""
    @staticmethod
    def test_return_type(ori_quat_list_valid):
        quat_comps = recon.construct_quat_comps(ori_quat_list_valid)

        assert type(quat_comps) is np.ndarray
        assert quat_comps.shape == (4, len(ori_quat_list_valid))

    @staticmethod
    def test_calc(ori_quat_list_valid):
        quat_comps = recon.construct_quat_comps(ori_quat_list_valid)

        expected_comps = np.array([
            [0.22484510, 0.45464871, -0.70807342, 0.49129550],
            [0.36520321, 0.25903472, -0.40342268, 0.79798357]
        ]).T

        assert np.allclose(quat_comps, expected_comps)


class TestBetaOrientationsFromCubicSymmetry:
    @staticmethod
    def test_return_1_ori_return_type(ori_single_valid: Quat):

        for cubic_symmetry_index in range(0, 9):
            beta_oris = recon.beta_oris_from_cub_sym(ori_single_valid, cubic_symmetry_index, 0)

            assert type(beta_oris) is list
            assert len(beta_oris) == 1
            assert all([type(ori) is Quat for ori in beta_oris])

    @staticmethod
    def test_return_1_ori_calc(ori_single_valid: Quat):
        expected_comps = np.array([0.11460994, 0.97057328, 0.10987647, -0.18105038])

        for cubic_symmetry_index in range(0, 9):
            beta_oris = recon.beta_oris_from_cub_sym(ori_single_valid, cubic_symmetry_index, 0)

            assert np.allclose(beta_oris[0].quatCoef, expected_comps)

    @staticmethod
    def test_return_3_ori_return_type(ori_single_valid: Quat):
        beta_oris = recon.beta_oris_from_cub_sym(ori_single_valid, 9, 1)

        assert type(beta_oris) is list
        assert len(beta_oris) == 3
        assert all([type(ori) is Quat for ori in beta_oris])

    @staticmethod
    def test_return_3_ori_calc(ori_single_valid: Quat):
        expected_comps = [
            np.array([0.58944381, -0.19811007, -0.13576035, -0.77128304]),
            np.array([0.09026886, 0.01229830, -0.90115179, -0.42382277]),
            np.array([0.39243229, -0.10727847, -0.59866151, -0.68999466])
        ]

        beta_oris = recon.beta_oris_from_cub_sym(ori_single_valid, 9, 1)

        assert all([np.allclose(quat.quatCoef, row) for quat, row
                    in zip(beta_oris, expected_comps)])

    @staticmethod
    def test_return_2_ori_return_type(ori_single_valid: Quat):

        for cubic_symmetry_index in range(10, 12):
            beta_oris = recon.beta_oris_from_cub_sym(
                ori_single_valid, cubic_symmetry_index, 2
            )

            assert type(beta_oris) is list
            assert len(beta_oris) == 2
            assert all([type(ori) is Quat for ori in beta_oris])

    @staticmethod
    def test_return_2_ori_calc(ori_single_valid: Quat):
        expected_comps = [
            np.array([0.23608204, -0.12857975, 0.96217918, 0.04408791]),
            np.array([0.60433267, -0.44460035, -0.23599247, 0.61759219])
        ]

        for cubic_symmetry_index in range(10, 12):
            beta_oris = recon.beta_oris_from_cub_sym(
                ori_single_valid, cubic_symmetry_index, 2
            )

            assert all([np.allclose(quat.quatCoef, row) for quat, row
                        in zip(beta_oris, expected_comps)])

    @staticmethod
    def test_invalid_cub_sym_idx(ori_single_valid: Quat):
        with pytest.raises(ValueError):
            recon.beta_oris_from_cub_sym(ori_single_valid, -1, 0)

        with pytest.raises(ValueError):
            recon.beta_oris_from_cub_sym(ori_single_valid, 12, 0)

    @staticmethod
    def test_invalid_hex_sym_idx(ori_single_valid: Quat):
        with pytest.raises(ValueError):
            recon.beta_oris_from_cub_sym(ori_single_valid, 0, -1)

        with pytest.raises(ValueError):
            recon.beta_oris_from_cub_sym(ori_single_valid, 0, 12)


class TestCalcMisorientationOfVariants:
    @staticmethod
    def test_return_type(ori_single_valid_2: Quat, ori_single_valid_3: Quat,
                                                 unq_cub_sym_comps: np.ndarray):
        rtn_data = recon.calc_misori_of_variants(
            ori_single_valid_2.conjugate, ori_single_valid_3, unq_cub_sym_comps
        )

        assert len(rtn_data) == 2
        assert type(rtn_data[0]) is np.ndarray
        assert type(rtn_data[1]) is np.ndarray
        assert rtn_data[0].shape == (12, 12)
        assert rtn_data[1].shape == (12, 12)
        # assert rtn_data[0].dtype is float
        # assert rtn_data[1].dtype is np.uint8

    @staticmethod
    def test_calc(ori_single_valid_2: Quat, ori_single_valid_3: Quat,
                                          unq_cub_sym_comps: np.ndarray):
        min_misoris, min_cub_sym_idx = recon.calc_misori_of_variants(
            ori_single_valid_2.conjugate, ori_single_valid_3, unq_cub_sym_comps
        )

        expected_misoris = np.array([
            [72.10987323, 76.36096945, 53.98385462, 45.75704265,
             83.32604373, 60.50901377, 17.52239929, 60.61408066,
             60.19692765, 10.04164061, 61.53155245, 82.82891781],
            [50.17545655, 60.49509552, 75.77768501, 62.82079145,
             9.82849352, 59.81098561, 90.4698601, 71.02858613,
             59.30565985, 90.0287142, 61.14941712, 0.74061716],
            [62.82079145, 75.77768501, 60.49509552, 50.17545655,
             90.4698601, 71.02858613, 9.82849352, 59.81098561,
             61.14941712, 0.74061716, 59.30565985, 90.0287142],
            [45.75704265, 53.98385462, 76.36096945, 72.10987323,
             17.52239929, 60.61408066, 83.32604373, 60.50901377,
             61.53155245, 82.82891781, 60.19692765, 10.04164061],
            [76.48518832, 63.52975385, 48.77265821, 59.29252391,
             60.55064888, 0.75472565, 60.89890833, 89.54040843,
             11.23103549, 70.02838269, 90.30035319, 59.81093202],
            [85.62468809, 30.52316425, 88.56382757, 30.77323002,
             45.49742131, 59.49571592, 71.86764209, 120.50745327,
             53.74045499, 125.61812062, 70.53356882, 49.97637185],
            [59.29252391, 48.77265821, 63.52975385, 76.48518832,
             60.89890833, 89.54040843, 60.55064888, 0.75472565,
             90.30035319, 59.81093202, 11.23103549, 70.02838269],
            [30.77323002, 88.56382757, 30.52316425, 85.62468809,
             71.86764209, 120.50745327, 45.49742131, 59.49571592,
             70.53356882, 49.97637185, 53.74045499, 125.61812062],
            [30.42384486, 84.27216389, 31.11038499, 88.94283251,
             69.39128472, 125.34083175, 54.34494432, 48.97632448,
             73.06663122, 60.21188602, 44.71543453, 119.50725922],
            [77.10707819, 72.8271483, 44.45005165, 54.10304781,
             61.40692291, 11.0408083, 60.19692765, 82.42319965,
             18.90882407, 59.50881108, 83.19209904, 60.67666977],
            [88.94283251, 31.11038499, 84.27216389, 30.42384486,
             54.34494432, 48.97632448, 69.39128472, 125.34083175,
             44.71543453, 119.50725922, 73.06663122, 60.21188602],
            [54.10304781, 44.45005165, 72.8271483, 77.10707819,
             60.19692765, 82.42319965, 61.40692291, 11.0408083,
             83.19209904, 60.67666977, 18.90882407, 59.50881108]
        ]) / 180 * np.pi
        expected_sym_idxs = np.array([
            [1, 7, 3, 6, 8, 10, 6, 5, 0, 4, 7, 11],
            [3, 2, 8, 1, 3, 11, 7, 4, 10, 5, 9, 10],
            [2, 3, 1, 8, 6, 5, 8, 10, 9, 11, 11, 4],
            [7, 8, 6, 2, 7, 4, 3, 11, 6, 10, 0, 5],
            [8, 1, 3, 10, 8, 10, 9, 9, 3, 4, 7, 11],
            [7, 1, 6, 2, 7, 10, 2, 11, 5, 6, 1, 5],
            [11, 8, 2, 3, 9, 9, 3, 11, 6, 10, 8, 5],
            [1, 7, 2, 6, 1, 10, 6, 11, 2, 4, 4, 7],
            [2, 6, 1, 7, 2, 2, 4, 4, 1, 1, 6, 10],
            [6, 2, 7, 8, 6, 5, 0, 0, 7, 11, 3, 4],
            [6, 2, 7, 1, 5, 5, 1, 1, 7, 11, 2, 2],
            [3, 6, 1, 7, 0, 0, 7, 4, 8, 5, 6, 10]
        ])

        assert np.allclose(min_misoris, expected_misoris)
        assert np.all(min_cub_sym_idx == expected_sym_idxs)


class TestCountBetaVariants:
    @staticmethod
    def test_return_type(beta_oris: List[Quat]):
        ori_tol = 5.
        possible_beta_oris = [[beta_oris[0]]]
        variant_count = recon.count_beta_variants(beta_oris, possible_beta_oris,
                                                  1, ori_tol)

        assert type(variant_count) is np.ndarray
        assert len(variant_count) == 6
        assert variant_count.dtype == int

    @staticmethod
    def test_good_count(beta_oris: List[Quat]):
        ori_tol = 5.
        possible_beta_oris = [
            [beta_oris[0] * Quat.fromAxisAngle(np.array([1, 0, 0]),
                                               0.9*ori_tol*np.pi/180)],
            [beta_oris[1]],
            [beta_oris[1]],
            [beta_oris[1], beta_oris[3], beta_oris[4]]
        ]
        variant_count = recon.count_beta_variants(
            beta_oris, possible_beta_oris, 1, ori_tol
        )

        expected_variant_count = [1, 3, 0, 1, 1, 0]

        assert all(np.equal(variant_count, expected_variant_count))

    @staticmethod
    def test_1_bad_ori(beta_oris: List[Quat]):
        ori_tol = 5.
        possible_beta_oris = [
            [beta_oris[0] * Quat.fromAxisAngle(np.array([1, 0, 0]),
                                               1.1*ori_tol*np.pi/180)],
            [beta_oris[1]],
            [beta_oris[1]],
            [beta_oris[1], beta_oris[3], beta_oris[4]]
        ]
        with pytest.warns(UserWarning):
            variant_count = recon.count_beta_variants(
                beta_oris, possible_beta_oris, 1, ori_tol
            )

        expected_variant_count = [0, 3, 0, 1, 1, 0]

        assert all(np.equal(variant_count, expected_variant_count))

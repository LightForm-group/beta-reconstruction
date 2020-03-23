import numpy as np

# from defdap.quat import Quat

from beta_reconstruction.crystal_relations import (
    unq_hex_syms, hex_syms, unq_cub_syms, burg_trans
)


def calc_beta_oris(alpha_ori):
    """Calculate the possible beta orientations for a given alpha
    orientation using the Burgers relation and crystal symmetries.

    Parameters
    ----------
    alpha_ori : defdap.Quat.quat
        Orientation of an alpha grain

    Returns
    -------
    beta_oris : list of defdap.Quat.quat
        List of possible beta orientations
    """
    beta_oris = []

    for sym in unq_hex_syms:
        beta_oris.append(burg_trans * sym.conjugate * alpha_ori)

    return beta_oris


def construct_quat_comps(oris):
    """Construct an array of the quaternion components from input list

    Parameters
    ----------
    oris : list of defdap.Quat.quat (or other 1D enumerable type)
        Orientations to return the quaternion components of

    Returns
    -------
    quat_comps : np.ndarray
        Array of quaternion components, shape (4, n)

    """
    quat_comps = np.empty((4, len(oris)))
    for i, ori in enumerate(oris):
        quat_comps[:, i] = ori.quatCoef

    return quat_comps


def report_progress(curr, total):
    """Report the progress of the reconstruction process

    Parameters
    ----------
    curr : int
        Index of current grain
    total : int
        Total number of grains
    """
    int(round(total / 100))
    if curr % int(round(total / 100)) == 0:
        print("\r Done {:} %".format(int(curr / total * 100)), end="")


def beta_oris_from_cub_sym(alpha_ori, unq_cub_sym_idx, hex_sym_idx):
    """

    Parameters
    ----------
    alpha_ori : defdap.quat.Quat
    unq_cub_sym_idx : int
    hex_sym_idx : int

    Returns
    -------
    beta_oris : list of defdap.Quat.quat
        Possible beta orientations from given symmetries

    """
    if not (0 <= unq_cub_sym_idx <= 11):
        raise ValueError("unq_cub_sym_idx must be between 0 and 11 inclusive")
    if not (0 <= hex_sym_idx <= 11):
        raise ValueError("hex_sym_idx must be between 0 and 11 inclusive")

    beta_oris = []

    beta_ori_base = hex_syms[hex_sym_idx].conjugate * alpha_ori

    # all cases have one possible beta orientation
    beta_oris.append(burg_trans * beta_ori_base)

    if unq_cub_sym_idx == 9:
        # two extra possible beta orientations
        # B - unq_hex_syms[1] is C^+_3z:
        beta_oris.append(
            burg_trans * unq_hex_syms[1].conjugate * beta_ori_base
        )
        # C - unq_hex_syms[2] is C^+_6z:
        beta_oris.append(
            burg_trans * unq_hex_syms[2].conjugate * beta_ori_base
        )

    if unq_cub_sym_idx > 9:
        # one extra possible beta orientations
        # D - unq_hex_syms[4] is C'_22:
        beta_oris.append(
            burg_trans * unq_hex_syms[4].conjugate * beta_ori_base
        )

    return beta_oris


def calc_misori_of_variants(alpha_ori_inv, neighbour_ori, unq_cub_sym_comps):
    """Calculate all possible sym variants for disorientaion between two
    orientaions undergoing a Burgers type transformation. Then calculate
    the misorioritation to the nearest cubic symmetry, this is the deviation
    to a perfect Burgers transformation.

    Parameters
    ----------
    alpha_ori_inv : defdap.Quat.quat
        Inverse of first orientation
    neighbour_ori : defdap.Quat.quat
        Second orientation
    unq_cub_sym_comps: np.ndaray
        Components of the unique cubic symmetries

    Returns
    -------
    min_misoris : np.ndarry

    min_cub_sym_idx : np.ndarry

    """
    a2a1inv = neighbour_ori * alpha_ori_inv

    misoris = np.zeros((144, 12))
    # calculate all possible S^B_m (eqn 11. from [1]) from the
    # measured misorientation from 2 neighbour alpha grains
    # for each S^B_m calculate the 'closest' cubic symmetry
    # (from reduced subset) and the deviation from this symmetry
    for i, hex_sym in enumerate(hex_syms):
        dummy = a2a1inv * hex_sym

        for j, hex_sym_2 in enumerate(hex_syms):
            beta_var = burg_trans * ((hex_sym_2.conjugate * dummy) * burg_trans.conjugate)

            misoris[12*i + j] = np.einsum("ij,i->j", unq_cub_sym_comps,
                                          beta_var.quatCoef)

    misoris = np.abs(misoris)
    misoris[misoris > 1] = 1.
    misoris = 2 * np.arccos(misoris) * 180 / np.pi

    min_cub_sym_idx = np.argmin(misoris, axis=1)
    min_misoris = misoris[np.arange(144), min_cub_sym_idx]

    min_cub_sym_idx = min_cub_sym_idx.reshape((12, 12))
    min_misoris = min_misoris.reshape((12, 12))

    return min_misoris, min_cub_sym_idx


def calc_beta_oris_from_misori(alpha_ori, neighbour_oris, burg_tol=5.):
    """Calculate the possible beta orientations for a given alpha
    orientation using the misorientaion relation to neighbour orientations.

    Parameters
    ----------
    alpha_ori : defdap.Quat.quat

    neighbour_oris : list of defdap.Quat.quat

    burg_tol : flaot

    Returns
    -------
    beta_oris : list of lists of defdap.Quat.quat
        Possible beta orientations, grouped by each neighbour. Any
        neighbour with deviation greater than the tolerance is excluded.
    beta_devs :  list of float
        Deviations from perfect Burgers transformation

    """
    # This needed to move further up calculation process
    unq_cub_sym_comps = construct_quat_comps(unq_cub_syms)

    alpha_ori_inv = alpha_ori.conjugate

    beta_oris = []
    beta_devs = []

    for neighbour_ori in neighbour_oris:

        min_misoris, min_cub_sym_idxs = calc_misori_of_variants(
            alpha_ori_inv, neighbour_ori, unq_cub_sym_comps
        )

        # find the hex symmetries (i, j) from give the minimum
        # deviation from the burgers relation for the minimum store:
        # the deviation, the hex symmetries (i, j) and the cubic
        # symmetry if the deviation is over a threshold then set
        # cubic symmetry to -1
        min_misori_idx = np.unravel_index(np.argmin(min_misoris), min_misoris.shape)
        burg_dev = min_misoris[min_misori_idx]

        if burg_dev < burg_tol:
            beta_oris.append(beta_oris_from_cub_sym(
                alpha_ori, min_cub_sym_idxs[min_misori_idx], min_misori_idx[0]
            ))
            beta_devs.append(burg_dev)

    return beta_oris, beta_devs


def do_reconstruction(ebsd_map, burg_tol=5., ori_tol=3.):
    """Apply beta reconstruction to a ebsd map object. Nothing is returned
    and output is stored directly in the ebsd map (this should probably change)

    Parameters
    ----------
    ebsd_map: dedap.ebsd.Map
        EBSD map to apply reconstruction to
    burg_tol: float
        Maximum deviation from the Burgers relation to allow (degrees)
    ori_tol: float
        Maximum deviation from a beta orientaion (degrees)
    """
    num_grains = len(ebsd_map)
    for grain_id, grain in enumerate(ebsd_map):
        report_progress(grain_id, num_grains)

        grain.betaOris = calc_beta_oris(grain.refOri)

        neighbour_ids = list(ebsd_map.neighbourNetwork.neighbors(grain_id))
        neighbour_oris = [ebsd_map[i].refOri for i in neighbour_ids]

        grain.possibleBetaOris, grain.betaDeviations = calc_beta_oris_from_misori(
            grain.refOri, neighbour_oris, burg_tol=burg_tol
        )

        # do all the accounting stuff
        # divide 2 because of 2* in misorientation
        ori_tol = np.cos(ori_tol / 2 * np.pi / 180.)

        allPossibleBetaOris = [item for sublist in grain.possibleBetaOris for
                               item in sublist]
        uniqueBetaOris = []
        countBetaOris = []
        variantIndexes = []

        for ori in allPossibleBetaOris:
            found = False
            for i, uniqueOri in enumerate(uniqueBetaOris):
                misOri = ori.misOri(uniqueOri, "cubic")
                if misOri > ori_tol:
                    found = True
                    countBetaOris[i] += 1

            if not found:
                uniqueBetaOris.append(ori)
                countBetaOris.append(1)

                for i, betaVariant in enumerate(grain.betaOris):
                    misOri = ori.misOri(betaVariant, "cubic")
                    if misOri > ori_tol:
                        variantIndexes.append(i)
                        break
                else:
                    variantIndexes.append(-1)
                    print("Couldn't find beta variant. Grain {:}".format(
                        grain_id))

        variantCount = [0, 0, 0, 0, 0, 0]
        for i in range(len(variantIndexes)):
            if i > -1:
                variantCount[variantIndexes[i]] = countBetaOris[i]

        grain.variantCount = variantCount

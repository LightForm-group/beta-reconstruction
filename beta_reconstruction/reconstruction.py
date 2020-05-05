import numpy as np
import warnings

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
    min_misoris : np.ndarry shape (12, 12)

    min_cub_sym_idx : np.ndarry shape (12, 12)

    """
    # calculate all possible S^B_m (eqn 11. from [1]) from the
    # measured misorientation from 2 neighbour alpha grains
    # for each S^B_m calculate the 'closest' cubic symmetry
    # (from reduced subset) and the deviation from this symmetry

    # Vectorised calculation of:
    # hex_sym[j].inv * ((neighbour_ori * alpha_ori_inv) * hex_sym[i])
    # labelled: d = h2.inv * (c * h1)
    hex_sym_comps = construct_quat_comps(hex_syms)
    c = (neighbour_ori * alpha_ori_inv).quatCoef
    h1 = np.repeat(hex_sym_comps, 12, axis=1)  # outer loop
    h2 = np.tile(hex_sym_comps, (1, 12))  # inner loop
    d = np.zeros_like(h1)

    c_dot_h1 = c[1]*h1[1] + c[2]*h1[2] + c[3]*h1[3]
    c_dot_h2 = c[1]*h2[1] + c[2]*h2[2] + c[3]*h2[3]
    h1_dot_h2 = h1[1]*h2[1] + h1[2]*h2[2] + h1[3]*h2[3]

    d[0] = (c[0]*h1[0]*h2[0] - h2[0]*c_dot_h1 +
            c[0]*h1_dot_h2 + h1[0]*c_dot_h2 +
            h2[1] * (c[2]*h1[3] - c[3]*h1[2]) +
            h2[2] * (c[3]*h1[1] - c[1]*h1[3]) +
            h2[3] * (c[1]*h1[2] - c[2]*h1[1]))
    d[1] = (c[0]*h2[0]*h1[1] + h1[0]*h2[0]*c[1] - c[0]*h1[0]*h2[1] +
            c_dot_h1*h2[1] + c_dot_h2*h1[1] - h1_dot_h2*c[1] +
            h2[0] * (c[2]*h1[3] - c[3]*h1[2]) +
            c[0] * (h1[2]*h2[3] - h1[3]*h2[2]) +
            h1[0] * (c[2]*h2[3] - c[3]*h2[2]))
    d[2] = (c[0]*h2[0]*h1[2] + h1[0]*h2[0]*c[2] - c[0]*h1[0]*h2[2] +
            c_dot_h1*h2[2] + c_dot_h2*h1[2] - h1_dot_h2*c[2] +
            h2[0] * (c[3]*h1[1] - c[1]*h1[3]) +
            c[0] * (h1[3]*h2[1] - h1[1]*h2[3]) +
            h1[0] * (c[3]*h2[1] - c[1]*h2[3]))
    d[3] = (c[0]*h2[0]*h1[3] + h1[0]*h2[0]*c[3] - c[0]*h1[0]*h2[3] +
            c_dot_h1*h2[3] + c_dot_h2*h1[3] - h1_dot_h2*c[3] +
            h2[0] * (c[1]*h1[2] - c[2] * h1[1]) +
            c[0] * (h1[1]*h2[2] - h1[2] * h2[1]) +
            h1[0] * (c[1]*h2[2] - c[2] * h2[1]))

    # Vectorised calculation of:
    # burg_trans * (d * burg_trans.inv)
    # labelled: beta_vars = b * (c * b.inv)
    b = burg_trans.quatCoef
    beta_vars = np.zeros_like(h1)

    b_dot_b = b[1]*b[1] + b[2]*b[2] + b[3]*b[3]
    b_dot_d = b[1]*d[1] + b[2]*d[2] + b[3]*d[3]

    beta_vars[0] = d[0] * (b[0]*b[0] + b_dot_b)
    beta_vars[1] = (d[1] * (b[0]*b[0] - b_dot_b) + 2*b_dot_d*b[1] +
                    2*b[0] * (b[2]*d[3] - b[3]*d[2]))
    beta_vars[2] = (d[2] * (b[0]*b[0] - b_dot_b) + 2*b_dot_d*b[2] +
                    2*b[0] * (b[3]*d[1] - b[1]*d[3]))
    beta_vars[3] = (d[3] * (b[0]*b[0] - b_dot_b) + 2*b_dot_d*b[3] +
                    2*b[0] * (b[1]*d[2] - b[2]*d[1]))

    # calculate misorientation to each of the cubic symmetries
    misoris = np.einsum("ij,ik->jk", beta_vars, unq_cub_sym_comps)
    misoris = np.abs(misoris)
    misoris[misoris > 1] = 1.
    misoris = 2 * np.arccos(misoris)

    # find the cubic symmetry with minimum misorientation for each of
    # the beta misorientation variants
    min_cub_sym_idx = np.argmin(misoris, axis=1)
    min_misoris = misoris[np.arange(144), min_cub_sym_idx]
    # reshape to 12 x 12 for each of the hex sym multiplications
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
    burg_tol *= np.pi / 180.
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
        min_misori_idx = np.unravel_index(np.argmin(min_misoris),
                                          min_misoris.shape)
        burg_dev = min_misoris[min_misori_idx]

        if burg_dev < burg_tol:
            beta_oris.append(beta_oris_from_cub_sym(
                alpha_ori, min_cub_sym_idxs[min_misori_idx], min_misori_idx[0]
            ))
            beta_devs.append(burg_dev)

    return beta_oris, beta_devs


def calc_beta_oris_from_boundary_misori(grain, neighbour_network, quat_array,
                                        burg_tol=5.):
    """Calculate the possible beta orientations for pairs of alpha and
    neighbour orientations using the misorientaion relation to neighbour
    orientations.

    Parameters
    ----------
    grain : list of defdap.Quat.quat

    neighbour_network :

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

    beta_oris = []
    beta_devs = []
    alpha_oris = []

    neighbour_grains = list(neighbour_network.neighbors(grain))
    for neighbour_grain in neighbour_grains:

        bseg = neighbour_network[grain][neighbour_grain]['boundary']
        # check sense of bseg
        if grain is bseg.grain1:
            ipoint = 0
        else:
            ipoint = 1

        for boundary_point_pair in bseg.boundaryPointPairsX:
            point = boundary_point_pair[ipoint]
            alpha_ori = quat_array[point[1], point[0]]

            point = boundary_point_pair[ipoint - 1]
            neighbour_ori = quat_array[point[1], point[0]]

            min_misoris, min_cub_sym_idxs = calc_misori_of_variants(
                alpha_ori.conjugate, neighbour_ori, unq_cub_sym_comps
            )

            # find the hex symmetries (i, j) from give the minimum
            # deviation from the burgers relation for the minimum store:
            # the deviation, the hex symmetries (i, j) and the cubic
            # symmetry if the deviation is over a threshold then set
            # cubic symmetry to -1
            min_misori_idx = np.unravel_index(np.argmin(min_misoris),
                                              min_misoris.shape)
            burg_dev = min_misoris[min_misori_idx]

            if burg_dev < burg_tol / 180 * np.pi:
                beta_oris.append(beta_oris_from_cub_sym(
                    alpha_ori, min_cub_sym_idxs[min_misori_idx], min_misori_idx[0]
                ))
                beta_devs.append(burg_dev)
                alpha_oris.append(alpha_ori)

    return beta_oris, beta_devs, alpha_oris


def count_beta_variants(beta_oris, possible_beta_oris, grain_id, ori_tol,
                        variant_count=None):
    """

    Parameters
    ----------
    beta_oris
    possible_beta_oris
    grain_id
    ori_tol

    Returns
    -------

    """
    # do all the accounting stuff
    # divide 2 because of 2* in misorientation definition
    ori_tol = np.cos(ori_tol / 2 * np.pi / 180.)
    # flatten list of lists
    possible_beta_oris = [item for sublist in possible_beta_oris
                          for item in sublist]
    unique_beta_oris = []
    count_beta_oris = []
    variant_idxs = []
    for ori in possible_beta_oris:
        found = False
        for i, uniqueOri in enumerate(unique_beta_oris):
            mis_ori = ori.misOri(uniqueOri, "cubic")
            if mis_ori > ori_tol:
                found = True
                count_beta_oris[i] += 1
                break

        if not found:
            unique_beta_oris.append(ori)
            count_beta_oris.append(1)

            for i, betaVariant in enumerate(beta_oris):
                mis_ori = ori.misOri(betaVariant, "cubic")
                if mis_ori > ori_tol:
                    variant_idxs.append(i)
                    break
            else:
                variant_idxs.append(-1)
                warnings.warn("Couldn't find beta variant. "
                              "Grain {:}".format(grain_id))

    if variant_count is None:
        variant_count = [0, 0, 0, 0, 0, 0]
    for i in range(len(variant_idxs)):
        if variant_idxs[i] > -1:
            variant_count[variant_idxs[i]] += count_beta_oris[i]

    return variant_count


def do_reconstruction(ebsd_map, mode=1, burg_tol=5., ori_tol=3.):
    """Apply beta reconstruction to a ebsd map object. Nothing is
    returned and output is stored directly in the ebsd map (this should
    probably change)

    Parameters
    ----------
    ebsd_map: dedap.ebsd.Map
        EBSD map to apply reconstruction to
    mode: int
        How to perform reconstruction
            'average': grain average orientations
            'boundary': grain boundary orientations
    burg_tol: float
        Maximum deviation from the Burgers relation to allow (degrees)
    ori_tol: float
        Maximum deviation from a beta orientaion (degrees)
    """
    # this is the only function that interacts with the ebsd map/grain objects
    num_grains = len(ebsd_map)
    for grain_id, grain in enumerate(ebsd_map):
        report_progress(grain_id, num_grains)

        beta_oris = calc_beta_oris(grain.refOri)

        if mode == 'boundary':
            possible_beta_oris, beta_deviations, alpha_oris = \
                calc_beta_oris_from_boundary_misori(
                    grain, ebsd_map.neighbourNetwork, ebsd_map.quatArray,
                    burg_tol=burg_tol
                )

            variant_count = None
            for possible_beta_ori, beta_deviation, alpha_ori in zip(
                    possible_beta_oris, beta_deviations, alpha_oris):

                beta_oris_l = calc_beta_oris(alpha_ori)

                variant_count = count_beta_variants(
                    beta_oris_l, [possible_beta_ori], grain_id, ori_tol,
                    variant_count=variant_count
                )

        else:
            neighbour_grains = list(ebsd_map.neighbourNetwork.neighbors(grain))
            neighbour_oris = [grain.refOri for grain in neighbour_grains]

            # determine the possible beta orientations based on misorientation
            # between neighbouring alpha grains
            possible_beta_oris, beta_deviations = calc_beta_oris_from_misori(
                grain.refOri, neighbour_oris, burg_tol=burg_tol
            )

            variant_count = count_beta_variants(
                beta_oris, possible_beta_oris, grain_id, ori_tol
            )

        # save results in the grain objects
        grain.betaOris = beta_oris
        grain.possibleBetaOris = possible_beta_oris
        grain.betaDeviations = beta_deviations
        grain.variantCount = variant_count

import numpy as np
import warnings
from typing import List, Tuple
import pathlib

import networkx as nx
from tqdm.auto import tqdm
from defdap import ebsd
from defdap.quat import Quat

from beta_reconstruction.crystal_relations import (
    unq_hex_syms, hex_syms, unq_cub_syms, burg_trans
)


def calc_beta_oris(alpha_ori: Quat) -> List[Quat]:
    """Calculate the possible beta orientations for a given alpha orientation.

    Uses the Burgers relation and crystal symmetries to calculate beta orientations.

    Parameters
    ----------
    alpha_ori
        Orientation of an alpha grain

    Returns
    -------
    list of Quat
        List of possible beta orientations
    """
    beta_oris = []

    for sym in unq_hex_syms:
        beta_oris.append(burg_trans * sym.conjugate * alpha_ori)

    return beta_oris


def construct_quat_comps(oris: List[Quat]) -> np.ndarray:
    """Return a NumPy array of the provided quaternion components

    Input quaternions may be given as a list of Quat objects or any iterable
    whose items have 4 components which map to the quaternion.

    Parameters
    ----------
    oris
        A list of Quat objects to return the components of

    Returns
    -------
    np.ndarray
        Array of quaternion components, shape (4, n)

    """
    quat_comps = np.empty((4, len(oris)))
    for i, ori in enumerate(oris):
        quat_comps[:, i] = ori.quatCoef

    return quat_comps


def beta_oris_from_cub_sym(alpha_ori: Quat, unq_cub_sym_idx: int, hex_sym_idx: int) -> List[Quat]:
    """

    Parameters
    ----------
    alpha_ori
        The orientation of the grain in the alpha phase.
    unq_cub_sym_idx

    hex_sym_idx


    Returns
    -------
    list of Quat
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


def calc_misori_of_variants(alpha_ori_inv: Quat, neighbour_ori: Quat,
                            unq_cub_sym_comps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate possible symmetry variants between two orientations.

    Calculate all possible sym variants for misorientation between two
    orientations undergoing a Burgers type transformation. Then calculate
    the misorientation to the nearest cubic symmetry, this is the deviation
    to a perfect Burgers transformation.

    Parameters
    ----------
    alpha_ori_inv
        Inverse of first orientation
    neighbour_ori
        Second orientation
    unq_cub_sym_comps
        Components of the unique cubic symmetries

    Returns
    -------
    min_misoris : np.ndarray 
       The minimum misorientation for each of the possible beta variants - shape (12, 12)

    min_cub_sym_idx : np.ndarray
       The minimum cubic symmetry index for each of the possible variants - shape (12, 12)

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


def calc_beta_oris_from_misori(alpha_ori: Quat, neighbour_oris: List[Quat],
                               burg_tol: float = 5) -> Tuple[List[List[Quat]], List[float]]:
    """Calculate the possible beta orientations for a given alpha
    orientation using the misorientation relation to neighbour orientations.

    Parameters
    ----------
    alpha_ori
        A quaternion representing the alpha orientation

    neighbour_oris
        Quaternions representing neighbour grain orientations

    burg_tol
        The threshold misorientation angle to determine neighbour relations

    Returns
    -------
    list of lists of defdap.Quat.quat
        Possible beta orientations, grouped by each neighbour. Any
        neighbour with deviation greater than the tolerance is excluded.
    list of float
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
                alpha_ori, min_cub_sym_idxs[min_misori_idx], int(min_misori_idx[0])
            ))
            beta_devs.append(burg_dev)

    return beta_oris, beta_devs


def calc_beta_oris_from_boundary_misori(grain: ebsd.Grain, neighbour_network: nx.Graph,
                                        quat_array: np.ndarray, burg_tol: float = 5) -> Tuple[
      List[List[Quat]], List[float], List[Quat]]:
    """Calculate the possible beta orientations for pairs of alpha and
    neighbour orientations using the misorientation relation to neighbour
    orientations.

    Parameters
    ----------
    grain
        The grain currently being reconstructed

    neighbour_network
        A neighbour network mapping grain boundary connectivity

    quat_array
        Array of quaternions, representing the orientations of the pixels of the EBSD map

    burg_tol :
        The threshold misorientation angle to determine neighbour relations

    Returns
    -------
    list of lists of defdap.Quat.quat
        Possible beta orientations, grouped by each neighbour. Any
        neighbour with deviation greater than the tolerance is excluded.
    list of float
        Deviations from perfect Burgers transformation
    list of Quat
        Alpha orientations
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
                    alpha_ori, min_cub_sym_idxs[min_misori_idx], int(min_misori_idx[0])
                ))
                beta_devs.append(burg_dev)
                alpha_oris.append(alpha_ori)

    return beta_oris, beta_devs, alpha_oris


def count_beta_variants(beta_oris: List[Quat], possible_beta_oris: list, grain_id: int,
                        ori_tol: float) -> np.ndarray:
    """

    Parameters
    ----------
    beta_oris
        Possible beta orientations from burgers relation - 6 for each orientation
    possible_beta_oris
        Possible beta orientations from misorientations
    grain_id
        Used for debugging
    ori_tol
        Tolerance for binning of the orientations into the possible 6
    Returns
    -------
    list of int:
        The newly updated variant count


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

    variant_count = np.zeros(6, dtype=int)
    for i in range(len(variant_idxs)):
        if variant_idxs[i] > -1:
            variant_count[variant_idxs[i]] += count_beta_oris[i]

    return variant_count


def load_map(ebsd_path: str, min_grain_size: int = 3, boundary_tolerance: int = 3,
             use_kuwahara: bool = False, kuwahara_tolerance: int = 5) -> ebsd.Map:
    """Load in EBSD data and do the required prerequisite computations."""

    ebsd_path = pathlib.Path(ebsd_path)
    if ebsd_path.suffix == ".ctf":
        map_type = "OxfordText"
    elif ebsd_path.suffix == ".crc":
        map_type = "OxfordBinary"
    else:
        raise TypeError("Unknown EBSD map type. Can only read .ctf and .crc files.")

    ebsd_map = ebsd.Map(ebsd_path.with_suffix(""), "hexagonal", dataType=map_type)
    ebsd_map.buildQuatArray()

    if use_kuwahara:
        ebsd_map.filterData(misOriTol=kuwahara_tolerance)

    ebsd_map.findBoundaries(boundDef=boundary_tolerance)
    ebsd_map.findGrains(minGrainSize=min_grain_size)

    ebsd_map.calcGrainAvOris()

    ebsd_map.buildNeighbourNetwork()

    return ebsd_map


def assign_modal_variant(ebsd_map: ebsd.Map) -> ebsd.Map:
    """Given a map of grains with variant counts, assign the prior beta orientation of the
    grains to the variant with the highest count."""

    for grain in ebsd_map:
        variantCount = np.array(grain.variantCount)
        modeVariant = np.where(variantCount == np.max(variantCount))[0]
        if len(modeVariant) == 1:
            modeVariant = modeVariant[0]
            parentBetaOri = grain.betaOris[modeVariant]
        else:
            #  multiple variants with same max
            modeVariant = -1
            parentBetaOri = Quat(1., 0., 0., 0.)

        grain.modeVariant = modeVariant
        grain.parentBetaOri = parentBetaOri

    return ebsd_map


def assign_beta_variants(ebsd_map: ebsd.Map,  mode: str = "modal"):
    """Given a map of grains with variant counts, determine the prior beta orientation of the
    grains.

    Parameters
    ----------
    ebsd_map:
        EBSD map to assign the beta variants for.
    mode
        How to perform beta orientation assignment
            'modal': The beta orientation is assigned to the variant with the highest count.

    Returns
    --------
    ebsd.Map
        An ebsd map with the beta orientations assigned to grains.
    """

    if mode == "modal":
        ebsd_map = assign_modal_variant(ebsd_map)
    else:
        raise NotImplementedError(f"Mode '{mode}' is not a recognised way to assign variants.")
    print("Assignment of beta variants complete.")
    return ebsd_map


def do_reconstruction(ebsd_map: ebsd.Map, mode: int = 1, burg_tol: float = 5, ori_tol: float = 3):
    """Apply beta reconstruction to a ebsd map object.

    The reconstructed beta map is stored directly in the ebsd map (this should
    probably change)

    Parameters
    ----------
    ebsd_map:
        EBSD map to apply reconstruction to
    mode
        How to perform reconstruction
            'average': grain average orientations
            'boundary': grain boundary orientations
    burg_tol
        Maximum deviation from the Burgers relation to allow (degrees)
    ori_tol: float
        Maximum deviation from a beta orientation (degrees)
    """
    # this is the only function that interacts with the ebsd map/grain objects
    for grain_id, grain in enumerate(tqdm(ebsd_map)):

        beta_oris = calc_beta_oris(grain.refOri)
        variant_count = np.zeros(6, dtype=int)

        if mode == 'boundary':
            possible_beta_oris, beta_deviations, alpha_oris = \
                calc_beta_oris_from_boundary_misori(
                    grain, ebsd_map.neighbourNetwork, ebsd_map.quatArray,
                    burg_tol=burg_tol
                )

            for possible_beta_ori, beta_deviation, alpha_ori in zip(
                    possible_beta_oris, beta_deviations, alpha_oris):

                beta_oris_l = calc_beta_oris(alpha_ori)

                variant_count += count_beta_variants(beta_oris_l,
                                                     [possible_beta_ori],
                                                     grain_id, ori_tol)

        else:
            neighbour_grains = list(ebsd_map.neighbourNetwork.neighbors(grain))
            neighbour_oris = [grain.refOri for grain in neighbour_grains]

            # determine the possible beta orientations based on misorientation
            # between neighbouring alpha grains
            possible_beta_oris, beta_deviations = calc_beta_oris_from_misori(
                grain.refOri, neighbour_oris, burg_tol=burg_tol
            )

            variant_count += count_beta_variants(
                beta_oris, possible_beta_oris, grain_id, ori_tol
            )

        # save results in the grain objects
        grain.betaOris = beta_oris
        grain.possibleBetaOris = possible_beta_oris
        grain.betaDeviations = beta_deviations
        grain.variantCount = variant_count

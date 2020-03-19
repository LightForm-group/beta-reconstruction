import numpy as np

from defdap.quat import Quat

from beta_reconstruction.crystal_relations import (
    unq_hex_syms, unq_cub_syms, burg_trans
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
        print("\r Done {:} %".format(curr / total * 100))
        # print("\r Grain number {} of {}".format(grain_id + 1, num_grains),
        #       end="")


def calc_beta_oris_from_misori(alpha_ori, neighbour_oris):
    """Calculate the possible beta orientations for a given alpha
    orientation using the misorientaion relation to neighbour orientations.

    Parameters
    ----------
    alpha_ori : defdap.Quat.quat
    neighbour_oris : list of defdap.Quat.quat

    Returns
    -------
    beta_oris : list of defdap.Quat.quat
        List of possible beta orientations
    """
    # This needed to move further up calculation process
    reduced_cub_sym_comps = construct_quat_comps(unq_cub_syms)

    alpha_ori_inv = alpha_ori.conjugate

    # loop over neighbour_oris

    for neighbour_ori in neighbour_oris:

        a2a1inv = neighbour_ori * alpha_ori_inv

        mis144 = np.zeros((12, 12))
        RCS144 = np.zeros((12, 12), dtype=int)

        # calculate all possible S^B_m (eqn 11. from [1]) from the
        # measured misorientation from 2 neighbour alpha grains
        # for each S^B_m calculate the 'closest' cubic symmetry
        # (from reduced subset) and the deviation from this symmetry
        for i, hex_sym in enumerate(unq_hex_syms):
            dummy = a2a1inv * hex_sym

            for j, hex_sym_2 in enumerate(unq_hex_syms):
                Bvariant = burg_trans * ((hex_sym_2.conjugate * dummy) * burg_trans.conjugate)

                misOris = np.einsum("ij,i->j", reduced_cub_sym_comps, Bvariant.quatCoef)

                #                 misOris = []
                #                 for cubicSymm in cubicFundSymms:
                #                     misOri = Bvariant.dot(cubicSymm)
                #                     misOris.append(misOri)
                #                 misOris = np.abs(np.array(misOris))

                misOris = np.abs(misOris)
                misOris[misOris > 1] = 1.
                misOris = 2 * np.arccos(misOris) * 180 / np.pi

                minMisOriIdx = np.argmin(misOris)
                mis144[i, j] = misOris[minMisOriIdx]
                RCS144[i, j] = minMisOriIdx

        # find the hex symmetries (i, j) from give the minimum
        # deviation from the burgers relation for the minimum store:
        # the deviation, the hex symmetries (i, j) and the cubic
        # symmetry if the deviation is over a threshold then set
        # cubic symmetry to -1
        minMisOriIdx = np.unravel_index(np.argmin(mis144), mis144.shape)
        devFromBurgers = mis144[minMisOriIdx]
        cubicSymmIndx = RCS144[minMisOriIdx] if devFromBurgers < maxDevFromBurgers else -1
        a1Symm = minMisOriIdx[0]
        a2Symm = minMisOriIdx[1]
        #         print(devFromBurgers, cubicSymmIndx)

        #         with np.printoptions(precision=2, suppress=True, linewidth=100):
        #             print(minMisOriIdx)
        #             print(mis144)
        #             print(devFromBurgers)
        #             print(RCS144)
        #             print(cubicSymmIndx)
        #             print(" ")

        possibleBetaOris = []
        if cubicSymmIndx > -1 and cubicSymmIndx < 9:
            # one possible beta orientation
            # A:
            possibleBetaOris.append(
                burg_trans * unq_hex_syms[a1Symm].conjugate * grain.refOri
            )

        elif cubicSymmIndx == 9:
            # three possible beta orientation
            # A:
            possibleBetaOris.append(
                burg_trans * unq_hex_syms[a1Symm].conjugate * grain.refOri
            )
            # B:
            # hexFundSymms[1] is C^+_3z
            possibleBetaOris.append(
                burg_trans * unq_hex_syms[1].conjugate * unq_hex_syms[a1Symm].conjugate * grain.refOri
            )
            # C:
            # hexFundSymms[2] is C^+_6z
            possibleBetaOris.append(
                burg_trans * unq_hex_syms[2].conjugate * unq_hex_syms[a1Symm].conjugate * grain.refOri
            )

        elif cubicSymmIndx > 9:
            # two possible beta orientation
            # A:
            possibleBetaOris.append(
                burg_trans * unq_hex_syms[a1Symm].conjugate * grain.refOri
            )
            # D:
            # hexFundSymms[4] is C'_22
            possibleBetaOris.append(
                burg_trans * hexFundSymms[4].conjugate * unq_hex_syms[a1Symm].conjugate * grain.refOri
            )

        grain.possibleBetaOris.append(possibleBetaOris)
        grain.betaDeviations.append(devFromBurgers)


def do_reconstruction(ebsd_map, maxDevFromBurgers=5, oriTo=3):

    num_grains = len(ebsd_map)
    for grain_id, grain in enumerate(ebsd_map):
        report_progress(grain_id, num_grains)

        grain.possibleBetaOris = []
        grain.betaDeviations = []

        neighbour_ids = list(ebsd_map.neighbourNetwork.neighbors(grain_id))
        neighbour_oris = [ebsd_map[i].refOri for i in neighbour_ids]

        calc_beta_oris_from_misori(grain.refOri, neighbour_oris)


        #         print(grain_id, neighbourID)
        #         print(possibleBetaOris)
        #         print(devFromBurgers)
        #         break

        # do all the accounting stuff
        oriTol = 3.  # in degrees
        oriTol = np.cos(
            oriTol / 2 * np.pi / 180.)  # divide 2 because of 2* in misorientation

        allPossibleBetaOris = [item for sublist in grain.possibleBetaOris for
                               item in sublist]
        uniqueBetaOris = []
        countBetaOris = []
        variantIndexes = []

        for ori in allPossibleBetaOris:
            found = False
            for i, uniqueOri in enumerate(uniqueBetaOris):
                misOri = ori.misOri(uniqueOri, "cubic")
                if misOri > oriTol:
                    found = True
                    countBetaOris[i] += 1

            if not found:
                uniqueBetaOris.append(ori)
                countBetaOris.append(1)

                for i, betaVariant in enumerate(grain.betaOris):
                    misOri = ori.misOri(betaVariant, "cubic")
                    if misOri > oriTol:
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



import numpy as np

from defdap.quat import Quat

from beta_reconstruction.crystal_relations import reduced_hex_symms, reduced_cubic_symms, burgers_trans


def calculate_beta_oris(alpha_ori):
    beta_oris = []

    for symm in reduced_hex_symms:
        beta_oris.append(burgers_trans * symm.conjugate * alpha_ori)

    return beta_oris




def do_reconstruction(maxDevFromBurgers=5, oriTo=3):

    cubicSymComps = np.empty((4, len(cubicFundSymms)))
    for i, cubicSymm in enumerate(cubicFundSymms):
        cubicSymComps[:, i] = cubicSymm.quatCoef

    numGrains = len(EbsdMap)
    for grainID, grain in enumerate(EbsdMap):
        # if True:
        if grainID % 10 == 0:
            print("\r Grain number {} of {}".format(grainID + 1, numGrains),
                  end="")

        grain.possibleBetaOris = []
        grain.betaDeviations = []

        neighbourIDs = list(EbsdMap.neighbourNetwork.neighbors(grainID))
        neighbourGrains = [EbsdMap[i] for i in neighbourIDs]

        grainInvOri = grain.refOri.conjugate

        for neighbourID, neighbourGrain in zip(neighbourIDs, neighbourGrains):
            #         if neighbourID != 479:
            #             continue

            neighbourGrainOri = neighbourGrain.refOri

            a2a1inv = neighbourGrainOri * grainInvOri

            mis144 = np.zeros((12, 12))
            RCS144 = np.zeros((12, 12), dtype=int)

            # calculate all posible S^B_m (eqn 11. from [1]) from the measured misorientation from 2 neighbour alpha grains
            # for each S^B_m calculate the 'closest' cubic symmetry (from subset) and the deviation from this symmetry
            for i, hexSymm in enumerate(hexSymms):
                dummy = a2a1inv * hexSymm

                for j, hexSymm2 in enumerate(hexSymms):
                    Bvariant = burgersQuat * ((
                                                          hexSymm2.conjugate * dummy) * burgersQuat.conjugate)

                    misOris = np.einsum("ij,i->j", cubicSymComps,
                                        Bvariant.quatCoef)

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

            # find the hex symmetries (i, j) from give the minimum deviation from the burgers relation
            # for the minimum store: the deviation, the hex symmetries (i, j) and the cubic symmetry
            # if the deviation is over a threshold then set cubic symmetry to -1
            minMisOriIdx = np.unravel_index(np.argmin(mis144), mis144.shape)
            devFromBurgers = mis144[minMisOriIdx]
            cubicSymmIndx = RCS144[
                minMisOriIdx] if devFromBurgers < maxDevFromBurgers else -1
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
                    burgersQuat * hexSymms[a1Symm].conjugate * grain.refOri
                )

            elif cubicSymmIndx == 9:
                # three possible beta orientation
                # A:
                possibleBetaOris.append(
                    burgersQuat * hexSymms[a1Symm].conjugate * grain.refOri
                )
                # B:
                # hexFundSymms[1] is C^+_3z
                possibleBetaOris.append(
                    burgersQuat * hexFundSymms[1].conjugate * hexSymms[
                        a1Symm].conjugate * grain.refOri
                )
                # C:
                # hexFundSymms[2] is C^+_6z
                possibleBetaOris.append(
                    burgersQuat * hexFundSymms[2].conjugate * hexSymms[
                        a1Symm].conjugate * grain.refOri
                )

            elif cubicSymmIndx > 9:
                # two possible beta orientation
                # A:
                possibleBetaOris.append(
                    burgersQuat * hexSymms[a1Symm].conjugate * grain.refOri
                )
                # D:
                # hexFundSymms[4] is C'_22
                possibleBetaOris.append(
                    burgersQuat * hexFundSymms[4].conjugate * hexSymms[
                        a1Symm].conjugate * grain.refOri
                )

            grain.possibleBetaOris.append(possibleBetaOris)
            grain.betaDeviations.append(devFromBurgers)

        #         print(grainID, neighbourID)
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
                        grainID))

        variantCount = [0, 0, 0, 0, 0, 0]
        for i in range(len(variantIndexes)):
            if i > -1:
                variantCount[variantIndexes[i]] = countBetaOris[i]

        grain.variantCount = variantCount



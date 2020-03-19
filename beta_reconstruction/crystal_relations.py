import numpy as np
from defdap.quat import Quat

hex_symms = Quat.symEqv("hexagonal")
# subset of hexagonal symmetries that give a unique orientaions when the
# Burgers transofmration is applied
unq_hex_syms = [
    hex_symms[0],
    hex_symms[5],
    hex_symms[4],
    hex_symms[2],
    hex_symms[10],
    hex_symms[11]
]

cubic_symms = Quat.symEqv("cubic")
# subset of cubic symmetries that give a unique orientaions when the
# Burgers transofmration is applied
unq_cub_syms = [
    cubic_symms[0],
    cubic_symms[7],
    cubic_symms[9],
    cubic_symms[1],
    cubic_symms[22],
    cubic_symms[16],
    cubic_symms[12],
    cubic_symms[15],
    cubic_symms[4],
    cubic_symms[8],
    cubic_symms[21],
    cubic_symms[20]
]

# HCP -> BCC
burg_trans = Quat(135*np.pi/180, 90*np.pi/180, 354.74*np.pi/180).conjugate

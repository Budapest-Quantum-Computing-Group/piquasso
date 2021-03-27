#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import rpy2.robjects.numpy2ri

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr


rpy2.robjects.numpy2ri.activate()


def to_r_matrix(A):
    nr, nc = A.shape
    rA = ro.r.matrix(A, nrow=nr, ncol=nc)
    ro.r.assign("x", rA)
    return rA


def cramer_multidim_test(x, y):
    # NOTE: Uncomment below to install 'cramer'.
    # utils = importr('utils')
    # utils.install_packages('cramer')
    importr('cramer')

    xR = to_r_matrix(x)
    yR = to_r_matrix(y)

    cramer_test = ro.r['cramer.test']
    return cramer_test(xR, yR)

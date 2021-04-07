#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

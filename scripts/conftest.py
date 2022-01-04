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

import pytest

import rpy2.robjects.numpy2ri

from rpy2 import robjects
from rpy2.robjects.packages import importr, PackageNotInstalledError


@pytest.fixture
def cramer_hypothesis_test():
    """
    NOTE: It prints the p-value, the critical value, and the value of the statistic if
    the flag `-s` is used with `pytest`.
    """

    rpy2.robjects.numpy2ri.activate()

    try:
        importr("cramer")

    except PackageNotInstalledError:
        utils = importr("utils")
        utils.install_packages("cramer")

        importr("cramer")

    def _to_r_matrix(A):
        height, width = A.shape
        rA = robjects.r.matrix(A, nrow=height, ncol=width)
        robjects.r.assign("x", rA)

        return rA

    def func(x, y):
        xR = _to_r_matrix(x)
        yR = _to_r_matrix(y)

        cramer_test = robjects.r["cramer.test"]

        rvector = cramer_test(xR, yR)

        return not bool(rvector.rx2("result")[0])

    yield func

    rpy2.robjects.numpy2ri.deactivate()

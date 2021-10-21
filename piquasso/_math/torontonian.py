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

from typing import TypeVar

import numpy as np

from .combinatorics import powerset

TNum = TypeVar("TNum", np.float64, np.complex128)


def torontonian(A: np.ndarray) -> complex:
    d = A.shape[0] // 2

    if d == 0:
        return 1.0 + 0j

    ret = 0.0 + 0j

    for subset in powerset(range(0, d)):
        index = np.ix_(subset, subset)

        A_reduced = np.block(
            [
                [A[:d, :d][index], A[:d, d:][index]],
                [A[d:, :d][index], A[d:, d:][index]],
            ]
        )

        factor = 1.0 if ((d - len(subset)) % 2 == 0) else -1.0

        ret += factor / np.sqrt(
            np.linalg.det(np.identity(len(A_reduced)) - A_reduced).real + 0.0j
        )

    return ret

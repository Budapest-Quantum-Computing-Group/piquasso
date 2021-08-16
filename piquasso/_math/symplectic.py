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

import numpy as np

from scipy.linalg import block_diag

from .linalg import is_square


def symplectic_form(d):
    one_mode_symplectic_form = np.array([[0, 1], [-1, 0]])

    symplectic_form = block_diag(*([one_mode_symplectic_form] * d))

    return symplectic_form


def xp_symplectic_form(d):
    return np.block(
        [
            [np.zeros((d, d)), np.identity(d)],
            [-np.identity(d), np.zeros((d, d))],
        ]
    )


def complex_symplectic_form(d):
    return block_diag(np.identity(d), -np.identity(d))


def is_symplectic(matrix, *, form_func):
    if not is_square(matrix):
        return False

    d = len(matrix) // 2

    form = form_func(d)

    return np.allclose(matrix @ form @ matrix.conjugate().transpose(), form)

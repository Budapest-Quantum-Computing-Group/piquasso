#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

from .plain_hafnian import hafnian

from ..linalg import reduce_

from .loop_hafnian import loop_hafnian


def hafnian_with_reduction(matrix, reduce_on):
    reduced_matrix = reduce_(matrix, reduce_on)

    return hafnian(reduced_matrix)


def _reduce_matrix_with_diagonal(matrix, diagonal, reduce_on):
    reduced_diagonal = reduce_(diagonal, reduce_on=reduce_on)
    reduced_matrix = reduce_(matrix, reduce_on=reduce_on)

    np.fill_diagonal(reduced_matrix, reduced_diagonal)

    return reduced_matrix


def loop_hafnian_with_reduction(matrix, diagonal, reduce_on):
    reduced_matrix = _reduce_matrix_with_diagonal(matrix, diagonal, reduce_on)

    return loop_hafnian(reduced_matrix)

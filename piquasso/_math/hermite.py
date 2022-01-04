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


def modified_hermite_multidim(B, n, alpha):
    try:
        index = tuple(n).index(next(filter(lambda x: x != 0, tuple(n))))
    except StopIteration:
        return 1.0

    if sum(n) == 1:
        return alpha[index]

    n_minus_one = np.copy(n)
    n_minus_one[index] -= 1

    partial_sum = np.empty(shape=(len(n),), dtype=complex)

    for idx, value in enumerate(n_minus_one):
        n_minus_two = np.copy(n_minus_one)
        n_minus_two[idx] -= 1

        partial_sum[idx] = (
            value * modified_hermite_multidim(B, n_minus_two, alpha)
            if value != 0
            else 0.0
        )

    return (
        alpha[index] * modified_hermite_multidim(B, n_minus_one, alpha)
        - B[index, :] @ partial_sum
    )

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
import numba as nb


@nb.njit(parallel=True, cache=True)
def calculate_interferometer_on_fock_space(interferometer, helper_indices):
    cutoff = len(helper_indices[0]) + 2
    subspace_representations = []

    subspace_representations.append(np.array([[1.0]], dtype=interferometer.dtype))
    subspace_representations.append(interferometer)

    for n in range(2, cutoff):
        subspace_indices = helper_indices[0][n - 2]
        first_subspace_indices = helper_indices[2][n - 2]

        first_nonzero_indices = helper_indices[1][n - 2]

        sqrt_occupation_numbers = helper_indices[3][n - 2]
        sqrt_first_occupation_numbers = helper_indices[4][n - 2]

        previous_representation = subspace_representations[n - 1]

        result_shape = (
            first_nonzero_indices.shape[0],
            sqrt_occupation_numbers.shape[0],
        )

        representation = np.zeros(result_shape, dtype=interferometer.dtype)

        for k in range(result_shape[0]):
            denominator = sqrt_first_occupation_numbers[k]
            previous_representation_indexed = previous_representation[
                first_subspace_indices[k]
            ]

            for j in range(sqrt_occupation_numbers.shape[1]):
                one_particle_contrib = (
                    interferometer[first_nonzero_indices[k], j] / denominator
                )

                for i in range(result_shape[1]):
                    representation[k, i] += (
                        one_particle_contrib
                        * sqrt_occupation_numbers[i, j]
                        * previous_representation_indexed[subspace_indices[i, j]]
                    )

        subspace_representations.append(representation.astype(interferometer.dtype))

    return subspace_representations

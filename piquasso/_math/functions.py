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


def gaussian_wigner_function(quadrature_matrix, *, d, mean, cov):
    assert len(quadrature_matrix[0]) == len(mean), (
        "'quadrature_matrix' elements should have the same dimension as 'mean': "
        f"dim(quadrature_matrix[0])={len(quadrature_matrix[0])}, dim(mean)={len(mean)}."
    )

    return [
        gaussian_wigner_function_for_scalar(quadrature_array, d=d, mean=mean, cov=cov)
        for quadrature_array in quadrature_matrix
    ]


def gaussian_wigner_function_for_scalar(X, *, d, mean, cov):
    return (
        (1 / (np.pi ** d))
        * np.sqrt((1 / np.linalg.det(cov)))
        * np.exp(
            - (X - mean)
            @ np.linalg.inv(cov)
            @ (X - mean)
        )
    ).real

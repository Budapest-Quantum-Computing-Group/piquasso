#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

import scipy

import numpy as np

from piquasso._math.permanent import np_glynn_gray_permanent
from piquasso._math.hafnian import hafnian_with_reduction, loop_hafnian_with_reduction

from piquasso.api.calculator import BaseCalculator


class NumpyCalculator(BaseCalculator):
    """The calculations for a simulation using NumPy."""

    def __init__(self):
        self.np = np
        self.fallback_np = np
        self.block_diag = scipy.linalg.block_diag
        self.block = np.block
        self.logm = scipy.linalg.logm
        self.expm = scipy.linalg.expm
        self.powm = np.linalg.matrix_power
        self.polar = scipy.linalg.polar
        self.sqrtm = scipy.linalg.sqrtm
        self.svd = np.linalg.svd

        self.permanent = np_glynn_gray_permanent
        self.hafnian = hafnian_with_reduction
        self.loop_hafnian = loop_hafnian_with_reduction

    def maybe_convert_to_numpy(self, value):
        return value

    def assign(self, array, index, value):
        array[index] = value

        return array

    def scatter(self, indices, updates, dim):
        embedded_matrix = np.zeros((dim,) * 2, dtype=complex)
        composite_index = np.array(indices)[:, 0], np.array(indices)[:, 1]

        embedded_matrix[composite_index] = np.array(updates)

        return embedded_matrix

    def embed_in_identity(self, matrix, indices, dim):
        embedded_matrix = np.identity(dim, dtype=complex)

        embedded_matrix[indices] = matrix

        return embedded_matrix

    def custom_gradient(self, func):
        def wrapper(*args, **kwargs):
            result, _ = func(*args, **kwargs)
            return result

        return wrapper

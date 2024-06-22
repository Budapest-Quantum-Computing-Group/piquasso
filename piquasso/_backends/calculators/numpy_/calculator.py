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

import scipy

import numpy as np

from piquasso._math.permanent import permanent
from piquasso._math.hafnian import hafnian_with_reduction, loop_hafnian_with_reduction

from ..calculator import BuiltinCalculator


class NumpyCalculator(BuiltinCalculator):
    """The calculations for a simulation using NumPy (and SciPy).

    This is enabled by default in the built-in simulators.
    """

    def __init__(self):
        self.np = np
        self.fallback_np = np
        self.forward_pass_np = np
        self.block_diag = scipy.linalg.block_diag
        self.block = np.block
        self.logm = scipy.linalg.logm
        self.expm = scipy.linalg.expm
        self.powm = np.linalg.matrix_power
        self.polar = scipy.linalg.polar
        self.sqrtm = scipy.linalg.sqrtm
        self.svd = np.linalg.svd
        self.schur = scipy.linalg.schur

        self.permanent = permanent
        self.hafnian = hafnian_with_reduction
        self.loop_hafnian = loop_hafnian_with_reduction

    def preprocess_input_for_custom_gradient(self, value):
        return value

    def assign(self, array, index, value):
        array[index] = value

        return array

    def scatter(self, indices, updates, shape):
        embedded_matrix = np.zeros(shape, dtype=complex)
        indices_array = np.array(indices)
        composite_index = tuple([indices_array[:, i] for i in range(len(shape))])

        embedded_matrix[composite_index] = np.array(updates)

        return embedded_matrix

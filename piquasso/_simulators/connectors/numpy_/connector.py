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

from piquasso._math.permanent import permanent as permanent_with_reduction
from piquasso._math.hafnian import (
    hafnian_with_reduction,
    loop_hafnian_with_reduction,
    loop_hafnian_with_reduction_batch,
)

from ..connector import BuiltinConnector

from .interferometer import calculate_interferometer_on_fock_space


def instancemethod(func):
    def wrapped(self, *args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


class NumpyConnector(BuiltinConnector):
    """The calculations for a simulation using NumPy (and SciPy).

    This is enabled by default in the built-in simulators.
    """

    np = fallback_np = forward_pass_np = np

    block_diag = instancemethod(scipy.linalg.block_diag)
    block = instancemethod(np.block)
    logm = instancemethod(scipy.linalg.logm)
    expm = instancemethod(scipy.linalg.expm)
    powm = instancemethod(np.linalg.matrix_power)
    polar = instancemethod(scipy.linalg.polar)
    svd = instancemethod(np.linalg.svd)
    schur = instancemethod(scipy.linalg.schur)
    permanent = instancemethod(permanent_with_reduction)
    hafnian = instancemethod(hafnian_with_reduction)
    loop_hafnian = instancemethod(loop_hafnian_with_reduction)
    loop_hafnian_batch = instancemethod(loop_hafnian_with_reduction_batch)
    calculate_interferometer_on_fock_space = instancemethod(
        calculate_interferometer_on_fock_space
    )

    def sqrtm(self, matrix):
        return scipy.linalg.sqrtm(matrix).astype(np.complex128)

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

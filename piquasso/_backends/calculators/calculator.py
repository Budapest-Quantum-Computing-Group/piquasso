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

from piquasso.api.calculator import BaseCalculator


class BuiltinCalculator(BaseCalculator):
    """Base class for built-in calculators."""

    range = range

    def custom_gradient(self, func):
        def noop_grad(*args, **kwargs):
            result, _ = func(*args, **kwargs)
            return result

        return noop_grad

    def accumulator(self, dtype, size, **kwargs):
        return []

    def write_to_accumulator(self, accumulator, index, value):
        accumulator.append(value)

        return accumulator

    def stack_accumulator(self, accumulator):
        return self.forward_pass_np.stack(accumulator)

    def decorator(self, func):
        return func

    def gather_along_axis_1(self, array, indices):
        return array[:, indices]

    def transpose(self, matrix):
        return self.np.transpose(matrix)

    def embed_in_identity(self, matrix, indices, dim):
        embedded_matrix = self.np.identity(dim, dtype=complex)

        embedded_matrix = self.assign(embedded_matrix, indices, matrix)

        return embedded_matrix

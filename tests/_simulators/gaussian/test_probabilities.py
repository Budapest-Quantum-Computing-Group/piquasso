#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

import piquasso as pq

from piquasso._simulators.gaussian.probabilities import DensityMatrixCalculation


class RecurrenceTestDensityMatrixCalculation(DensityMatrixCalculation):
    def __init__(self, A, linear, normalization, use_loop_hafnian=False):
        super().__init__(connector=pq.NumpyConnector())
        self._A = A
        self._linear = linear
        self._normalization = normalization
        self._use_loop_hafnian = use_loop_hafnian
        self._density_matrix_element_cache[(0,) * len(linear)] = normalization

    def calculate_hafnian(self, reduce_on: np.ndarray) -> complex:
        if self._use_loop_hafnian:
            return self.connector.loop_hafnian(self._A, self._linear, reduce_on)

        return self.connector.hafnian(self._A, reduce_on)


def test_density_matrix_recurrence_matches_hafnian_elements():
    A = np.array(
        [
            [0.1, 0.2 + 0.1j, -0.3, 0.05j],
            [0.2 + 0.1j, -0.2, 0.4 - 0.2j, 0.15],
            [-0.3, 0.4 - 0.2j, 0.25, -0.1j],
            [0.05j, 0.15, -0.1j, 0.05],
        ],
        dtype=complex,
    )
    occupation_numbers = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 0],
            [1, 1],
        ]
    )

    calculation = RecurrenceTestDensityMatrixCalculation(
        A=A,
        linear=np.zeros(4, dtype=complex),
        normalization=0.7 + 0.1j,
    )

    density_matrix = calculation.get_density_matrix(occupation_numbers)

    for i, bra in enumerate(occupation_numbers):
        for j in range(i, len(occupation_numbers)):
            ket = occupation_numbers[j]
            expected = calculation.get_density_matrix_element(bra=bra, ket=ket)

            assert np.isclose(density_matrix[i, j], expected)


def test_density_matrix_recurrence_matches_loop_hafnian_elements():
    A = np.array(
        [
            [0.05, 0.1 + 0.1j, 0.15, -0.05j],
            [0.1 + 0.1j, -0.05, 0.2 - 0.1j, 0.05],
            [0.15, 0.2 - 0.1j, 0.1, 0.1j],
            [-0.05j, 0.05, 0.1j, -0.1],
        ],
        dtype=complex,
    )
    linear = np.array([0.1, -0.2j, 0.05 + 0.1j, -0.15], dtype=complex)
    occupation_numbers = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 0],
            [1, 1],
        ]
    )

    calculation = RecurrenceTestDensityMatrixCalculation(
        A=A,
        linear=linear,
        normalization=0.9 - 0.2j,
        use_loop_hafnian=True,
    )

    density_matrix = calculation.get_density_matrix(occupation_numbers)

    for i, bra in enumerate(occupation_numbers):
        for j in range(i, len(occupation_numbers)):
            ket = occupation_numbers[j]
            expected = calculation.get_density_matrix_element(bra=bra, ket=ket)

            assert np.isclose(density_matrix[i, j], expected)


def test_density_matrix_uses_hermitian_symmetry():
    A = np.zeros((2, 2), dtype=complex)
    calculation = RecurrenceTestDensityMatrixCalculation(
        A=A,
        linear=np.zeros(2, dtype=complex),
        normalization=1.0,
    )
    occupation_numbers = np.array([[0], [1], [2]])

    density_matrix = calculation.get_density_matrix(occupation_numbers)

    assert np.allclose(density_matrix, density_matrix.conj().T)

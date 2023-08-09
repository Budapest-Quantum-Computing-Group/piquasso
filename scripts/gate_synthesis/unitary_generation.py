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

import itertools

import numpy as np
import scipy
import tensorflow as tf


class UnitaryGenerator:
    def __init__(self, cutoff=12, gate_cutoff=4, dtype=np.complex128, seed=None):

        self._cutoff = cutoff
        self._gate_cutoff = gate_cutoff
        self._dtype = dtype
        self._seed = seed

        if seed is not None:
            np.random.seed(seed)

    def generate_number_of_random_unitaries(self, degree, amount):
        target_kets_list = []
        coefficients = []

        for _ in range(amount):
            unitary, coefficient = self.generate_random_unitary(degree)
            coefficients.append(coefficient)

            target_unitary = np.identity(self._cutoff, dtype=np.complex128)
            target_unitary[: self._gate_cutoff, : self._gate_cutoff] = unitary

            target_kets = np.array(
                [target_unitary[:, i] for i in np.arange(self._gate_cutoff)]
            )
            target_kets = tf.constant(target_kets, dtype=np.complex128)
            target_kets_list.append(target_kets)

        coefficients = np.asarray(coefficients)
        target_kets_array = np.asarray(target_kets_list)

        return target_kets_array, coefficients

    def generate_random_unitary(self, degree):
        hamiltonian, coeffs = self._calculate_hamiltonian(degree)
        unitary = scipy.linalg.expm(1j * hamiltonian)

        assert np.allclose(np.identity(self._gate_cutoff), unitary @ np.conj(unitary).T)

        return unitary, coeffs

    def _calculate_hamiltonian(self, degree):
        result = np.identity(self._gate_cutoff, dtype=self._dtype)

        coefficients = self._generate_all_random_coefficients(degree)
        terms = self._calculate_all_normal_odering_terms(degree)

        result = result * coefficients[-1]  # Constant term

        for i in range(len(coefficients) - 1):
            result += coefficients[i] * terms[i]

        return result, coefficients

    def _generate_all_random_coefficients(self, degree):
        """
        Change this or just subsequent `_generate_random_coefficients_of_degree`
        for different coefficient generation methods.
        NOTE: Coefficients are ordered in a descending manner regarding the degree of terms.
              For each degree of normally ordered terms the first term is the highest creation operator.
              Example:
              Let 'a' be the annihilation operator, and 'b' be the creation operator, for degree 3.
              Then b^3 + b^2a + ba^2 + a^3 + b^2 + ba + a^2 + b + a + 1 is the correct ordering.
        """
        if degree < 0:
            return []

        coeff = self._generate_random_coefficients_of_degree(degree)

        return coeff + self._generate_all_random_coefficients(degree - 1)

    def _generate_random_coefficients_of_degree(self, degree):
        if degree == 0:
            return [np.random.uniform(0, 1, 1).item()]

        poly = []

        if degree % 2 == 0:

            complex_coeffs = [
                self._generate_uniform_random_complex() for _ in range(int(degree / 2))
            ]
            conj_coeffs = [np.conj(complex_coeffs[i]) for i in range(int(degree / 2))]
            conj_coeffs.reverse()
            real_coeff = [np.random.uniform(0, 1, 1).item()]
            poly += complex_coeffs + real_coeff + conj_coeffs

        else:

            complex_coeffs = [
                self._generate_uniform_random_complex()
                for _ in range(int(degree / 2) + 1)
            ]
            conj_coeffs = [
                np.conj(complex_coeffs[i]) for i in range(int(degree / 2) + 1)
            ]
            conj_coeffs.reverse()
            poly += complex_coeffs + conj_coeffs

        return poly

    def _calculate_all_normal_odering_terms(self, degree):
        terms = []

        if degree == 0:
            return terms

        for i in range(degree):
            current_degree_terms = self._calculate_normal_ordering_term_of_degree(i + 1)
            terms.append(current_degree_terms)

        terms = list(itertools.chain.from_iterable(terms))
        terms.reverse()  # Due to the order of coefficient generation

        return terms

    def _calculate_normal_ordering_term_of_degree(self, degree):
        creation_op = self._get_creation_operator()
        annihilation_op = self._get_annihilation_operator()

        terms = []

        for i in range(degree + 1):
            creation_power = np.linalg.matrix_power(creation_op, degree - i)
            annihilation_power = np.linalg.matrix_power(annihilation_op, i)
            terms.append(np.matmul(creation_power, annihilation_power))

        return terms

    def _get_creation_operator(self):
        return np.diag(np.sqrt(np.arange(1, self._gate_cutoff, dtype=self._dtype)), -1)

    def _get_annihilation_operator(self):
        return np.diag(np.sqrt(np.arange(1, self._gate_cutoff, dtype=self._dtype)), 1)

    def _generate_uniform_random_complex(self):
        """
        In the unit disc centered at the origin.
        """
        return (
            np.sqrt(np.random.uniform(0, 1, 1))
            * np.exp(1.0j * np.random.uniform(0, 2 * np.pi, 1))
        ).item()

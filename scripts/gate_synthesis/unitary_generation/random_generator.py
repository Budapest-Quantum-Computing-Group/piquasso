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
from unitary_generation.general_functions import *

def generate_random_coefficients_of_degree(degree):
    if degree == 0:
        return [np.random.uniform(0, 1, 1).item()]

    poly = []

    if degree % 2 == 0:

        complex_coeffs = [
            generate_uniform_random_complex() for _ in range(int(degree / 2))
        ]
        conj_coeffs = [np.conj(complex_coeffs[i]) for i in range(int(degree / 2))]
        conj_coeffs.reverse()
        real_coeff = [np.random.uniform(0, 1, 1).item()]
        poly += complex_coeffs + real_coeff + conj_coeffs

    else:

        complex_coeffs = [
            generate_uniform_random_complex()
            for _ in range(int(degree / 2) + 1)
        ]
        conj_coeffs = [
            np.conj(complex_coeffs[i]) for i in range(int(degree / 2) + 1)
        ]
        conj_coeffs.reverse()
        poly += complex_coeffs + conj_coeffs

    return poly


def generate_all_random_coefficients(degree):
    """
    Change this or just subsequent `generate_random_coefficients_of_degree`
    for different coefficient generation methods.
    NOTE: Coefficients are ordered in a descending manner regarding the degree of terms.
            For each degree of normally ordered terms the first term is the highest creation operator.
            Example:
            Let 'a' be the annihilation operator, and 'b' be the creation operator, for degree 3.
            Then b^3 + b^2a + ba^2 + a^3 + b^2 + ba + a^2 + b + a + 1 is the correct ordering.
    """
    if degree < 0:
        return []

    coeff = generate_random_coefficients_of_degree(degree)

    return coeff + generate_all_random_coefficients(degree - 1)


def calculate_hamiltonian(degree, coefficients=None):
    result = np.identity(gate_cutoff, dtype=dtype)

    if coefficients is None:
        coefficients = generate_all_random_coefficients(degree)

    terms = calculate_all_normal_odering_terms(degree)

    result = result * coefficients[-1]  # Constant term

    for i in range(len(coefficients) - 1):
        result += coefficients[i] * terms[i]

    return result, coefficients


def generate_random_unitary(degree, coefficients=None):
    hamiltonian, coeffs = calculate_hamiltonian(degree, coefficients)
    unitary = scipy.linalg.expm(1j * hamiltonian)

    assert np.allclose(np.identity(gate_cutoff), unitary @ np.conj(unitary).T)

    target_unitary = np.identity(cutoff, dtype=dtype)
    target_unitary[: gate_cutoff, : gate_cutoff] = unitary

    target_kets = np.array(
        [target_unitary[:, i] for i in np.arange(gate_cutoff)]
    )
    target_kets = tf.constant(target_kets, dtype=dtype)

    return target_kets, coeffs


def generate_number_of_random_unitaries(degree, amount):
    target_kets_list = []
    coefficients = []

    for _ in range(amount):
        target_kets, coefficient = generate_random_unitary(degree)
        coefficients.append(coefficient)
        target_kets_list.append(target_kets)

    coefficients = np.asarray(coefficients)
    target_kets_array = np.asarray(target_kets_list)

    return target_kets_array, coefficients
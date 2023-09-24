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

from param_config import *
lower_bound = lower_coefficient_bound
upper_bound = upper_coefficient_bound
tnp = tf.experimental.numpy
### Generation independent functions
def get_number_of_coefficients_of_degree(degree):
    # NOTE: This coincidentally also matches the amount of float variables in the system
    return int((degree + 2) * (degree + 1) / 2)


def get_creation_operator():
    return np.diag(np.sqrt(np.arange(1, gate_cutoff, dtype=dtype)), -1)


def get_annihilation_operator():
    return np.diag(np.sqrt(np.arange(1, gate_cutoff, dtype=dtype)), 1)


def calculate_unitary_from_hamiltonian(hamiltonian):
    unitary = scipy.linalg.expm(1j * hamiltonian)

    assert np.allclose(np.identity(gate_cutoff), unitary @ np.conj(unitary).T)

    target_unitary = np.identity(cutoff, dtype=dtype)
    target_unitary[: gate_cutoff, : gate_cutoff] = unitary

    target_kets = np.array(
        [target_unitary[:, i] for i in np.arange(gate_cutoff)]
    )
    target_kets = tf.constant(target_kets, dtype=dtype)

    return target_kets


### Regular random generator functions
def generate_uniform_random_complex():
    """
    In the unit disc centered at the origin.
    """
    return (
        np.sqrt(np.random.uniform(0, 1, 1))
        * np.exp(1.0j * np.random.uniform(0, 2 * np.pi, 1))
    ).item()


def generate_complex_linspace():
    x = np.linspace(lower_bound, upper_bound, number_of_inner_points)
    xs, ys = np.meshgrid(x, 1j*x)
    return xs + ys


def calculate_all_normal_odering_terms(degree):
    terms = []

    if degree == 0:
        return terms

    for i in range(degree):
        current_degree_terms = calculate_normal_ordering_term_of_degree(i + 1)
        terms.append(current_degree_terms)

    terms = list(itertools.chain.from_iterable(terms))
    # Due to the order of coefficient generation.
    # This way the first elements are the highest degree, and decreasing
    terms.reverse()

    return terms

def calculate_normal_ordering_term_of_degree(degree):
    creation_op = get_creation_operator()
    annihilation_op = get_annihilation_operator()

    terms = []

    for i in range(degree + 1):
        creation_power = np.linalg.matrix_power(creation_op, degree - i)
        annihilation_power = np.linalg.matrix_power(annihilation_op, i)
        terms.append(np.matmul(creation_power, annihilation_power))

    return terms




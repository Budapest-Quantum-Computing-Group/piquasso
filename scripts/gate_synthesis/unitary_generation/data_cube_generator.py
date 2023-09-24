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

def get_all_indices(degree):
    number_of_coefficients = degree + 1
    indices = range(number_of_inner_points)
    combinations = itertools.product(indices, repeat=number_of_coefficients)
    return list(combinations)


def create_combination_full_matrices(degree,
                                     complex_linspace,
                                     complex_conj_linspace,
                                     real_linspace,
                                     ):
    # Determining how to distribute the coefficients
    # If the degree is odd, every coefficient is complex, otherwise there is one real term.
    midpoint = int(degree / 2)
    coeff_list_for_range = midpoint
    if degree % 2 == 1:
        coeff_list_for_range += 1

    operator_terms = calculate_normal_ordering_term_of_degree(degree)
    operator_iter = iter(operator_terms)

    complex_list = \
        [
            np.full((
                number_of_inner_points,
                number_of_inner_points,
                gate_cutoff,
                gate_cutoff,
                ),
                next(operator_iter)
            ) * complex_linspace \
            for _ in range(coeff_list_for_range)
        ]

    real_term = None
    if degree % 2 == 0:
        real_term = \
        [
            np.full((
                number_of_inner_points,
                gate_cutoff,
                gate_cutoff,
            ), next(operator_iter)) *\
            real_linspace
        ]

    complex_conj_list = \
        [
            np.full((
                number_of_inner_points,
                number_of_inner_points,
                gate_cutoff,
                gate_cutoff,
                ),
                next(operator_iter)
            ) * complex_conj_linspace \
            for _ in range(coeff_list_for_range)
        ]
    # To become the mirror image of the complex_list
    complex_conj_list.reverse()

    return complex_list, complex_conj_list, real_term


def calculate_every_addition_combination_of_degree(degree,
                                                   complex_list,
                                                   complex_linspace,
                                                   complex_conj_list,
                                                   complex_conj_linspace,
                                                   real_terms,
                                                   real_linspace,
                                                   ):
    combined_operator_terms = [] # shape: (possible_combinations, gate_cutoff, gate_cutoff)
    coefficients = [] # shape: (possible_combinations, number_of_coefficients)
    indices = get_all_indices(degree)
    for index_comb in indices:
        matrix_sum = np.zeros((gate_cutoff, gate_cutoff), dtype)
        this_term_coefficients = []
        for term_index in range(len(complex_list)):
            matrix_sum += complex_list[term_index][index_comb[2*term_index:2*term_index+2]]
            matrix_sum += complex_conj_list[term_index][index_comb[2*term_index:2*term_index+2]]
            # [0][0] for better resulting shape
            this_term_coefficients.append(
                complex_linspace[index_comb[2*term_index:2*term_index+2]][0][0]
            )
            this_term_coefficients.append(
                complex_conj_linspace[index_comb[2*term_index:2*term_index+2]][0][0]
            )
            if degree % 2 == 0:
                matrix_sum += real_terms[0][index_comb[-1]]
                this_term_coefficients.append(real_linspace[index_comb[-1]][0][0])

        combined_operator_terms.append(tf.constant(matrix_sum))
        coefficients.append(tf.constant(this_term_coefficients))

    return tf.convert_to_tensor(combined_operator_terms), tf.convert_to_tensor(coefficients)


def get_hamiltonian_terms_of_degree(degree):
    # Initialization
    complex_linspace = generate_complex_linspace().\
            reshape(number_of_inner_points, number_of_inner_points, 1, 1)
    complex_conj_linspace = np.conj(complex_linspace)
    # Still needs to be complex for computational ease later on.
    real_linspace = np.linspace(
            lower_bound,
            upper_bound,
            number_of_inner_points,
            dtype=dtype).\
                reshape(number_of_inner_points, 1, 1)

    # Edge case
    if degree == 0:
        return np.full((
                    number_of_inner_points,
                    gate_cutoff,
                    gate_cutoff,
                ), np.identity(gate_cutoff))\
                * real_linspace, \
                    real_linspace.reshape(number_of_inner_points, 1)

    # Main part
    complex_list, complex_conj_list, real_terms = \
        create_combination_full_matrices(
            degree,
            complex_linspace,
            complex_conj_linspace,
            real_linspace,
        )

    combined_operator_terms, coefficients = \
        calculate_every_addition_combination_of_degree(
            degree,
            complex_list,
            complex_linspace,
            complex_conj_list,
            complex_conj_linspace,
            real_terms,
            real_linspace,
        )

    return combined_operator_terms, coefficients


### Functions utilizing Tensorflow
@tf.function
def calculate_unitary_from_hamiltonian_tf(hamiltonian):
    unitary = tf.linalg.expm(1j * hamiltonian)

    # tf.function problem
    # assert tnp.allclose(tnp.identity(self._gate_cutoff), unitary @ tnp.conj(unitary).T)
    small_eye = tf.eye(gate_cutoff, gate_cutoff)
    big_eye = tf.eye(cutoff, gate_cutoff)
    unitary_to_pad = unitary - small_eye
    pad_diff = cutoff - gate_cutoff
    pad = tf.constant([[0,pad_diff],[0,0]])
    padded_unitary = tf.pad(unitary_to_pad, pad, "constant")
    target_unitary = padded_unitary + big_eye

    return target_unitary


@tf.function
def add_up_all_combinations_tf(combination_list, coeffs):
    if len(combination_list) == 1:
        return combination_list[0]

    tile1 = combination_list[0]
    coeff_tile1 = coeffs[0]
    for i in range(len(combination_list)-1):  # degree amount of iterations.
        tile2 = tnp.full(((len(tile1),) + combination_list[i+1].shape), combination_list[i+1])
        coeff_tile2 = tnp.full(((coeff_tile1.shape[0], len(coeffs[i+1],)) + coeffs[i+1][0].shape), coeffs[i+1])
        tile1 = tnp.full(((len(combination_list[i+1]),) + tile1.shape), tile1)
        coeff_tile1 = tnp.full(((len(coeffs[i+1]),) + coeff_tile1.shape), coeff_tile1)
        tile1 = tile1 + tf.transpose(tile2, perm=(1, 0, 2, 3))
        tile1 = tile1.reshape(tile1.shape[0] * tile1.shape[1], tile1.shape[2], tile1.shape[3])
        coeff_tile1 = tf.concat((coeff_tile1, tf.transpose(coeff_tile2, perm=(1, 0, 2))), axis=2)
        coeff_tile1 = coeff_tile1.reshape(coeff_tile1.shape[0] * coeff_tile1.shape[1], coeff_tile1.shape[2])

    return tile1, coeff_tile1
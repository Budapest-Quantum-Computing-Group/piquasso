import sys

sys.path.append(".")

import numpy as np
import scipy


def uniform_random_complex() -> complex:
    """
    In the unit disc centered at the origin.
    """
    return (
        np.sqrt(np.random.uniform(0, 1, 1))
        * np.exp(1.0j * np.random.uniform(0, 2 * np.pi, 1))
    ).item()


def creation_operator(size):
    return np.diag(np.sqrt(np.arange(1, size, dtype=np.complex128)), -1)


def annihilation_operator(size):
    return np.diag(np.sqrt(np.arange(1, size, dtype=np.complex128)), 1)


def normal_polynomial_text(order: int) -> str:
    if order == 0:
        return ""

    poly = ""

    for i in range(order):
        poly += ("d" * (order - i)) + ("a" * i) + " + "

    poly += ("a" * order) + " + "

    return poly + normal_polynomial_text(order - 1)


def generate_random_normal_polynomial_coeffs(order: int, seed=None) -> list:
    if seed is not None:
        np.random.seed(seed)

    if order == 0:
        return [np.random.uniform(0, 1, 1).item()]

    poly = []

    if order % 2 == 0:

        complex_coeffs = [uniform_random_complex() for _ in range(int(order / 2))]
        conj_coeffs = [np.conj(complex_coeffs[i]) for i in range(int(order / 2))]
        conj_coeffs.reverse()
        real_coeff = [np.random.uniform(0, 1, 1).item()]
        poly += complex_coeffs + real_coeff + conj_coeffs

    else:

        complex_coeffs = [uniform_random_complex() for _ in range(int(order / 2) + 1)]
        conj_coeffs = [np.conj(complex_coeffs[i]) for i in range(int(order / 2) + 1)]
        conj_coeffs.reverse()
        poly += complex_coeffs + conj_coeffs

    return poly


def generate_all_random_normal_polynomial_coeffs(order: int, seed=None) -> list:
    """
    Same as above but calculates every single coefficient recursively, rather than for just one order
    """

    if order < 0:
        return []

    coeff = generate_random_normal_polynomial_coeffs(order, seed)
    return coeff + generate_all_random_normal_polynomial_coeffs(order - 1, seed)


def normal_polynomial(order: int, cutoff: int, seed=None):

    if order == 0:
        coeff = generate_random_normal_polynomial_coeffs(order, seed)
        return coeff * np.identity(cutoff, dtype=np.complex128), coeff

    nth_order_terms = np.zeros((cutoff, cutoff), dtype=np.complex128)

    coeffs = generate_random_normal_polynomial_coeffs(order, seed)

    for i in range(order + 1):

        creation_op = creation_operator(cutoff)
        annihilation_op = annihilation_operator(cutoff)

        creation_power = np.linalg.matrix_power(creation_op, order - i)

        annihilation_power = np.linalg.matrix_power(annihilation_op, i)

        nth_order_terms += coeffs[i] * (creation_power @ annihilation_power)

    lower_matrix, lower_coeffs = normal_polynomial(order - 1, cutoff, seed)
    return nth_order_terms + lower_matrix, coeffs + lower_coeffs


def generate_unitary(order: int, cutoff: int, seed=None):
    polynomial, coeffs = normal_polynomial(order, cutoff, seed)
    return scipy.linalg.expm(1j * polynomial), coeffs


if __name__ == "__main__":
    while True:
        cutoff = 5
        seed = 3
        order = int(input())
        coeffs = generate_all_random_normal_polynomial_coeffs(order, seed)
        unitary, same_coeffs = generate_unitary(order, cutoff, seed)
        assert np.allclose(np.asarray(coeffs) - np.asarray(same_coeffs), 0)
        self_multiplication = unitary @ np.conj(unitary).T
        assert np.allclose(self_multiplication, np.identity(cutoff))
        print(self_multiplication)

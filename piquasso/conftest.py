#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import numpy as np


@pytest.fixture(scope="session")
def tolerance():
    return 1E-9


@pytest.fixture
def generate_symmetric_matrix():
    def func(N):
        A = np.random.rand(N, N)

        return A + A.T

    return func


@pytest.fixture
def generate_complex_symmetric_matrix(generate_symmetric_matrix):
    def func(N):
        real = generate_symmetric_matrix(N)
        imaginary = generate_symmetric_matrix(N)

        return real + 1j * imaginary

    return func


@pytest.fixture
def generate_unitary_matrix():
    from scipy.stats import unitary_group

    def func(N):
        return np.array(unitary_group.rvs(N), dtype=complex)

    return func


@pytest.fixture
def generate_hermitian_matrix(generate_unitary_matrix):
    from scipy.linalg import logm

    def func(N):
        U = generate_unitary_matrix(N)

        return 1j * logm(U)

    return func

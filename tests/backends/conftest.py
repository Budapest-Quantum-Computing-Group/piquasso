#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest


@pytest.fixture
def generate_hermitian_matrix(generate_unitary_matrix):
    from scipy.linalg import logm

    def func(N):
        U = generate_unitary_matrix(N)

        return 1j * logm(U)

    return func

#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso.gaussian.state import GaussianState
from piquasso import constants


class TestGaussianStateRepresentations:

    @pytest.fixture
    def mean(self):
        return np.array([1 - 2j, 3 + 4j], dtype=complex)

    @pytest.fixture
    def C(self):
        return np.array(
            [
                [     1, 1 + 2j],
                [1 - 2j,      6],
            ],
            dtype=complex,
        )

    @pytest.fixture
    def G(self):
        return np.array(
            [
                [3 + 1j, 1 + 1j],
                [1 + 1j, 2 - 3j],
            ],
            dtype=complex,
        )

    @pytest.fixture
    def mu(self):
        return np.array([1, -2, 3, 4]) * np.sqrt(2 * constants.HBAR_DEFAULT)

    @pytest.fixture
    def corr(self):
        return np.array(
            [
                [ 9,  2,  4,  6],
                [ 2,  3, -2,  0],
                [ 4, -2, 17, -6],
                [ 6,  0, -6, -9],
            ]
        ) * constants.HBAR_DEFAULT

    @pytest.fixture
    def cov(self):
        return np.array(
            [
                [  5,  10,  -8, -10],
                [ 10, -13,  22,  32],
                [ -8,  22, -19, -54],
                [-10,  32, -54, -73]
            ]
        ) * constants.HBAR_DEFAULT

    @pytest.fixture
    def xp_mean(self):
        return np.array([1, 3, -2, 4]) * np.sqrt(2 * constants.HBAR_DEFAULT)

    @pytest.fixture
    def xp_corr(self):
        return np.array(
            [
                [ 9,  4,  2,  6],
                [ 4, 17, -2, -6],
                [ 2, -2,  3,  0],
                [ 6, -6,  0, -9]
            ]
        ) * constants.HBAR_DEFAULT

    @pytest.fixture
    def xp_cov(self):
        return np.array(
            [
                [  5,  -8,  10, -10],
                [ -8, -19,  22, -54],
                [ 10,  22, -13,  32],
                [-10, -54,  32, -73]
            ]
        ) * constants.HBAR_DEFAULT

    @pytest.fixture(autouse=True)
    def setup(self, C, G, mean):
        self.state = GaussianState(C, G, mean)

    def test_xp_representation(
        self,
        xp_mean,
        xp_corr,
        xp_cov,
    ):
        assert np.allclose(
            xp_mean,
            self.state.xp_mean,
        )
        assert np.allclose(
            xp_corr,
            self.state.xp_corr,
        )
        assert np.allclose(
            xp_cov,
            self.state.xp_cov,
        )

    def test_quad_representation(
        self,
        mu,
        corr,
        cov,
    ):
        assert np.allclose(
            mu,
            self.state.mu,
        )
        assert np.allclose(
            corr,
            self.state.corr,
        )
        assert np.allclose(
            cov,
            self.state.cov,
        )


class TestGaussianStateOperations:
    @pytest.fixture
    def mean(self):
        return np.array([1 - 2j, 3 + 4j, 2 - 5j], dtype=complex)

    @pytest.fixture
    def C(self):
        return np.array(
            [
                [     1, 1 + 2j, 4 - 6j],
                [1 - 2j,      6, 5 + 9j],
                [4 + 6j, 5 - 9j,    -10],
            ],
            dtype=complex,
        )

    @pytest.fixture
    def G(self):
        return np.array(
            [
                [3 + 1j, 1 + 1j, 2 + 2j],
                [1 + 1j, 2 - 3j, 5 - 6j],
                [2 + 2j, 5 - 6j,      7]
            ],
            dtype=complex,
        )

    @pytest.fixture(autouse=True)
    def setup(self, C, G, mean):
        self.state = GaussianState(C, G, mean)

    def test_rotated(self):
        phi = np.pi / 2
        rotated_state = self.state.rotated(phi)

        rotated_C = np.array(
            [
                [     1, 1 + 2j, 4 - 6j],
                [1 - 2j,      6, 5 + 9j],
                [4 + 6j, 5 - 9j,    -10],
            ]
        )

        rotated_G = np.array(
            [
                [-3 - 1j, -1 - 1j, -2 - 2j],
                [-1 - 1j, -2 + 3j, -5 + 6j],
                [-2 - 2j, -5 + 6j,      -7],
            ]
        )

        rotated_mean = np.array([-2 - 1j,  4 - 3j, -5 - 2j])

        assert np.allclose(
            rotated_C,
            rotated_state.C,
        )
        assert np.allclose(
            rotated_G,
            rotated_state.G,
        )
        assert np.allclose(
            rotated_mean,
            rotated_state.mean,
        )

    def test_apply_to_C_and_G_for_1_modes(self, C, G):
        alpha = np.exp(1j * np.pi/3)

        T = np.array(
            [
                [alpha]
            ],
            dtype=complex,
        )

        expected_C = C.copy()

        expected_C[1, 1] = alpha.conjugate() * C[1, 1] * alpha

        expected_C[1, 0] = alpha.conjugate() * C[1, 0]
        expected_C[1, 2] = alpha.conjugate() * C[1, 2]

        expected_C[0, 1] = C[0, 1] * alpha
        expected_C[2, 1] = C[2, 1] * alpha

        expected_G = G.copy()

        expected_G[1, 1] = alpha * G[1, 1] * alpha

        expected_G[1, 0] = alpha * G[1, 0]
        expected_G[1, 2] = alpha * G[1, 2]

        expected_G[0, 1] = G[0, 1] * alpha
        expected_G[2, 1] = G[2, 1] * alpha

        self.state.apply_to_C_and_G(T, modes=(1,))

        assert np.allclose(self.state.C, expected_C)
        assert np.allclose(self.state.G, expected_G)

    def test_apply_to_C_and_G_for_2_modes(self, C, G):

        T = np.array(
            [
                [     1, 5 + 6j],
                [5 - 6j,      7],
            ],
            dtype=complex,
        )

        expected_C = np.zeros((3, 3), dtype=complex)

        expected_C[:2, :2] = T.conjugate() @ C[:2, :2] @ T.transpose()

        expected_C[(0, 1), 2] = T.conjugate() @ C[(0, 1), 2]
        expected_C[2, (0, 1)] = C[2, (0, 1)] @ T.transpose()

        expected_C[2, 2] = C[2, 2]

        expected_G = np.zeros((3, 3), dtype=complex)

        expected_G[:2, :2] = T @ G[:2, :2] @ T.transpose()

        expected_G[(0, 1), 2] = T @ G[(0, 1), 2]
        expected_G[2, (0, 1)] = G[2, (0, 1)] @ T.transpose()

        expected_G[2, 2] = G[2, 2]

        self.state.apply_to_C_and_G(T, modes=(0, 1))

        assert np.allclose(self.state.C, expected_C)
        assert np.allclose(self.state.G, expected_G)

    def test_apply_to_C_and_G_for_all_modes(self, C, G):
        T = np.array(
            [
                [       1,    3 + 4j,   9 - 10j],
                [  3 - 4j,         7, -11 + 12j],
                [ 9 + 10j, -11 - 12j,        18]
            ],
            dtype=complex,
        )

        expected_C = T.conjugate() @ C @ T.transpose()

        expected_G = T @ G @ T.transpose()

        self.state.apply_to_C_and_G(T, modes=(0, 1, 2))

        assert np.allclose(self.state.C, expected_C)
        assert np.allclose(self.state.G, expected_G)


def test_apply(
    generate_complex_symmetric_matrix,
    generate_hermitian_matrix,
    generate_unitary_matrix
):
    C = generate_hermitian_matrix(5)
    G = generate_complex_symmetric_matrix(5)
    mean = np.random.rand(5) + 1j * np.random.rand(5)

    state = GaussianState(
        C=C,
        G=G,
        mean=mean,
    )

    expected_mean = mean.copy()
    expected_C = C.copy()
    expected_G = G.copy()

    T = generate_unitary_matrix(3)

    expected_mean[(0, 1, 3), ] = T @ expected_mean[(0, 1, 3), ]

    columns = np.array(
        [
            [0, 1, 3],
            [0, 1, 3],
            [0, 1, 3],
        ]
    )

    rows = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [3, 3, 3],
        ]
    )

    index = rows, columns

    expected_C[index] = T.conjugate() @ expected_C[index] @ T.transpose()
    expected_C[(0, 1, 3), 2] = T.conjugate() @ expected_C[(0, 1, 3), 2]
    expected_C[(0, 1, 3), 4] = T.conjugate() @ expected_C[(0, 1, 3), 4]
    expected_C[:, (0, 1, 3)] = np.conj(expected_C[(0, 1, 3), :]).transpose()

    expected_G[index] = T @ expected_G[index] @ T.transpose()
    expected_G[(0, 1, 3), 2] = T @ expected_G[(0, 1, 3), 2]
    expected_G[(0, 1, 3), 4] = T @ expected_G[(0, 1, 3), 4]
    expected_G[:, (0, 1, 3)] = expected_G[(0, 1, 3), :].transpose()

    state.apply(T, modes=(0, 1, 3))

    assert np.allclose(state.mean, expected_mean)
    assert np.allclose(state.C, expected_C)
    assert np.allclose(state.G, expected_G)

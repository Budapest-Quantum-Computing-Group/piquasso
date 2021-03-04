#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso import _math

from piquasso.decompositions.clements import T, Clements


def test_T_beamsplitter_is_unitary():
    theta = np.pi / 3
    phi = np.pi / 4

    beamsplitter = T({"params": [theta, phi], "modes": [0, 1]}, d=2)

    assert _math.is_unitary(beamsplitter)


def test_eliminate_lower_offdiagonal_2_modes(dummy_unitary, tolerance):

    U = dummy_unitary(d=2)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_lower_offdiagonal(1, 0)

    beamsplitter = T(operation, 2)

    rotated_U = beamsplitter @ U

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_lower_offdiagonal_3_modes(dummy_unitary, tolerance):

    U = dummy_unitary(d=3)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_lower_offdiagonal(1, 0)

    beamsplitter = T(operation, 3)

    rotated_U = beamsplitter @ U

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_upper_offdiagonal_2_modes(dummy_unitary, tolerance):

    U = dummy_unitary(d=2)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_upper_offdiagonal(1, 0)

    beamsplitter = T.i(operation, 2)

    rotated_U = U @ beamsplitter

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_upper_offdiagonal_3_modes(dummy_unitary, tolerance):

    U = dummy_unitary(d=3)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_upper_offdiagonal(1, 0)

    beamsplitter = T.i(operation, 3)

    rotated_U = U @ beamsplitter

    assert np.abs(rotated_U[1, 0]) < tolerance


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_clements_decomposition_on_n_modes(
            n, dummy_unitary, tolerance
        ):

    U = dummy_unitary(d=n)

    decomposition = Clements(U)

    diagonalized = decomposition.U

    assert np.abs(diagonalized[0, 1]) < tolerance
    assert np.abs(diagonalized[1, 0]) < tolerance

    assert (
        sum(sum(np.abs(diagonalized)))
        - sum(np.abs(np.diag(diagonalized))) < tolerance
    ), "Absolute sum of matrix elements should be equal to the "
    "diagonal elements, if the matrix is properly diagonalized."


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_clements_decomposition_and_composition_on_n_modes(
            n, dummy_unitary, tolerance
        ):

    U = dummy_unitary(d=n)

    decomposition = Clements(U)

    diagonalized = decomposition.U

    assert (
        sum(sum(np.abs(diagonalized)))
        - sum(np.abs(np.diag(diagonalized))) < tolerance
    ), "Absolute sum of matrix elements should be equal to the "
    "diagonal elements, if the matrix is properly diagonalized."

    original = Clements.from_decomposition(decomposition)

    assert (U - original < tolerance).all()

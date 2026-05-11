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
#
# Portions of this file are based on work by Bence Soóki-Tóth, used with
# permission and originally made available under the MIT License.
#
# Bence Soóki-Tóth. "Efficient calculation of permanent function gradients
# in photonic quantum computing simulations", Eötvös Loránd University, 2025.

import numpy as np
import pytest

pytest.importorskip(
    "piquasso.jax_extensions._perm_boost_core",
    reason="perm_boost C++ extension is not compiled",
)

from piquasso.jax_extensions.permanent import perm  # noqa: E402


def assym_reduce(array, row_reduce_on, col_reduce_on):
    particles = np.sum(row_reduce_on)

    proper_row_index = np.zeros(particles, dtype=int)
    proper_col_index = np.zeros(particles, dtype=int)

    row_stride = 0
    col_stride = 0

    for index in range(len(row_reduce_on)):
        row_multiplier = row_reduce_on[index]
        proper_row_index[row_stride : row_stride + row_multiplier] = index
        row_stride += row_multiplier

    for index in range(len(col_reduce_on)):
        col_multiplier = col_reduce_on[index]
        proper_col_index[col_stride : col_stride + col_multiplier] = index
        col_stride += col_multiplier

    return array[np.ix_(proper_row_index, proper_col_index)]


def test_permanent_empty_matrix():
    matrix = np.empty((0, 0), dtype=np.complex128)
    input = output = np.zeros(0, dtype=np.uint64)
    assert np.isclose(perm(matrix, input, output), 1.0)


def test_permanent_single_zero_entry():
    matrix = np.array([[0]], dtype=np.complex128)
    input = output = np.ones(1, dtype=np.uint64)
    assert np.isclose(perm(matrix, input, output), 0.0)


def test_permanent_row_col_mismatch():
    matrix = np.eye(2, dtype=np.complex128)
    input = np.ones(2, dtype=np.uint64)
    output = np.ones(3, dtype=np.uint64)
    with pytest.raises(Exception):
        perm(matrix, input, output)


def test_permanent_negative_entries():
    matrix = np.array([[1, -2], [-3, 4]], dtype=np.complex128)
    input = output = np.ones(2, dtype=np.uint64)
    # expected: 1 * 4 + (-2) * (-3) = 10
    assert np.isclose(perm(matrix, input, output), 10.0)


def test_permanent_large_values():
    matrix = np.full((3, 3), 1e10, dtype=np.complex128)
    input = output = np.ones(3, dtype=np.uint64)
    result = perm(matrix, input, output)
    assert np.isfinite(result)


def test_permanent_high_repetition():
    matrix = np.array([[2, 3], [4, 5]], dtype=np.complex128)
    input = np.array([3, 0], dtype=np.uint64)
    output = np.array([2, 1], dtype=np.uint64)
    result = perm(matrix, input, output)
    assert np.isfinite(result)


def test_permanent_all_zeros_input_output():
    matrix = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    input = output = np.zeros(4, dtype=np.uint64)
    assert np.isclose(perm(matrix, input, output), 1.0)


def test_permanent_large_matrix_small_repetition():
    matrix = np.eye(8, dtype=np.complex128)
    input = output = np.ones(8, dtype=np.uint64)
    result = perm(matrix, input, output)
    assert np.isclose(result, 1.0)


def test_permanent_real_matrix():
    matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
    input = output = np.ones(2, dtype=np.uint64)
    with pytest.raises(Exception):
        perm(matrix, input, output)


def test_permanent_complex_conjugate_symmetry():
    matrix = np.array([[1 + 1j, 2 - 1j], [3 + 0j, 4 + 2j]], dtype=np.complex128)
    input = output = np.ones(2, dtype=np.uint64)
    result = perm(matrix, input, output)
    result_conj = perm(np.conj(matrix), input, output)
    assert np.isclose(result_conj, np.conj(result))


def test_permanent_trivial_case():
    matrix = np.array([[4.2]], dtype=np.complex128)
    input = output = np.ones(1, dtype=np.uint64)
    assert np.isclose(perm(matrix, input, output), 4.2)


def test_permanent_zero_input():
    interferometer = np.array(
        [
            [
                0.62270314 + 0.55117657j,
                -0.0258677 - 0.07171713j,
                0.09597446 - 0.54168404j,
            ],
            [
                0.34756795 - 0.29444499j,
                -0.43514701 + 0.18975153j,
                0.71929752 + 0.22304973j,
            ],
            [
                -0.24500645 - 0.20227626j,
                -0.45222962 - 0.75121057j,
                0.06995606 - 0.3540245j,
            ],
        ],
        dtype=np.complex128,
    )

    input = output = np.zeros(3, dtype=np.uint64)

    assert np.isclose(perm(interferometer, input, output), 1.0)


def test_permanent_no_repetition():
    interferometer = np.array(
        [
            [
                0.62113733 - 0.01959968j,
                -0.15627468 - 0.0772489j,
                -0.2705819 - 0.18997122j,
                0.26504798 - 0.30838768j,
                0.03372169 - 0.11154586j,
                0.15278476 + 0.52137824j,
            ],
            [
                -0.1776024 - 0.21000195j,
                0.18950753 + 0.20741494j,
                -0.15537846 + 0.19161071j,
                0.07400899 - 0.37578572j,
                -0.44458249 - 0.0047501j,
                -0.62212719 + 0.23055313j,
            ],
            [
                -0.05572001 - 0.20287464j,
                0.22359337 + 0.30693557j,
                -0.13719319 + 0.23245719j,
                0.1102451 + 0.02659467j,
                0.81942653 + 0.04327346j,
                -0.17215559 + 0.15114287j,
            ],
            [
                -0.24319645 - 0.44143551j,
                -0.50022937 - 0.08513718j,
                0.07671116 - 0.05858231j,
                0.0679656 + 0.52109972j,
                -0.0482276 - 0.12736588j,
                -0.11768435 + 0.41307881j,
            ],
            [
                -0.29469977 - 0.20027018j,
                0.22135149 - 0.02983563j,
                -0.18587346 - 0.83950064j,
                -0.21606625 - 0.14975436j,
                0.11702974 - 0.02297493j,
                -0.01552763 + 0.01646485j,
            ],
            [
                -0.29741767 + 0.15644426j,
                -0.61959257 - 0.23497653j,
                0.07397837 + 0.05367843j,
                -0.05838964 - 0.57132173j,
                0.28736069 - 0.00798998j,
                -0.13763068 - 0.09058005j,
            ],
        ],
        dtype=np.complex128,
    )

    input = output = np.ones(6, dtype=np.uint64)

    assert np.isclose(
        perm(interferometer, input, output),
        0.022227325527358795 + 0.01807052573717885j,
    )


def test_permanent_2_by_2_asymmetric():
    input = np.array([2, 0], dtype=np.uint64)
    output = np.array([0, 2], dtype=np.uint64)
    interferometer = np.array([[1, 1j], [1, -1j]], dtype=np.complex128) / np.sqrt(2)
    assert np.isclose(perm(interferometer, input, output), -1)


def test_permanent_4_by_4():
    unitary = np.array(
        [
            [
                0.50142122 - 0.15131566j,
                0.0265964 - 0.67076793j,
                -0.48706586 - 0.15221542j,
                0.1148288 - 0.0381445j,
            ],
            [
                -0.30504953 + 0.02783877j,
                -0.33442264 + 0.27130083j,
                -0.52279136 - 0.6644152j,
                -0.04684954 + 0.06143215j,
            ],
            [
                0.20279518 + 0.57890235j,
                -0.47584393 - 0.16777288j,
                0.08168919 + 0.10673884j,
                -0.34434342 + 0.48221603j,
            ],
            [
                0.31022502 - 0.39919511j,
                0.21927998 + 0.24751057j,
                -0.06266598 - 0.05334444j,
                -0.78447544 + 0.11350836j,
            ],
        ],
        dtype=np.complex128,
    )

    rows = np.array([3, 0, 0, 1], dtype=np.uint64)
    cols = np.array([0, 0, 3, 1], dtype=np.uint64)
    assert np.isclose(
        perm(unitary, rows, cols),
        0.4302957973670928 + 0.3986355418194044j,
    )


def test_permanent_6_by_6():
    interferometer = np.array(
        [
            [
                0.00456448 - 0.37857979j,
                0.0795763 + 0.10465352j,
                0.03365652 + 0.41152925j,
                0.05555206 + 0.18738511j,
                -0.39720048 + 0.1311295j,
                0.40848797 + 0.537455j,
            ],
            [
                0.00901246 + 0.19215191j,
                0.4609619 + 0.13507775j,
                -0.47801241 + 0.19203936j,
                0.23140442 - 0.62025569j,
                0.00333535 + 0.14816728j,
                0.0783928 + 0.02267907j,
            ],
            [
                0.43945864 - 0.04054749j,
                0.00377335 - 0.30622153j,
                0.22692424 - 0.43506116j,
                0.08710784 - 0.32486556j,
                -0.58818325 + 0.08888961j,
                0.04507539 - 0.0406845j,
            ],
            [
                -0.29415733 + 0.17675044j,
                0.33184779 + 0.39750559j,
                0.34950358 - 0.26546825j,
                -0.45096811 - 0.10690984j,
                -0.12192328 + 0.20376462j,
                -0.237482 + 0.30640823j,
            ],
            [
                -0.02410776 + 0.30238262j,
                -0.46322751 + 0.20107186j,
                -0.09817621 - 0.15398772j,
                -0.29490926 - 0.28320581j,
                0.02993443 - 0.3052611j,
                0.5788743 + 0.15254606j,
            ],
            [
                0.53706284 + 0.35589712j,
                -0.26260555 + 0.25708476j,
                0.23762169 + 0.20229218j,
                0.16489218 + 0.03524027j,
                0.24701479 + 0.48757485j,
                -0.06188047 + 0.14647274j,
            ],
        ],
        dtype=np.complex128,
    )

    output = np.array([2, 1, 3, 0, 1, 2], dtype=np.uint64)
    input = np.array([1, 1, 0, 3, 2, 2], dtype=np.uint64)
    assert np.isclose(
        perm(interferometer, input, output),
        0.13160241373727416 + 0.36535625772184577j,
    )


def test_permanent_asymmetric_matrix():
    unitary = np.array(
        [
            [0.909394 + 0.264435j, 0.00450261 + 0.0188079j, 0.316704 + 0.0490014j],
            [-0.109435 - 0.117915j, 0.35198 + 0.812257j, 0.312276 + 0.304881j],
        ],
        dtype=np.complex128,
    )
    rows = np.array([2, 1], dtype=np.uint64)
    cols = np.array([1, 1, 1], dtype=np.uint64)

    ones = np.ones(3, dtype=np.uint64)

    assert np.isclose(
        perm(unitary, rows, cols),
        perm(
            assym_reduce(unitary, rows.astype(dtype=int), cols.astype(dtype=int)),
            ones,
            ones,
        ),
    )

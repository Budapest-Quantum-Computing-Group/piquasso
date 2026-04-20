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
import jax

from piquasso._math.perm_boost.permanent import perm
from piquasso._math.jax.permanent import permanent_with_reduction


def perm_wrapper(permanent_func):
    def wrapper(primal, rows, cols):
        res = permanent_func(primal, rows, cols)
        return res.real, res.imag
    return wrapper


def test_grad_perm_trivial_case():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex128)
    rows = cols = np.ones(3, dtype=np.uint64)
    output = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.array_equal(
        output,
        np.array(
            [
                [93.0 + 0.0j, 78.0 + 0.0j, 67.0 + 0.0j],
                [42.0 + 0.0j, 30.0 + 0.0j, 22.0 + 0.0j],
                [27.0 + 0.0j, 18.0 + 0.0j, 13.0 + 0.0j],
            ]
        ),
    )

    # Test holomorphic=False by comparing to the Jacobian
    # of the real and imaginary parts separately
    jacobian = jax.jacobian(perm_wrapper(perm))(matrix, rows, cols)

    matrix, rows, cols = jax.numpy.array(matrix), jax.numpy.array(rows), jax.numpy.array(cols)
    expected = jax.jacobian(perm_wrapper(permanent_with_reduction))(matrix, rows, cols)
    assert np.allclose(jacobian[0], expected[0])
    assert np.allclose(jacobian[1], expected[1])


def test_grad_perm_identity():
    matrix = np.eye(3, dtype=np.complex128)
    rows = cols = np.ones(3, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, np.eye(3, dtype=np.complex128))

    # Test holomorphic=False by comparing to the Jacobian
    # of the real and imaginary parts separately
    jacobian = jax.jacobian(perm_wrapper(perm))(matrix, rows, cols)

    matrix, rows, cols = jax.numpy.array(matrix), jax.numpy.array(rows), jax.numpy.array(cols)
    expected = jax.jacobian(perm_wrapper(permanent_with_reduction))(matrix, rows, cols)
    assert np.allclose(jacobian[0], expected[0])
    assert np.allclose(jacobian[1], expected[1])


def test_grad_perm_single_entry():
    matrix = np.array([[2.0]], dtype=np.complex128)
    rows = cols = np.ones(1, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, np.array([[1.0 + 0j]]))

    # Test holomorphic=False by comparing to the Jacobian
    # of the real and imaginary parts separately
    jacobian = jax.jacobian(perm_wrapper(perm))(matrix, rows, cols)

    matrix, rows, cols = jax.numpy.array(matrix), jax.numpy.array(rows), jax.numpy.array(cols)
    expected = jax.jacobian(perm_wrapper(permanent_with_reduction))(matrix, rows, cols)
    assert np.allclose(jacobian[0], expected[0])
    assert np.allclose(jacobian[1], expected[1])


def test_grad_perm_zero_matrix():
    matrix = np.zeros((2, 2), dtype=np.complex128)
    rows = cols = np.ones(2, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, 0)

    # Test holomorphic=False by comparing to the Jacobian
    # of the real and imaginary parts separately
    jacobian = jax.jacobian(perm_wrapper(perm))(matrix, rows, cols)

    matrix, rows, cols = jax.numpy.array(matrix), jax.numpy.array(rows), jax.numpy.array(cols)
    expected = jax.jacobian(perm_wrapper(permanent_with_reduction))(matrix, rows, cols)
    assert np.allclose(jacobian[0], expected[0])
    assert np.allclose(jacobian[1], expected[1])


def test_grad_perm_all_ones():
    matrix = np.ones((2, 2), dtype=np.complex128)
    rows = cols = np.ones(2, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, np.ones((2, 2), dtype=np.complex128))

    # Test holomorphic=False by comparing to the Jacobian
    # of the real and imaginary parts separately
    jacobian = jax.jacobian(perm_wrapper(perm))(matrix, rows, cols)

    matrix, rows, cols = jax.numpy.array(matrix), jax.numpy.array(rows), jax.numpy.array(cols)
    expected = jax.jacobian(perm_wrapper(permanent_with_reduction))(matrix, rows, cols)
    assert np.allclose(jacobian[0], expected[0])
    assert np.allclose(jacobian[1], expected[1])


def test_grad_perm_zero_input_output():
    matrix = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    rows = cols = np.zeros(3, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, 0)


def test_grad_perm_zero_input():
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
    grad = jax.grad(perm, holomorphic=True)(interferometer, input, output)
    assert np.allclose(grad, 0)


def test_grad_perm_no_repetition():
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
    grad = jax.grad(perm, holomorphic=True)(interferometer, input, output)
    assert np.allclose(
        grad,
        np.array(
            [
                [
                    -0.05371236 + 2.66353718e-02j,
                    -0.04582236 - 8.18820862e-02j,
                    0.03960814 + 5.80689287e-02j,
                    -0.02045293 + 1.38801402e-01j,
                    -0.0149418 - 4.19752512e-02j,
                    -0.05229726 - 5.69852986e-02j,
                ],
                [
                    -0.11103513 + 3.88283318e-02j,
                    -0.14391923 + 1.05461540e-02j,
                    0.00725286 - 8.25634953e-03j,
                    0.05221953 + 2.01817507e-02j,
                    0.01545186 - 4.29582156e-03j,
                    -0.00398563 - 7.07909136e-02j,
                ],
                [
                    0.04208331 + 4.16242008e-02j,
                    -0.00862918 - 5.05756669e-03j,
                    0.02234116 + 1.42226190e-02j,
                    -0.02339448 - 3.14244737e-03j,
                    0.05196793 + 5.23028802e-02j,
                    0.00677792 + 9.11863479e-02j,
                ],
                [
                    -0.06956478 + 2.20011711e-02j,
                    0.09569006 + 4.02943688e-02j,
                    -0.05576987 + 4.91285771e-02j,
                    -0.00468075 - 6.13113076e-02j,
                    0.02943685 + 4.46223167e-02j,
                    0.05577522 - 2.92991471e-02j,
                ],
                [
                    0.011846 + 2.46933770e-03j,
                    0.0153745 + 4.67444609e-02j,
                    -0.04146899 + 2.13750603e-02j,
                    0.04391381 + 4.23307063e-02j,
                    -0.0189607 - 2.91139206e-02j,
                    -0.05233688 + 1.27779869e-04j,
                ],
                [
                    -0.03119874 + 7.55431598e-02j,
                    -0.15592711 - 2.44391509e-02j,
                    0.01571884 - 4.49747533e-02j,
                    -0.01909996 - 9.40045056e-02j,
                    -0.02374948 - 7.11344685e-02j,
                    0.04924467 - 3.22028064e-02j,
                ],
            ]
        ),
    )

    # Test holomorphic=False by comparing to the Jacobian
    # of the real and imaginary parts separately
    jacobian = jax.jacobian(perm_wrapper(perm))(interferometer, input, output)

    interferometer, input, output = jax.numpy.array(interferometer), jax.numpy.array(input), jax.numpy.array(output)
    expected = jax.jacobian(perm_wrapper(permanent_with_reduction))(interferometer, input, output)
    assert np.allclose(jacobian[0], expected[0])
    assert np.allclose(jacobian[1], expected[1])


def test_grad_perm_4_by_4():
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
    grad = jax.grad(perm, holomorphic=True)(unitary, rows, cols)
    assert np.allclose(
        grad,
        np.array(
            [
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -3.18426548 - 1.54032741j,
                    -0.09908472 - 0.37279947j,
                ],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.54425976 + 0.15950196j,
                    -0.49015741 - 0.62882923j,
                ],
            ]
        ),
    )

    # Test holomorphic=False by comparing to the Jacobian
    # of the real and imaginary parts separately
    jacobian = jax.jacobian(perm_wrapper(perm))(unitary, rows, cols)

    unitary, rows, cols = jax.numpy.array(unitary), jax.numpy.array(rows), jax.numpy.array(cols)
    expected = jax.jacobian(perm_wrapper(permanent_with_reduction))(unitary, rows, cols)
    assert np.allclose(jacobian[0], expected[0])
    assert np.allclose(jacobian[1], expected[1])


def test_grad_perm_6_by_6():
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
    rows = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint64)
    cols = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(interferometer, rows, cols)
    assert np.allclose(
        grad,
        np.array(
            [
                [
                    0.00613756 - 0.05596387j,
                    0.05478057 + 0.04177092j,
                    0.01846764 + 0.11305476j,
                    0.02605859 + 0.01649895j,
                    -0.01002297 - 0.03639016j,
                    0.02551702 - 0.04844018j,
                ],
                [
                    0.00808861 - 0.01126793j,
                    0.00079495 + 0.05701643j,
                    -0.02790181 - 0.01987287j,
                    0.00725814 - 0.0567719j,
                    0.0763283 + 0.01962546j,
                    0.02445184 + 0.05709078j,
                ],
                [
                    -0.06986495 - 0.00897261j,
                    -0.05773915 - 0.08317558j,
                    -0.02051542 + 0.01104616j,
                    -0.02175885 - 0.04162514j,
                    -0.08400035 - 0.01669359j,
                    0.00111028 - 0.05966386j,
                ],
                [
                    0.05612247 + 0.00807638j,
                    -0.02331525 + 0.00396841j,
                    -0.00055769 - 0.02280161j,
                    -0.04036997 - 0.04793161j,
                    -0.01883466 + 0.01268112j,
                    0.04200545 - 0.02340915j,
                ],
                [
                    0.02595464 - 0.0743598j,
                    -0.04747572 + 0.01271623j,
                    -0.0620618 + 0.00135199j,
                    -0.05118836 - 0.00165811j,
                    0.01362084 - 0.06675524j,
                    -0.09843736 + 0.05753861j,
                ],
                [
                    0.04273109 + 0.00110463j,
                    0.00367212 - 0.00144348j,
                    0.01679765 - 0.00517546j,
                    -0.00579187 - 0.00950378j,
                    -0.02463916 + 0.11107093j,
                    -0.03805067 - 0.05460198j,
                ],
            ]
        ),
    )

    # Test holomorphic=False by comparing to the Jacobian
    # of the real and imaginary parts separately
    jacobian = jax.jacobian(perm_wrapper(perm))(interferometer, rows, cols)

    interferometer, rows, cols = jax.numpy.array(interferometer), jax.numpy.array(rows), jax.numpy.array(cols)
    expected = jax.jacobian(perm_wrapper(permanent_with_reduction))(interferometer, rows, cols)
    assert np.allclose(jacobian[0], expected[0])
    assert np.allclose(jacobian[1], expected[1])


def test_jacobian_no_holomorphic():
    mat = jax.numpy.array(
        [[0.5 + 0.j, -0.8660254 + 0.j], [0.8660254 + 0.j, 0.5 + 0.j]],
        dtype=jax.numpy.complex128,
    )

    rows = jax.numpy.array([2, 0], jax.numpy.uint64)
    cols = jax.numpy.array([2, 0], jax.numpy.uint64)

    jacobian = jax.jacobian(perm_wrapper(perm))(mat, rows, cols)

    real_expected = np.array([[2. + 0.j, 0. + 0.j], [0. + 0.j, 0. + 0.j]], dtype=np.complex128)
    assert np.allclose(jacobian[0], real_expected)

    imag_expected = np.array([[0. - 2.j, 0. + 0.j], [0. + 0.j, 0. + 0.j]], dtype=np.complex128)
    assert np.allclose(jacobian[1], imag_expected)

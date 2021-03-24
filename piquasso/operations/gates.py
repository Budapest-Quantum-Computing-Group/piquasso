#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
from piquasso.core.registry import _register
from piquasso.api.operation import Operation
from piquasso.api.constants import HBAR
from piquasso.api.errors import InvalidParameter

from piquasso._math.linalg import is_square


class _PassiveLinearGate(Operation):
    def __init__(self, passive_representation):
        self._passive_representation = passive_representation


class _LinearGate(Operation):
    def __init__(self, passive_representation, active_representation):
        self._passive_representation = passive_representation
        self._active_representation = active_representation


@_register
class Interferometer(_PassiveLinearGate):
    r"""Applies a general interferometer.

    Args:
        matrix (np.array):
            The representation of the interferometer matrix, corresponding to a
            passive transformation on the one-particle subspace.
    """

    def __init__(self, matrix):
        if not is_square(matrix):
            raise InvalidParameter(
                "The interferometer matrix should be a square matrix."
            )

        self._set_params(matrix=matrix)

        super().__init__(passive_representation=matrix)


@_register
class B(_PassiveLinearGate):
    r"""Applies a beamsplitter operation.

    The matrix representation of the beamsplitter operation
    is

    .. math::
        B = \begin{bmatrix}
            t  & r^* \\
            -r & t
        \end{bmatrix},

    where :math:`t = \cos(\theta)` and :math:`r = e^{- i \phi} \sin(\theta)`.

    Args:
        phi (float): Phase angle of the beamsplitter.
            (defaults to :math:`\phi = \pi/2` that gives a symmetric beamsplitter)
        theta (float): The transmittivity angle of the beamsplitter.
            (defaults to :math:`\theta=\pi/4` that gives a 50-50 beamsplitter)

    """

    def __init__(self, theta=0., phi=np.pi / 4):
        self._set_params(theta=theta, phi=phi)

        t = np.cos(theta)
        r = np.exp(-1j * phi) * np.sin(theta)

        super().__init__(
            np.array(
                [
                    [t, np.conj(r)],
                    [-r, t]
                ]
            )
        )


@_register
class R(_PassiveLinearGate):
    r"""Rotation or phaseshift operation.

    The annihilation and creation operators are evolved in the following
    way:

    .. math::
        P(\phi) \hat{a}_k P(\phi)^\dagger = e^{i \phi} \hat{a}_k \\
        P(\phi) \hat{a}_k^\dagger P(\phi)^\dagger
            = e^{- i \phi} \hat{a}_k^\dagger

    Args:
        phi (float): The angle of the rotation.
    """

    def __init__(self, phi: float):
        self._set_params(phi=phi)

        super().__init__(np.array([[np.exp(1j * phi)]]))


@_register
class MZ(_PassiveLinearGate):
    r"""Mach-Zehnder interferometer.

    .. math::
        MZ(\phi_{int}, \phi_{ext}) =
            B(\pi/4, \pi/2) (R(\phi_{int}) \oplus \mathbb{1})
            B(\pi/4, \pi/2) (R(\phi_{ext}) \oplus \mathbb{1})

    where :math:`\phi_{int}, \phi_{ext} \in \mathbb{R}`.

    Let :math:`MZ(\phi_{int}, \phi_{ext}) =: MZ`. Then

    .. math::
        MZ a_i MZ^\dagger =
            e^{i \phi_{ext}} (e^{i \phi_{int}} - 1) a_i + i (e^{i \phi_{int}} - 1) a_j

    .. math::
        MZ a_j MZ^\dagger =
            i e^{i \phi_{ext}} (e^{i \phi_{int}} + 1) a_i + (1 - e^{i \phi_{int}}) a_j


    Args:
        int (float): The internal angle.
        ext (float): The external angle.
    """

    def __init__(self, *, int_: float, ext: float):
        self._set_params(int_=int_, ext=ext)

        int_phase, ext_phase = np.exp(1j * np.array([int_, ext]))

        super().__init__(
            1/2 * np.array(
                [
                    [ext_phase * (int_phase - 1), 1j * (int_phase + 1)],
                    [1j * ext_phase * (int_phase + 1), 1 - int_phase]
                ]
            )
        )


@_register
class F(R):
    r"""Fourier gate.

    Corresponds to the Rotaton gate :class:`R` with :math:`\phi = \pi/2`.
    """

    def __init__(self):
        super().__init__(phi=np.pi/2)


@_register
class GaussianTransform(_LinearGate):
    """Applies a transformation to the state.

    Args:
        P (np.array):
            The representation of the passive transformation on the one-particle
            subspace.
        A (np.array):
            The representation of the active transformation on the one-particle
            subspace.
    """

    def __init__(self, P, A):
        self._set_params(P=P, A=A)

        super().__init__(
            passive_representation=P,
            active_representation=A,
        )


@_register
class S(_LinearGate):
    r"""Applies the squeezing operator.

    The definition of the operator is:

    .. math::
        S(z) = \exp\left(\frac{1}{2}(z^* a^2 -z {a^\dagger}^2)\right)

    where :math:`z = r e^{i\theta}`. The :math:`r` parameter is the amplitude of the
    squeezing and :math:`\theta` is the angle of the squeezing.

    This act of squeezing at a given rotation angle :math:`\theta` results in a
    shrinkage in the :math:`\hat{x}` quadrature and a stretching in the other quadrature
    :math:`\hat{p}` as follows:

    .. math::
        S^\dagger(z) x_{\theta} S(z) =
            e^{-r} x_{\theta}, \: S^\dagger(z) p_{\theta} S(z) = e^{r} p_{\theta}

    The action of the :math:`\hat{S}(z)` gate on the ladder operators :math:`\hat{a}`
    and :math:`\hat{a}^\dagger` can be defined as follows:

    .. math::
        {S(z)}^{\dagger}\hat{a}S(z) =
            \alpha\hat{a} - \beta \hat{a}^{\dagger} \\
            {S(z)}^{\dagger}\hat{a}^\dagger S(z) =
            \alpha\hat{a}^\dagger - \beta^* \hat{a}

    where :math:`\alpha` and :math:`\beta` are :math:`\cosh(amp)`,
    :math:`e^{i\theta}\sinh(amp)` respectively.

    Args:
        amp (float): The amplitude of the squeezing operation.
        theta (float): The squeezing angle.
    """

    def __init__(self, amp, theta=0):
        self._set_params(amp=amp, theta=theta)

        super().__init__(
            passive_representation=np.array(
                [
                    [np.cosh(amp)]
                ]
            ),
            active_representation=np.array(
                [
                    [- np.sinh(amp) * np.exp(1j * theta)]
                ]
            ),
        )


@_register
class P(_LinearGate):
    r"""Applies the quadratic phase operation to the state.

    The operator of the quadratic phase gate is

    .. math::
        P(s) = \exp (i \frac{s \hat{x}}{2\hbar}),

    and it evolves the annihilation operator as

    .. math::
        P(s)^\dagger a_i P(s) = (1 + i \frac{s}{2}) a_i + i \frac{s}{2} a_i^\dagger.
    """

    def __init__(self, s):
        self._set_params(s=s)

        super().__init__(
            passive_representation=np.array([[1 + s/2 * 1j]]),
            active_representation=np.array([[s/2 * 1j]]),
        )


@_register
class S2(_LinearGate):
    r"""2-mode squeezing gate.

    .. math::
        S a_1 S^\dagger = a_1 \cosh r + a_2^\dagger \exp(i \phi)
        S a_2 S^\dagger = a_2 \cosh r + a_1^\dagger \exp(i \phi)

    Args:
        r (float): The amplitude of the squeezing operation.
        phi (float): The squeezing angle.
    """

    def __init__(self, r, phi):
        self._set_params(r=r, phi=phi)

        super().__init__(
            passive_representation=np.array(
                [
                    [np.cosh(r), 0],
                    [0, np.cosh(r)],
                ]
            ),
            active_representation=np.array(
                [
                    [0, np.sinh(r) * np.exp(1j * phi)],
                    [np.sinh(r) * np.exp(1j * phi), 0],
                ]
            ),
        )


@_register
class CX(_LinearGate):
    def __init__(self, s):
        self._set_params(s=s)

        super().__init__(
            passive_representation=np.array(
                [
                    [    1, - s / 2],
                    [s / 2,       1],
                ]
            ),
            active_representation=np.array(
                [
                    [    0, s / 2],
                    [s / 2,     0],
                ]
            ),
        )


@_register
class CZ(_LinearGate):
    def __init__(self, s):
        self._set_params(s=s)

        super().__init__(
            passive_representation=np.array(
                [
                    [           1, 1j * (s / 2)],
                    [1j * (s / 2),            1],
                ]
            ),
            active_representation=np.array(
                [
                    [           0, 1j * (s / 2)],
                    [1j * (s / 2),            0],
                ]
            ),
        )


@_register
class D(Operation):
    r"""Phase space displacement operation.

    One must either specify `alpha` only, or the combination of `r` and `phi`.

    When `r` and `phi` are the given parameters, `alpha` is calculated via:

    .. math:
        \alpha = r \exp(i \phi).

    See:
        :ref:`gaussian_displacement`

    Args:
        alpha (complex): The displacement.
        r (float): The displacement magnitude.
        phi (float): The displacement angle.
    """

    def __init__(self, *, alpha=None, r=None, phi=None):
        if (
            alpha is None and (phi is None or r is None)
            or
            alpha is not None and (phi is not None or r is not None)
        ):
            raise InvalidParameter(
                "Either specify 'alpha' only, or the combination of 'r' and 'phi': "
                f"alpha={alpha}, r={r}, phi={phi}."
            )

        if alpha is None:
            alpha = r * np.exp(1j * phi)

        super().__init__(alpha=alpha)


@_register
class X(D):
    r"""Position displacement gate."""

    def __init__(self, x: float):
        super().__init__(
            alpha=(x / np.sqrt(2 * HBAR)),
        )


@_register
class Z(D):
    r"""Momentum displacement gate."""

    def __init__(self, p: float):
        super().__init__(
            alpha=(1j * p / np.sqrt(2 * HBAR)),
        )


@_register
class K(Operation):
    r"""Kerr gate.

    .. math::
        K(\xi) = \exp(i \xi \hat{n} \hat{n})

    .. math::
        K(\xi) a K(\xi) = a \exp(- i \xi (1 + 2 n))

    Args:
        xi (float): The magnitude of the Kerr nonlinear term.
    """

    def __init__(self, *, xi: float):
        super().__init__(xi=xi)


@_register
class CK(Operation):
    r"""Cross-Kerr gate.

    .. math::
        CK(\xi) = \exp(i \xi \hat{n}_i \hat{n}_j)

    .. math::
        CK(\xi) a_i CK(\xi) &= a_i \exp(- i \xi n_j) \\
        CK(\xi) a_j CK(\xi) &= a_j \exp(- i \xi n_i)

    Args:
        xi (float): The magnitude of the Cross-Kerr nonlinear term.
    """

    def __init__(self, *, xi: float):
        super().__init__(xi=xi)


@_register
class Sampling(Operation):
    r"""Boson Sampling.

    Simulates a boson sampling using generalized Clifford&Clifford algorithm
    from [Brod, Oszmaniec 2020].

    This method assumes that initial_state is given in the second quantization
    description (mode occupation). BoSS requires input states as numpy arrays,
    therefore the state is prepared as such structure.

    Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
    as it allows effective simulation of broader range of input states than original
    algorithm.

    Args:
        shots (int):
            A positive integer value representing number of samples for the experiment.
    """

    def __init__(self, shots=1):
        if shots < 1 or not isinstance(shots, int):
            raise InvalidParameter(
                f"The number of shots should be a positive integer: shots={shots}."
            )

        super().__init__(shots=shots)

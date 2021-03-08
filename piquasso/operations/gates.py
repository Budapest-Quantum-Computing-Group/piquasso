#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
from piquasso.core.registry import _register
from piquasso.api.operation import Operation, ModelessOperation


@_register
class PassiveTransform(Operation):
    r"""Applies a general passive transformation.

    Args:
        T (np.array):
            The representation of the passive transformation on the one-particle
            subspace.
    """

    def __init__(self, T):
        self._passive_representation = T


@_register
class B(Operation):
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
        super().__init__(theta, phi)

        self._passive_representation = self._get_passive_representation()

    def _get_passive_representation(self):
        theta, phi = self.params

        t = np.cos(theta)
        r = np.exp(-1j * phi) * np.sin(theta)

        return np.array([
            [t, np.conj(r)],
            [-r, t]
        ])


@_register
class R(Operation):
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
        super().__init__(phi)

        self._passive_representation = self._get_passive_representation()

    def _get_passive_representation(self):
        phi = self.params[0]
        phase = np.exp(1j * phi)

        return np.array([[phase]])


@_register
class MZ(Operation):
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
        super().__init__(int_, ext)

        self._passive_representation = self._get_passive_representation()

    def _get_passive_representation(self):
        int_phase, ext_phase = np.exp(1j * np.array(self.params))

        return 1/2 * np.array(
            [
                [ext_phase * (int_phase - 1), 1j * (int_phase + 1)],
                [1j * ext_phase * (int_phase + 1), 1 - int_phase]
            ]
        )


@_register
class F(R):
    r"""Fourier gate.

    Corresponds to the Rotaton gate :class:`R` with :math:`\phi = \pi/2`.
    """

    def __init__(self):
        super().__init__(phi=np.pi/2)


@_register
class GaussianTransform(Operation):
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
        super().__init__(P, A)

        self._passive_representation = P
        self._active_representation = A


@_register
class S(Operation):
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
        super().__init__(amp, theta)

        self._calculate_representations()

    def _calculate_representations(self):
        if len(self.params) == 1:
            theta = 0
        else:
            theta = self.params[1]

        alpha = np.cosh(self.params[0])

        beta = np.sinh(self.params[0]) * np.exp(1j * theta)

        self._passive_representation = np.array([[alpha]])

        self._active_representation = np.array([[- beta]])


@_register
class P(Operation):
    r"""Applies the quadratic phase operation to the state.

    The operator of the quadratic phase gate is

    .. math::
        P(s) = \exp (i \frac{s \hat{x}}{2\hbar}),

    and it evolves the annihilation operator as

    .. math::
        P(s)^\dagger a_i P(s) = (1 + i \frac{s}{2}) a_i + i \frac{s}{2} a_i^\dagger.
    """

    def __init__(self, s):
        super().__init__(s)

        self._calculate_representations()

    def _calculate_representations(self):
        s = self.params[0]

        self._passive_representation = np.array([[1 + s/2 * 1j]])

        self._active_representation = np.array([[s/2 * 1j]])


@_register
class S2(Operation):
    r"""2-mode squeezing gate.

    .. math::
        S a_1 S^\dagger = a_1 \cosh r + a_2^\dagger \exp(i \phi)
        S a_2 S^\dagger = a_2 \cosh r + a_1^\dagger \exp(i \phi)

    Args:
        r (float): The amplitude of the squeezing operation.
        phi (float): The squeezing angle.
    """

    def __init__(self, r, phi):
        super().__init__(r, phi)

        self._calculate_representations()

    def _calculate_representations(self):
        r = self.params[0]
        phi = self.params[1]

        self._passive_representation = np.array(
            [
                [np.cosh(r), 0],
                [0, np.cosh(r)],
            ]
        )

        self._active_representation = np.array(
            [
                [0, np.sinh(r) * np.exp(1j * phi)],
                [np.sinh(r) * np.exp(1j * phi), 0],
            ]
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
        assert \
            alpha is not None and r is None and phi is None \
            or \
            alpha is None and r is not None and phi is not None, \
            "Either specify 'alpha' only, or the combination of 'r' and 'phi'."

        if alpha is None:
            alpha = r * np.exp(1j * phi)

        super().__init__(alpha)


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
        super().__init__(xi)


@_register
class Interferometer(Operation):
    """Interferometer.

    Adds additional interferometer to the effective interferometer.

    This can be interpreted as placing another interferometer in the network, just
    before performing the sampling. This operation is realized by multiplying
    current effective interferometer matrix with new interferometer matrix.

    Do note, that new interferometer matrix works as interferometer matrix on
    qumodes (provided as the arguments) and as an identity on every other mode.

    Args:
        interferometer_matrix (np.array):
            A square matrix representing the interferometer.
    """

    @staticmethod
    def _is_square(matrix):
        shape = matrix.shape
        return len(shape) == 2 and shape[0] == shape[1]

    def __init__(self, interferometer_matrix):
        assert \
            self._is_square(interferometer_matrix), \
            "The interferometer matrix should be a square matrix."
        super().__init__(interferometer_matrix)


@_register
class Sampling(ModelessOperation):
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
        assert \
            shots > 0 and isinstance(shots, int),\
            "The number of shots should be a positive integer."
        super().__init__(shots)

#
# Copyright 2021 Budapest Quantum Computing Group
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

import numpy as np

from scipy.optimize import root_scalar

from piquasso.api.instruction import Instruction
from piquasso.api.constants import HBAR
from piquasso.api.errors import InvalidParameter

from piquasso._math.takagi import takagi
from piquasso._math.linalg import is_square, is_symmetric, is_symplectic


class _BogoliubovTransformation(Instruction):
    def __init__(
        self,
        *,
        passive_block=None,
        active_block=None,
        displacement_vector=None,
    ):
        self._passive_block = passive_block
        self._active_block = active_block
        self._displacement_vector = displacement_vector


class _ScalableBogoliubovTransformation(_BogoliubovTransformation):
    ERROR_MESSAGE_TEMPLATE = (
        "The instruction {instruction} is not applicable to modes {modes} with the "
        "specified parameters."
    )

    def _autoscale(self):
        if (
            self._passive_block is None
            or len(self.modes) == len(self._passive_block)
        ):
            pass
        elif len(self._passive_block) == 1:
            from scipy.linalg import block_diag
            self._passive_block = block_diag(
                *[self._passive_block] * len(self.modes)
            )
        else:
            raise InvalidParameter(
                self.ERROR_MESSAGE_TEMPLATE.format(instruction=self, modes=self.modes)
            )

        if (
            self._active_block is None
            or len(self.modes) == len(self._active_block)
        ):
            pass
        elif len(self._active_block) == 1:
            from scipy.linalg import block_diag
            self._active_block = block_diag(
                *[self._active_block] * len(self.modes)
            )
        else:
            raise InvalidParameter(
                self.ERROR_MESSAGE_TEMPLATE.format(instruction=self, modes=self.modes)
            )

        if (
            self._displacement_vector is None
            or len(self.modes) == len(self._displacement_vector)
        ):
            pass
        elif len(self._displacement_vector) == 1:
            self._displacement_vector = np.array(
                [self._displacement_vector[0]] * len(self.modes),
                dtype=complex,
            )
        else:
            raise InvalidParameter(
                self.ERROR_MESSAGE_TEMPLATE.format(instruction=self, modes=self.modes)
            )


class Interferometer(_BogoliubovTransformation):
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

        super().__init__(passive_block=matrix)


class Beamsplitter(_BogoliubovTransformation):
    r"""Applies a beamsplitter instruction.

    The matrix representation of the beamsplitter instruction
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
        r = np.exp(1j * phi) * np.sin(theta)

        super().__init__(
            passive_block=np.array(
                [
                    [t, -np.conj(r)],
                    [r, t],
                ]
            )
        )


class Phaseshifter(_ScalableBogoliubovTransformation):
    r"""Rotation or phaseshift instruction.

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

        super().__init__(
            passive_block=np.diag(np.exp(1j * np.atleast_1d(phi)))
        )


class MachZehnder(_BogoliubovTransformation):
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
            passive_block=1/2 * np.array(
                [
                    [ext_phase * (int_phase - 1), 1j * (int_phase + 1)],
                    [1j * ext_phase * (int_phase + 1), 1 - int_phase]
                ]
            )
        )


class Fourier(_ScalableBogoliubovTransformation):
    r"""Fourier gate.

    Corresponds to the Rotaton gate :class:`R` with :math:`\phi = \pi/2`.
    """

    def __init__(self):
        super().__init__(passive_block=np.array([[1j]]))


class GaussianTransform(_BogoliubovTransformation):
    """Applies a Gaussian transformation to the state.

    Args:
        passive (np.ndarray):
            The passive submatrix of the symplectic matrix corresponding to the
            transformation.
        active (np.ndarray):
            The active submatrix of the symplectic matrix corresponding to the
            transformation.
    """

    def __init__(self, passive, active):
        if not is_symplectic(
            np.block([[passive, active], [active.conj(), passive.conj()]])
        ):
            raise InvalidParameter(
                "The input parameters for instruction 'GaussianTransform' do not form "
                "a symplectic matrix."
            )

        self._set_params(passive=passive, active=active)

        super().__init__(
            passive_block=passive,
            active_block=active,
        )


class Squeezing(_ScalableBogoliubovTransformation):
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
        r (float): The amplitude of the squeezing instruction.
        phi (float): The squeezing angle.
    """

    def __init__(self, r, phi=0):
        self._set_params(r=r, phi=phi)

        super().__init__(
            passive_block=np.diag(
                np.atleast_1d(np.cosh(r))
            ),
            active_block=np.diag(
                - np.atleast_1d(np.sinh(r)) * np.exp(1j * np.atleast_1d(phi))
            ),
        )


class QuadraticPhase(_ScalableBogoliubovTransformation):
    r"""Applies the quadratic phase instruction to the state.

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
            passive_block=np.diag(1 + np.atleast_1d(s)/2 * 1j),
            active_block=np.diag(np.atleast_1d(s)/2 * 1j),
        )


class Squeezing2(_BogoliubovTransformation):
    r"""2-mode squeezing gate.

    .. math::
        S a_1 S^\dagger = a_1 \cosh r + a_2^\dagger \exp(i \phi)
        S a_2 S^\dagger = a_2 \cosh r + a_1^\dagger \exp(i \phi)

    Args:
        r (float): The amplitude of the squeezing instruction.
        phi (float): The squeezing angle.
    """

    def __init__(self, r, phi):
        self._set_params(r=r, phi=phi)

        super().__init__(
            passive_block=np.array(
                [
                    [np.cosh(r), 0],
                    [0, np.cosh(r)],
                ]
            ),
            active_block=np.array(
                [
                    [0, np.sinh(r) * np.exp(1j * phi)],
                    [np.sinh(r) * np.exp(1j * phi), 0],
                ]
            ),
        )


class ControlledX(_BogoliubovTransformation):
    def __init__(self, s):
        self._set_params(s=s)

        super().__init__(
            passive_block=np.array(
                [
                    [    1, - s / 2],
                    [s / 2,       1],
                ]
            ),
            active_block=np.array(
                [
                    [    0, s / 2],
                    [s / 2,     0],
                ]
            ),
        )


class ControlledZ(_BogoliubovTransformation):
    def __init__(self, s):
        self._set_params(s=s)

        super().__init__(
            passive_block=np.array(
                [
                    [           1, 1j * (s / 2)],
                    [1j * (s / 2),            1],
                ]
            ),
            active_block=np.array(
                [
                    [           0, 1j * (s / 2)],
                    [1j * (s / 2),            0],
                ]
            ),
        )


class Displacement(_ScalableBogoliubovTransformation):
    r"""Phase space displacement instruction.

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

        if alpha is not None and r is None and phi is None:
            self._set_params(alpha=alpha)
            alpha = np.atleast_1d(alpha)
        elif alpha is None and r is not None and phi is not None:
            self._set_params(r=r, phi=phi)
            alpha = np.atleast_1d(r) * np.exp(1j * np.atleast_1d(phi))
        else:
            raise InvalidParameter(
                "Either specify 'alpha' only, or the combination of 'r' and 'phi': "
                f"alpha={alpha}, r={r}, phi={phi}."
            )

        super().__init__(displacement_vector=alpha)


class PositionDisplacement(_ScalableBogoliubovTransformation):
    r"""Position displacement gate."""

    def __init__(self, x: float):
        self._set_params(x=x)

        super().__init__(
            displacement_vector=np.atleast_1d(x) / np.sqrt(2 * HBAR),
        )


class MomentumDisplacement(_ScalableBogoliubovTransformation):
    r"""Momentum displacement gate."""

    def __init__(self, p: float):
        self._set_params(p=p)

        super().__init__(
            displacement_vector=1j * np.atleast_1d(p) / np.sqrt(2 * HBAR),
        )


class Kerr(Instruction):
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


class CrossKerr(Instruction):
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


class Sampling(Instruction):
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


class Graph(Instruction):
    r"""Applies a graph given its adjacency matrix, see
    https://arxiv.org/pdf/1612.01199.pdf
    """

    def __init__(self, adjacency_matrix, mean_photon_number=1.0):
        super().__init__(
            adjacency_matrix=adjacency_matrix,
            mean_photon_number=mean_photon_number
        )

        if not is_symmetric(adjacency_matrix):
            raise InvalidParameter("The adjacency matrix should be symmetric.")

        singular_values, unitary = takagi(adjacency_matrix)

        scaling = self._get_scaling(singular_values, mean_photon_number)

        squeezing_parameters = np.arctanh(scaling * singular_values)

        # TODO: find a better solution for these.
        self._squeezing = GaussianTransform(
            passive=np.diag(np.cosh(squeezing_parameters)),
            active=np.diag(np.sinh(squeezing_parameters)),
        )

        self._interferometer = Interferometer(unitary)

    def _get_scaling(self, singular_values, mean_photon_number):
        r"""
        For a squeezed state :math:`rho` the mean photon number is calculated by

        .. math::
            \langle \hat{n} \rangle_\rho = \sum_{i = 0}^d \mathrm{sinh}(r_i)^2

        where :math:`r_i = \mathrm{arctan}(s_i)`, where :math:`s_i` are the singular
        values of the adjacency matrix.
        """

        def mean_photon_number_equation(scaling):
            return sum(
                (scaling * singular_value) ** 2 / (1 - (scaling * singular_value) ** 2)
                for singular_value
                in singular_values
            ) / len(singular_values) - mean_photon_number

        def mean_photon_number_gradient(scaling):
            return (
                (2.0 / scaling)
                * np.sum(
                    (
                        singular_values * scaling
                        / (1 - (singular_values * scaling) ** 2)
                    ) ** 2
                )
            )

        lower_bound = 0.0

        tolerance = 1e-10  # Needed to avoid zero division.

        upper_bound = 1.0 / (max(singular_values) + tolerance)

        result = root_scalar(
            mean_photon_number_equation,
            fprime=mean_photon_number_gradient,
            x0=(lower_bound - upper_bound) / 2.0,
            bracket=(lower_bound, upper_bound),
        )

        if not result.converged:
            raise InvalidParameter(
                f"No scaling found for adjacency matrix: {self.adjacency_matrix}."
            )

        return result.root

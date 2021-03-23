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

r"""
Gates can be characterized by a unitary operator :math:`\hat{U}`, which evolves the
quantum state in the following manner

.. math::
    \rho' = \hat{U} \rho \hat{U}^\dagger

Gates with at most quadratic Hamiltonians are called linear gates. Evolution of the
ladder operators by linear gates could be expressed in the form

.. math::

    \hat{U} \hat{\xi} \hat{U}^\dagger = S_{(c)} \hat{\xi} + \vec{d},

where :math:`S_{(c)} \in \operatorname{Sp}(2d, \mathbb{R})`,
:math:`\vec{d} \in \mathbb{C}^{2d}`,

.. math::
    \hat{\xi} = \begin{bmatrix}
        \hat{a} \\
        \hat{a}^\dagger
    \end{bmatrix},

where :math:`\hat{a}^\dagger` and :math:`\hat{a}` are the multimode creation and
annihilation operators, respectively.

Most of the gates defined here are linear gates, which can be characterized by
:math:`S_{(c)}` and :math:`\vec{d}`.
"""

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
    r"""Applies a general interferometer gate.

    The evolution of the ladder operators can be described by

    .. math::
        S_{(c)} = \begin{bmatrix}
            U & 0_{d \times d} \\
            0_{d \times d} & U^*
        \end{bmatrix},

    where :math:`U \in \mathbb{C}^{d \times d}` is a unitary matrix.

    Args:
        matrix (numpy.ndarray):
            The unitary interferometer matrix, corresponding to a
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
    r"""Applies a beamsplitter gate.

    The symplectic representation of the beamsplitter gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            t  & -r^* &    & \\
            r & t   &    & \\
               &     & t  & -r \\
               &     & r^* & t
        \end{bmatrix},

    where :math:`t = \cos(\theta)` and :math:`r = e^{i \phi} \sin(\theta)`.

    Args:
        phi (float):
            Phase angle of the beamsplitter.
            (defaults to :math:`\phi = \pi/2` that gives a symmetric beamsplitter)
        theta (float):
            The transmittivity angle of the beamsplitter.
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
    r"""Applies a rotation or phaseshift gate.

    The symplectic representation of the phaseshifter gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            e^{i \phi} & 0 \\
            0 & e^{- i \phi}
        \end{bmatrix}

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
        S_{(c)} = \begin{bmatrix}
        e^{i \phi_{ext} } (e^{i \phi_{int} } - 1)   & i (e^{i \phi_{int} } + 1) & & \\
        i e^{i \phi_{ext} } (e^{i \phi_{int} } + 1) & 1 - e^{i \phi_{int} }     & & \\
        & & e^{-i \phi_{ext} } (e^{-i \phi_{int} } - 1) & -i (e^{-i \phi_{int} } + 1) \\
        & & -i e^{-i \phi_{ext} } (e^{-i \phi_{int} } + 1) & 1 - e^{-i \phi_{int} }
        \end{bmatrix},

    where :math:`\phi_{int}, \phi_{ext} \in \mathbb{R}`.

    The Mach-Zehnder interferometer is equivalent to

    .. math::
        MZ(\phi_{int}, \phi_{ext}) =
            B(\pi/4, \pi/2) (R(\phi_{int}) \oplus \mathbb{1})
            B(\pi/4, \pi/2) (R(\phi_{ext}) \oplus \mathbb{1})


    Args:
        int (float): The internal angle.
        ext (float): The external angle.
    """

    def __init__(self, int_: float, ext: float):
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
    r"""Applies a Fourier gate.

    The symplectic representation of the phaseshifter gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            i & 0  \\
            0 & -i \\
        \end{bmatrix}

    Note:
        Corresponds to the :class:`Phaseshifter` gate :class:`R` with
        :math:`\phi = \pi/2`.
    """

    def __init__(self):
        super().__init__(passive_block=np.array([[1j]]))


class GaussianTransform(_BogoliubovTransformation):
    r"""Applies a Gaussian transformation gate.

    The symplectic representation of the Gaussian gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            P & A \\
            A^* & P^*
        \end{bmatrix},

    where :math:`P, A \in C^{d \times d}` fulfilling the relations

    .. math::
        P P^\dagger - A A^\dagger &= I_{d \times d} \\
        P A^T &= A P^T

    Args:
        passive (numpy.ndarray):
            The passive submatrix of the symplectic matrix corresponding to the
            transformation.
        active (numpy.ndarray):
            The active submatrix of the symplectic matrix corresponding to the
            transformation.

    Raises:
        InvalidParameters: Raised if the parameters do not form a symplectic matrix.
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
    r"""Applies the squeezing gate.

    The symplectic representation of the squeezing gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            \cosh r & - e^{i phi} \sinh r \\
            - e^{- i phi} \sinh r & \cosh r
        \end{bmatrix}.

    The unitary squeezing operator is

    .. math::
        S(z) = \exp ( \frac{1}{2}(z^* a^2 -z a^{\dagger 2} ),

    where :math:`z \in \mathbb{C}^{d \times d}` is a symmetric matrix.

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

    .. math::
        S_{(c)} = \begin{bmatrix}
            1 + i \frac{s}{2} & i \frac{s}{2} \\
            -i \frac{s}{2} & 1 - i \frac{s}{2}
        \end{bmatrix}.

    """

    def __init__(self, s):
        self._set_params(s=s)

        super().__init__(
            passive_block=np.diag(1 + np.atleast_1d(s)/2 * 1j),
            active_block=np.diag(np.atleast_1d(s)/2 * 1j),
        )


class Squeezing2(_BogoliubovTransformation):
    r"""Applies the two-mode squeezing gate to the state.

    .. math::
        S_{(c)} = \begin{bmatrix}
            \cosh r & 0 & 0 & e^{i \phi} \sinh r \\
            0 & \cosh r & e^{i \phi} \sinh r & 0 \\
            0 & e^{- i \phi} \sinh r & \cosh r & 0 \\
            e^{- i \phi} \sinh r & 0 & 0 & \cosh r
        \end{bmatrix}.

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
    r"""Applies the controlled X gate to the state.

    .. math::
        S_{(z)} = \begin{bmatrix}
            1            & - \frac{s}{2} & 0           & \frac{s}{2} \\
            \frac{s}{2} & 1              & \frac{s}{2} & 0            \\
            0            & \frac{s}{2} & 1            & - \frac{s}{2} \\
            \frac{s}{2} & 0            & \frac{s}{2} & 1
        \end{bmatrix}.
    """

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
    r"""Applies the controlled Z gate to the state.

    .. math::
        S_{(z)} = \begin{bmatrix}
            1            & i \frac{s}{2} & 0            & i \frac{s}{2}    \\
            i \frac{s}{2} & 1              & i \frac{s}{2} & 0             \\
            0            & -i \frac{s}{2} & 1            & - i \frac{s}{2} \\
            -i \frac{s}{2} & 0            & -i \frac{s}{2} & 1
        \end{bmatrix}.
    """
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

    Evolves the ladder operators by

    .. math::
        \hat{\xi} \mapsto \hat{\xi} + \begin{bmatrix} \alpha \\ \alpha^* \end{bmatrix},

    where :math:`\alpha \in \mathbb{C}^{d \times d}`.

    One must either specify `alpha` only, or the combination of `r` and `phi`.
    When `r` and `phi` are the given parameters, `alpha` is calculated via:

    .. math:
        \alpha = r \exp(i \phi).

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

    Note:
        This is a non-linear gate, therefore it couldn't be used with
        :class:`~piquasso._backends.gaussian.state.GaussianState`.

    .. math::
        K(\xi) = \exp(i \xi \hat{n} \hat{n})

    .. math::
        K(\xi) a K(\xi) = a \exp(- i \xi (1 + 2 n))

    Args:
        xi (float): The magnitude of the Kerr nonlinear term.
    """

    def __init__(self, xi: float):
        super().__init__(xi=xi)


class CrossKerr(Instruction):
    r"""Cross-Kerr gate.

    Note:
        This is a non-linear gate, therefore it couldn't be used with
        :class:`~piquasso._backends.gaussian.state.GaussianState`.

    .. math::
        CK(\xi) = \exp(i \xi \hat{n}_i \hat{n}_j)

    .. math::
        CK(\xi) a_i CK(\xi) &= a_i \exp(- i \xi n_j) \\
        CK(\xi) a_j CK(\xi) &= a_j \exp(- i \xi n_i)

    Args:
        xi (float): The magnitude of the Cross-Kerr nonlinear term.
    """

    def __init__(self, xi: float):
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

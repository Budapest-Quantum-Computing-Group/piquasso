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
Gates can be characterized by a unitary operator :math:`U`, which evolves the
quantum state in the following manner

.. math::
    \rho' = U \rho U^\dagger

Gates with at most quadratic Hamiltonians are called linear gates. Evolution of the
ladder operators by linear gates could be expressed in the form

.. math::

    U \xi U^\dagger = S_{(c)} \xi + d,

where :math:`S_{(c)} \in \operatorname{Sp}(2d, \mathbb{R})`,
:math:`d \in \mathbb{C}^{2d}`,

.. math::
    \xi = \begin{bmatrix}
        a_1, \dots, a_d, a_1^\dagger, \dots, a_d^\dagger
    \end{bmatrix}^T,

where :math:`a_1^\dagger, ..., a_d^\dagger` and :math:`a_1, \dots, a_d` are the
creation and annihilation operators, respectively.

Most of the gates defined here are linear gates, which can be characterized by
:math:`S_{(c)}` and :math:`d`.
"""

import numpy as np

from scipy.optimize import root_scalar
from scipy.linalg import block_diag

from piquasso.api.instruction import Gate
from piquasso.api.errors import InvalidParameter

from piquasso._math.decompositions import takagi
from piquasso._math.linalg import is_square, is_symmetric, is_invertible
from piquasso._math.symplectic import complex_symplectic_form, is_symplectic

from piquasso.core import _mixins


class _BogoliubovTransformation(Gate):
    def __init__(
        self,
        *,
        params: dict = None,
        passive_block: np.ndarray = None,
        active_block: np.ndarray = None,
        displacement_vector: np.ndarray = None,
    ):
        params = params or {}

        super().__init__(
            params=params,
            extra_params=dict(
                passive_block=passive_block,
                active_block=active_block,
                displacement_vector=displacement_vector,
            ),
        )


class _ScalableBogoliubovTransformation(
    _BogoliubovTransformation,
    _mixins.ScalingMixin,
):
    ERROR_MESSAGE_TEMPLATE = (
        "The instruction {instruction} is not applicable to modes {modes} with the "
        "specified parameters."
    )

    def _autoscale(self) -> None:

        passive_block = self._extra_params["passive_block"]
        if passive_block is None or len(self.modes) == len(passive_block):
            pass
        elif len(passive_block) == 1:
            self._extra_params["passive_block"] = block_diag(
                *[passive_block] * len(self.modes)
            )
        else:
            raise InvalidParameter(
                self.ERROR_MESSAGE_TEMPLATE.format(instruction=self, modes=self.modes)
            )

        active_block = self._extra_params["active_block"]
        if active_block is None or len(self.modes) == len(active_block):
            pass
        elif len(active_block) == 1:
            self._extra_params["active_block"] = block_diag(
                *[active_block] * len(self.modes)
            )
        else:
            raise InvalidParameter(
                self.ERROR_MESSAGE_TEMPLATE.format(instruction=self, modes=self.modes)
            )

        displacement_vector = self._extra_params["displacement_vector"]
        if displacement_vector is None or len(self.modes) == len(displacement_vector):
            pass
        elif len(displacement_vector) == 1:
            self._extra_params["displacement_vector"] = np.array(
                [displacement_vector[0]] * len(self.modes),
                dtype=complex,
            )
        else:
            raise InvalidParameter(
                self.ERROR_MESSAGE_TEMPLATE.format(instruction=self, modes=self.modes)
            )


class Interferometer(_BogoliubovTransformation):
    r"""Applies a general interferometer gate.

    The general unitary operator can be written as

    .. math::
        U = \exp \left (
            i \sum_{i, j = 1}^d H_{i j} a_i^\dagger a_j
        \right ),

    where the parameter `U` and :math:`H` is related by

    .. math::
        U = \exp \left (
            i H \right
        ).

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

    def __init__(self, matrix: np.ndarray) -> None:
        if not is_square(matrix):
            raise InvalidParameter(
                "The interferometer matrix should be a square matrix."
            )

        super().__init__(params=dict(matrix=matrix), passive_block=matrix)


class Beamsplitter(_BogoliubovTransformation):
    r"""Applies a beamsplitter gate.

    The general unitary operator can be written as

    .. math::
        BS_{ij} (\theta, \phi) = \exp \left (
            \theta e^{i \phi} a^\dagger_i a_j
            - \theta e^{- i \phi} a^\dagger_j a_i
        \right ).

    The symplectic representation of the beamsplitter gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            t  & -r^* &    & \\
            r & t   &    & \\
               &     & t  & -r \\
               &     & r^* & t
        \end{bmatrix},

    where :math:`t = \cos(\theta)` and :math:`r = e^{i \phi} \sin(\theta)`.
    """

    def __init__(self, theta: float = 0.0, phi: float = np.pi / 4) -> None:
        """
        Args:
            phi (float):
                Phase angle of the beamsplitter.
                (defaults to :math:`\phi = \pi/2` that gives a symmetric beamsplitter)
            theta (float):
                The transmittivity angle of the beamsplitter.
                (defaults to :math:`\theta=\pi/4` that gives a 50-50 beamsplitter)
        """

        t = np.cos(theta)
        r = np.exp(1j * phi) * np.sin(theta)

        super().__init__(
            params=dict(
                theta=theta,
                phi=phi,
            ),
            passive_block=np.array(
                [
                    [t, -np.conj(r)],
                    [r, t],
                ]
            ),
        )


class Phaseshifter(_ScalableBogoliubovTransformation):
    r"""Applies a rotation or a phaseshifter gate.

    The unitary operator corresponding to the phaseshifter gate on the :math:`i`-th mode
    is

    .. math::
        R_i (\phi) = \exp \left (
            i \phi a_i^\dagger a_i
        \right ).

    The symplectic representation of the phaseshifter gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            e^{i \phi} & 0 \\
            0 & e^{- i \phi}
        \end{bmatrix}.
    """

    def __init__(self, phi: float) -> None:
        """
        Args:
            phi (float): The angle of the rotation.
        """

        super().__init__(
            params=dict(phi=phi), passive_block=np.diag(np.exp(1j * np.atleast_1d(phi)))
        )


class MachZehnder(_BogoliubovTransformation):
    r"""Mach-Zehnder interferometer.

    The Mach-Zehnder interferometer is equivalent to

    .. math::
        MZ(\phi_{int}, \phi_{ext}) =
            B(\pi/4, \pi/2) (R(\phi_{int}) \oplus \mathbb{1})
            B(\pi/4, \pi/2) (R(\phi_{ext}) \oplus \mathbb{1}).

    The symplectic representation of the Mach-Zehnder interferometer is

    .. math::
        S_{(c)} = \begin{bmatrix}
        e^{i \phi_{ext} } (e^{i \phi_{int} } - 1)   & i (e^{i \phi_{int} } + 1) & & \\
        i e^{i \phi_{ext} } (e^{i \phi_{int} } + 1) & 1 - e^{i \phi_{int} }     & & \\
        & & e^{-i \phi_{ext} } (e^{-i \phi_{int} } - 1) & -i (e^{-i \phi_{int} } + 1) \\
        & & -i e^{-i \phi_{ext} } (e^{-i \phi_{int} } + 1) & 1 - e^{-i \phi_{int} }
        \end{bmatrix},

    where :math:`\phi_{int}, \phi_{ext} \in \mathbb{R}`.
    """

    def __init__(self, int_: float, ext: float) -> None:
        """
        Args:
            int_ (float): The internal angle.
            ext (float): The external angle.
        """
        int_phase, ext_phase = np.exp(1j * np.array([int_, ext]))

        super().__init__(
            params=dict(int_=int_, ext=ext),
            passive_block=1
            / 2
            * np.array(
                [
                    [ext_phase * (int_phase - 1), 1j * (int_phase + 1)],
                    [1j * ext_phase * (int_phase + 1), 1 - int_phase],
                ]
            ),
        )


class Fourier(_ScalableBogoliubovTransformation):
    r"""Applies a Fourier gate. It simply transforms the quadratures as follows:

    .. math::
        -\hat{p} = \operatorname{\textit{F}} \hat{x} \operatorname{\textit{F}}^\dagger
            \\ \hat{x} = \operatorname{\textit{F}} \hat{p}
                \operatorname{\textit{F}}^\dagger

    The unitary operator corresponding to the Fourier gate on the :math:`i`-th mode is

    .. math::
        F_{i} = \exp \left (
            i \frac{\pi}{2} a_i^\dagger a_i
        \right ).

    The symplectic representation of the Fourier gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            i & 0  \\
            0 & -i \\
        \end{bmatrix}.

    Note:
        Corresponds to the :class:`Phaseshifter` gate with :math:`\phi = \pi/2`.
    """

    def __init__(self) -> None:
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
    """

    def __init__(self, passive: np.ndarray, active: np.ndarray) -> None:
        """
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
        if not is_symplectic(
            np.block([[passive, active], [active.conj(), passive.conj()]]),
            form_func=complex_symplectic_form,
        ):
            raise InvalidParameter(
                "The input parameters for instruction 'GaussianTransform' do not form "
                "a symplectic matrix."
            )

        super().__init__(
            params=dict(passive=passive, active=active),
            passive_block=passive,
            active_block=active,
        )


class Squeezing(_ScalableBogoliubovTransformation):
    r"""Applies the squeezing gate.

    The unitary operator corresponding to the squeezing gate is

    .. math::
        S_{i} (z) = \exp \left (
            \frac{1}{2}(z^* a_i^2 - z a_i^{\dagger 2} )
        \right ).

    The symplectic representation of the squeezing gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            \cosh r & - e^{i \phi} \sinh r \\
            - e^{- i \phi} \sinh r & \cosh r
        \end{bmatrix}.

    The unitary squeezing operator is

    .. math::
        S(z) = \exp \left (
            \frac{1}{2}(z^* a_i^2 - z a_i^{\dagger 2})
        \right ),

    where :math:`z = re^{i\phi}`.
    """

    def __init__(self, r: float, phi: float = 0.0) -> None:
        """
        Args:
            r (float): The amplitude of the squeezing instruction.
            phi (float): The squeezing angle.
        """
        super().__init__(
            params=dict(r=r, phi=phi),
            passive_block=np.diag(np.atleast_1d(np.cosh(r))),
            active_block=np.diag(
                -np.atleast_1d(np.sinh(r)) * np.exp(1j * np.atleast_1d(phi))
            ),
        )


class QuadraticPhase(_ScalableBogoliubovTransformation):
    r"""Applies the quadratic phase instruction to the state.

    The unitary operator corresponding to the Fourier gate is

    .. math::
        QP_{i} (s) = \exp \left (
            i \frac{s}{2 \hbar} x_i^2
        \right ).

    The symplectic representation of the quadratic phase gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            1 + i \frac{s}{2} & i \frac{s}{2} \\
            -i \frac{s}{2} & 1 - i \frac{s}{2}
        \end{bmatrix}.

    """

    def __init__(self, s: float) -> None:
        super().__init__(
            params=dict(s=s),
            passive_block=np.diag(1 + np.atleast_1d(s) / 2 * 1j),
            active_block=np.diag(np.atleast_1d(s) / 2 * 1j),
        )


class Squeezing2(_BogoliubovTransformation):
    r"""Applies the two-mode squeezing gate to the state.

    The unitary operator corresponding to the two-mode squeezing gate is

    .. math::
        S_{ij} (z) = \exp \left (
            \frac{1}{2}(z^* a_i a_j - z a_i^\dagger a_j^\dagger )
        \right ).

    The symplectic representation of the two-mode squeezing gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            \cosh r & 0 & 0 & e^{i \phi} \sinh r \\
            0 & \cosh r & e^{i \phi} \sinh r & 0 \\
            0 & e^{- i \phi} \sinh r & \cosh r & 0 \\
            e^{- i \phi} \sinh r & 0 & 0 & \cosh r
        \end{bmatrix}.
    """

    def __init__(self, r: float, phi: float = 0.0) -> None:
        """
        Args:
            r (float): The amplitude of the squeezing instruction.
            phi (float): The squeezing angle.
        """
        super().__init__(
            params=dict(r=r, phi=phi),
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

    The unitary operator corresponding to the controlled X gate is

    .. math::
        CX_{ij} (s) = \exp \left (
            -i \frac{s}{\hbar} x_i p_j
        \right ).

    The symplectic representation of the controlled X gate is

    .. math::
        S_{(z)} = \begin{bmatrix}
            1            & - \frac{s}{2} & 0           & \frac{s}{2} \\
            \frac{s}{2} & 1              & \frac{s}{2} & 0            \\
            0            & \frac{s}{2} & 1            & - \frac{s}{2} \\
            \frac{s}{2} & 0            & \frac{s}{2} & 1
        \end{bmatrix}.
    """

    def __init__(self, s: float):
        super().__init__(
            params=dict(s=s),
            passive_block=np.array(
                [
                    [1, -s / 2],
                    [s / 2, 1],
                ]
            ),
            active_block=np.array(
                [
                    [0, s / 2],
                    [s / 2, 0],
                ]
            ),
        )


class ControlledZ(_BogoliubovTransformation):
    r"""Applies the controlled Z gate to the state.

    The unitary operator corresponding to the controlled Z gate is

    .. math::
        CZ_{ij} (s) = \exp \left (
            -i \frac{s}{\hbar} x_i x_j
        \right ).

    The symplectic representation of the controlled Z gate is

    .. math::
        S_{(z)} = \begin{bmatrix}
            1            & i \frac{s}{2} & 0            & i \frac{s}{2}    \\
            i \frac{s}{2} & 1              & i \frac{s}{2} & 0             \\
            0            & -i \frac{s}{2} & 1            & - i \frac{s}{2} \\
            -i \frac{s}{2} & 0            & -i \frac{s}{2} & 1
        \end{bmatrix}.
    """

    def __init__(self, s: float):
        super().__init__(
            params=dict(s=s),
            passive_block=np.array(
                [
                    [1, 1j * (s / 2)],
                    [1j * (s / 2), 1],
                ]
            ),
            active_block=np.array(
                [
                    [0, 1j * (s / 2)],
                    [1j * (s / 2), 0],
                ]
            ),
        )


class Displacement(_ScalableBogoliubovTransformation):
    r"""Phase space displacement instruction.

    Evolves the ladder operators by

    .. math::
        \xi \mapsto \xi + \begin{bmatrix}
            \alpha \\
            \alpha^*
        \end{bmatrix},

    where :math:`\alpha \in \mathbb{C}^{d \times d}`.

    One must either specify :math:`\alpha` only, or the combination of :math:`r` and
    :math:`\phi`.
    When :math:`r` and :math:`\phi` are the given parameters, :math:`\alpha` is
    calculated via:

    .. math::
        \alpha = r e^{i\phi}
    """

    def __init__(
        self, *, alpha: complex = None, r: float = None, phi: float = None
    ) -> None:
        """
        Args:
            alpha (complex): The displacement.
            r (float): The displacement magnitude.
            phi (float): The displacement angle.
        """
        alpha_: np.ndarray

        if alpha is not None and r is None and phi is None:
            params = dict(alpha=alpha)
            alpha_ = np.atleast_1d(alpha)
        elif alpha is None and r is not None and phi is not None:
            params = dict(r=r, phi=phi)
            alpha_ = np.atleast_1d(r) * np.exp(1j * np.atleast_1d(phi))
        else:
            raise InvalidParameter(
                "Either specify 'alpha' only, or the combination of 'r' and 'phi': "
                f"alpha={alpha}, r={r}, phi={phi}."
            )

        super().__init__(params=params, displacement_vector=alpha_)


class PositionDisplacement(_ScalableBogoliubovTransformation):
    r"""Position displacement gate. It affects only the :math:`\hat{x}` quadrature.

    Note:
        The specified displacement is automatically scaled by :math:`\sqrt{2 \hbar}`.
    """

    def __init__(self, x: float) -> None:
        """
        Args:
            x (float): The position displacement.
        """
        super().__init__(
            params=dict(x=x),
            displacement_vector=np.atleast_1d(x),
        )


class MomentumDisplacement(_ScalableBogoliubovTransformation):
    r"""Momentum displacement gate. It only affects the :math:`\hat{p}` quadrature.

    Note:
        The specified displacement is automatically scaled by :math:`\sqrt{2 \hbar}`.
    """

    def __init__(self, p: float) -> None:
        """
        Args:
            p (float): The momentum displacement.
        """
        super().__init__(
            params=dict(p=p),
            displacement_vector=1j * np.atleast_1d(p),
        )


class Kerr(Gate):
    r"""Kerr gate.

    The definition of the Kerr gate is

    .. math::
        K_i (\xi) = \exp \left (
            i \xi n_i n_i
        \right ).

    The Kerr gate transforms the annihilation operator as

    .. math::
        K(\xi) a K(\xi) = a \exp(- i \xi (1 + 2 n)).

    Note:
        This is a non-linear gate, therefore it couldn't be used with
        :class:`~piquasso._backends.gaussian.state.GaussianState`.
    """

    def __init__(self, xi: float) -> None:
        """
        Args:
            xi (float): The magnitude of the Kerr nonlinear term.
        """
        super().__init__(params=dict(xi=xi))


class CrossKerr(Gate):
    r"""Cross-Kerr gate.

    The definition of the Cross-Kerr gate is

    .. math::
        CK_{ij} (\xi) = \exp \left (
            i \xi n_i n_j
        \right )

    The Cross-Kerr gate transforms the annihilation operators as

    .. math::
        CK_{ij} (\xi) a_i CK_{ij} (\xi) &= a_i \exp(- i \xi n_j) \\
        CK_{ij} (\xi) a_j CK_{ij} (\xi) &= a_j \exp(- i \xi n_i)


    Note:
        This is a non-linear gate, therefore it couldn't be used with
        :class:`~piquasso._backends.gaussian.state.GaussianState`.
    """

    def __init__(self, xi: float) -> None:
        """
        Args:
            xi (float): The magnitude of the Cross-Kerr nonlinear term.
        """
        super().__init__(params=dict(xi=xi))


class Graph(Gate):
    r"""Applies a graph given its adjacency matrix, see
    `this article <https://arxiv.org/pdf/1612.01199.pdf>`_ for more details.
    It decomposes the adjacency matrix of a graph into single mode squeezers and
    interferometers.
    """

    def __init__(
        self, adjacency_matrix: np.ndarray, mean_photon_number: float = 1.0
    ) -> None:
        r"""
        Args:
            adjacency_matrix (numpy.ndarray): A symmetric matrix with a size of :math:`N
                \times N` and it can be real or complex.
            mean_photon_number (float, optional): The mean photon number :math:`\bar{n}`
                for a mode. Defaults to :math:`1.0`.
        Raises:
            InvalidParameter:
                If the adjacency matrix is not invertible or not symmetric.
        """
        self.adjacency_matrix = adjacency_matrix

        if not is_invertible(adjacency_matrix):
            raise InvalidParameter("The adjacency matrix is not invertible.")

        if not is_symmetric(adjacency_matrix):
            raise InvalidParameter("The adjacency matrix should be symmetric.")

        singular_values, unitary = takagi(adjacency_matrix)

        scaling = self._get_scaling(singular_values, mean_photon_number)

        squeezing_parameters = np.arctanh(scaling * singular_values)

        # TODO: find a better solution for these.
        squeezing = GaussianTransform(
            passive=np.diag(np.cosh(squeezing_parameters)),
            active=np.diag(np.sinh(squeezing_parameters)),
        )

        interferometer = Interferometer(unitary)

        super().__init__(
            params=dict(
                adjacency_matrix=adjacency_matrix,
                mean_photon_number=mean_photon_number,
            ),
            extra_params=dict(
                squeezing=squeezing,
                interferometer=interferometer,
            ),
        )

    def _get_scaling(
        self, singular_values: np.ndarray, mean_photon_number: float
    ) -> float:
        r"""
        For a squeezed state :math:`rho` the mean photon number is calculated by

        .. math::
            \langle n \rangle_\rho = \sum_{i = 0}^d \mathrm{sinh}(r_i)^2

        where :math:`r_i = \mathrm{arctan}(s_i)`, where :math:`s_i` are the singular
        values of the adjacency matrix.
        """

        def mean_photon_number_equation(scaling: float) -> float:
            return (
                sum(
                    (scaling * singular_value) ** 2
                    / (1 - (scaling * singular_value) ** 2)
                    for singular_value in singular_values
                )
                / len(singular_values)
                - mean_photon_number
            )

        def mean_photon_number_gradient(scaling: float) -> float:
            return (2.0 / scaling) * np.sum(
                (singular_values * scaling / (1 - (singular_values * scaling) ** 2))
                ** 2
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

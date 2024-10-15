#
# Copyright 2021-2024 Budapest Quantum Computing Group
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
Gates
-----

The built-in gates in Piquasso.

Gates can be characterized by a unitary operator :math:`U`, which evolves the
quantum state in the following manner

.. math::
    \rho' = U \rho U^\dagger

Gates with at most quadratic Hamiltonians are called linear gates. Evolution of the
ladder operators by linear gates could be expressed in the form

.. math::

    U^\dagger \xi U = S_{(c)} \xi + d,

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

import abc

import numpy as np

from piquasso.api.instruction import Gate
from piquasso.api.exceptions import InvalidParameter

from piquasso._math.linalg import is_square, is_symmetric, is_invertible
from piquasso._math.symplectic import complex_symplectic_form, is_symplectic

from typing import Optional


class _PassiveLinearGate(Gate, abc.ABC):
    @abc.abstractmethod
    def _get_passive_block(self, connector, config):
        """
        The upper left (passive) block of the transformation.
        """


class _ActiveLinearGate(Gate, abc.ABC):
    @abc.abstractmethod
    def _get_passive_block(self, connector, config):
        """
        The upper left (passive) block of the transformation.
        """

    @abc.abstractmethod
    def _get_active_block(self, connector, config):
        """
        The upper right (active) block of the transformation.
        """


class Interferometer(_PassiveLinearGate):
    r"""Applies a general interferometer gate.

    The general unitary operator can be written as

    .. math::
        I = \exp \left (
            i \sum_{i, j = 1}^d H_{i j} a_i^\dagger a_j
        \right ),

    where the parameter `U` and :math:`H` is related by

    .. math::
        U = \exp \left (
            i H
        \right ).

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

        super().__init__(params=dict(matrix=matrix))

    def _get_passive_block(self, connector, config):
        return self._params["matrix"]


class Beamsplitter(_PassiveLinearGate):
    r"""Applies a beamsplitter gate.

    The general unitary operator can be written as

    .. math::
        B_{ij} (\theta, \phi) = \exp \left (
            \theta e^{i \phi} a^\dagger_i a_j
            - \theta e^{- i \phi} a^\dagger_j a_i
        \right ).

    The beamsplitter transfer matrix is given by

    .. math::
        U = \begin{bmatrix}
            t & -r^* \\
            r & t
        \end{bmatrix},

    where :math:`t = \cos(\theta)` and :math:`r = e^{i \phi} \sin(\theta)`.
    Moreover, the symplectic representation of the beamsplitter gate is

    .. math::
        S_{(c)} = U \oplos U^* = \begin{bmatrix}
            t  & -r^* &    & \\
            r & t   &    & \\
               &     & t  & -r \\
               &     & r^* & t
        \end{bmatrix}.
    """

    NUMBER_OF_MODES = 2

    def __init__(self, theta: float = np.pi / 4, phi: float = 0.0) -> None:
        r"""
        Args:
            theta (float):
                The transmissivity angle of the beamsplitter.
                Defaults to :math:`\theta=\pi/4`, which yields a 50:50 beamsplitter.
            phi (float):
                Phase angle of the beamsplitter. Defaults to :math:`\phi = 0`.
        """

        super().__init__(
            params=dict(
                theta=theta,
                phi=phi,
            ),
        )

    def _get_passive_block(self, connector, config):
        np = connector.np

        theta = self._params["theta"]
        phi = self._params["phi"]

        t = np.cos(theta)
        r = np.exp(1j * phi) * np.sin(theta)

        return np.array(
            [
                [t, -np.conj(r)],
                [r, t],
            ],
            dtype=config.complex_dtype,
        )


class Beamsplitter5050(_PassiveLinearGate):
    r"""Applies a 50:50 beamsplitter gate.

    This gate corresponds to a regular beamsplitter gate :class:`Beamsplitter` with
    parameters :math:`\theta = \frac{\pi}{4}` and :math:`\phi = 0`.

    The beamsplitter transfer matrix is given by

    .. math::
        U = \frac{1}{\sqrt{2}} \begin{bmatrix}
            1 & -1 \\
            1 & 1
        \end{bmatrix}.
    """

    NUMBER_OF_MODES = 2

    def __init__(self) -> None:
        super().__init__()

    def _get_passive_block(self, connector, config):
        np = connector.np

        return np.array(
            [
                [1, -1],
                [1, 1],
            ],
            dtype=config.dtype,
        ) / np.sqrt(2)


class Phaseshifter(_PassiveLinearGate):
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

    NUMBER_OF_MODES = 1

    def __init__(self, phi: float) -> None:
        """
        Args:
            phi (float): The angle of the rotation.
        """

        super().__init__(
            params=dict(phi=phi),
        )

    def _get_passive_block(self, connector, config):
        np = connector.np

        phi = self._params["phi"]

        return np.array([[np.exp(1j * phi)]], dtype=complex)


class MachZehnder(_PassiveLinearGate):
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

    NUMBER_OF_MODES = 2

    def __init__(self, int_: float, ext: float) -> None:
        """
        Args:
            int_ (float): The internal angle.
            ext (float): The external angle.
        """

        super().__init__(
            params=dict(int_=int_, ext=ext),
        )

    def _get_passive_block(self, connector, config):
        np = connector.np

        int_ = self._params["int_"]
        ext = self._params["ext"]
        int_phase, ext_phase = np.exp(1j * np.array([int_, ext]))

        return (
            1
            / 2
            * connector.np.array(
                [
                    [ext_phase * (int_phase - 1), 1j * (int_phase + 1)],
                    [1j * ext_phase * (int_phase + 1), 1 - int_phase],
                ],
                dtype=config.complex_dtype,
            )
        )


class Fourier(_PassiveLinearGate):
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

    NUMBER_OF_MODES = 1

    def __init__(self) -> None:
        super().__init__()

    def _get_passive_block(self, connector, config):
        return connector.np.array([[1j]], dtype=config.complex_dtype)


class GaussianTransform(_ActiveLinearGate):
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

        super().__init__(params=dict(passive=passive, active=active))

    def _get_passive_block(self, connector, config):
        return self._params["passive"]

    def _get_active_block(self, connector, config):
        return self._params["active"]


class Squeezing(_ActiveLinearGate):
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

    NUMBER_OF_MODES = 1

    def __init__(self, r: float, phi: float = 0.0) -> None:
        """
        Args:
            r (float): The amplitude of the squeezing instruction.
            phi (float): The squeezing angle.
        """
        super().__init__(params=dict(r=r, phi=phi))

    def _get_passive_block(self, connector, config):
        np = connector.np

        r = self.params["r"]

        return np.array([[np.cosh(r)]], dtype=config.complex_dtype)

    def _get_active_block(self, connector, config):
        np = connector.np

        r = self.params["r"]
        phi = self.params["phi"]

        return np.array([[-np.sinh(r) * np.exp(1j * phi)]], dtype=config.complex_dtype)


class QuadraticPhase(_ActiveLinearGate):
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

    NUMBER_OF_MODES = 1

    def __init__(self, s: float) -> None:
        super().__init__(params=dict(s=s))

    def _get_passive_block(self, connector, config):
        s = self._params["s"]

        return connector.np.array([[1 + s / 2 * 1j]], dtype=config.complex_dtype)

    def _get_active_block(self, connector, config):
        s = self._params["s"]

        return connector.np.array([[s / 2 * 1j]], dtype=config.complex_dtype)


class Squeezing2(_ActiveLinearGate):
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

    NUMBER_OF_MODES = 2

    def __init__(self, r: float, phi: float = 0.0) -> None:
        """
        Args:
            r (float): The amplitude of the squeezing instruction.
            phi (float): The squeezing angle.
        """
        super().__init__(params=dict(r=r, phi=phi))

    def _get_passive_block(self, connector, config):
        np = connector.np

        r = self._params["r"]

        return np.array(
            [
                [np.cosh(r), 0],
                [0, np.cosh(r)],
            ],
            dtype=config.complex_dtype,
        )

    def _get_active_block(self, connector, config):
        np = connector.np

        r = self._params["r"]
        phi = self._params["phi"]

        return np.array(
            [
                [0, np.sinh(r) * np.exp(1j * phi)],
                [np.sinh(r) * np.exp(1j * phi), 0],
            ],
            dtype=config.complex_dtype,
        )


class ControlledX(_ActiveLinearGate):
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

    NUMBER_OF_MODES = 2

    def __init__(self, s: float):
        super().__init__(params=dict(s=s))

    def _get_passive_block(self, connector, config):
        s = self._params["s"]

        return connector.np.array(
            [
                [1, -s / 2],
                [s / 2, 1],
            ],
            dtype=config.complex_dtype,
        )

    def _get_active_block(self, connector, config):
        s = self._params["s"]

        return connector.np.array(
            [
                [0, s / 2],
                [s / 2, 0],
            ],
            dtype=config.complex_dtype,
        )


class ControlledZ(_ActiveLinearGate):
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

    NUMBER_OF_MODES = 2

    def __init__(self, s: float):
        super().__init__(params=dict(s=s))

    def _get_passive_block(self, connector, config):
        s = self._params["s"]

        return connector.np.array(
            [
                [1, 1j * (s / 2)],
                [1j * (s / 2), 1],
            ],
            dtype=config.complex_dtype,
        )

    def _get_active_block(self, connector, config):
        s = self._params["s"]

        return connector.np.array(
            [
                [0, 1j * (s / 2)],
                [1j * (s / 2), 0],
            ],
            dtype=config.complex_dtype,
        )


class Displacement(Gate):
    r"""Phase space displacement instruction.

    Evolves the ladder operators by

    .. math::
        \xi \mapsto \xi + \begin{bmatrix}
            \alpha \\
            \alpha^*
        \end{bmatrix},

    where :math:`\alpha \in \mathbb{C}^{d}`.

    One must either specify :math:`\alpha` only, or the combination of :math:`r` and
    :math:`\phi`.
    When :math:`r` and :math:`\phi` are the given parameters, :math:`\alpha` is
    calculated via:

    .. math::
        \alpha = r e^{i\phi}

    Note:
        The specified displacement is automatically scaled by :math:`\sqrt{2 \hbar}`.
    """

    NUMBER_OF_MODES = 1

    def __init__(
        self,
        r: float,
        phi: Optional[float] = 0.0,
    ) -> None:
        """
        Args:
            r (float): The magnitude of phase space displacement.
            phi (float): The angle of phase space displacement
        """

        super().__init__(params=dict(r=r, phi=phi))


class PositionDisplacement(Gate):
    r"""Position displacement gate. It affects only the :math:`\hat{x}` quadrature.

    Note:
        The specified displacement is automatically scaled by :math:`\sqrt{2 \hbar}`.
    """

    NUMBER_OF_MODES = 1

    def __init__(self, x: float) -> None:
        """
        Args:
            x (float): The position displacement.
        """
        super().__init__(params=dict(x=x), extra_params=dict(r=x, phi=0.0))


class MomentumDisplacement(Gate):
    r"""Momentum displacement gate. It only affects the :math:`\hat{p}` quadrature.

    Note:
        The specified displacement is automatically scaled by :math:`\sqrt{2 \hbar}`.
    """

    NUMBER_OF_MODES = 1

    def __init__(self, p: float) -> None:
        """
        Args:
            p (float): The momentum displacement.
        """
        super().__init__(params=dict(p=p), extra_params=dict(r=p, phi=np.pi / 2))


class CubicPhase(Gate):
    r"""Cubic Phase gate.

    The definition of the Cubic Phase gate is

    .. math::
        \operatorname{CP}(\gamma) = e^{i \hat{x}^3 \frac{\gamma}{3 \hbar}}

    The Cubic Phase gate transforms the annihilation operator as

    .. math::
        \operatorname{CP}^\dagger(\gamma) \hat{a} \operatorname{CP}(\gamma) = \hat{a}
            + i\frac{\gamma(\hat{a} +\hat{a}^\dagger)^2}{2\sqrt{2/\hbar}}

    It transforms the :math:`\hat{p}` quadrature as follows:

    .. math::
        \operatorname{CP}^\dagger(\gamma) \hat{p} \operatorname{CP}(\gamma) =
            \hat{p} + \gamma \hat{x}^2.

    Note:
        This is a non-linear gate, therefore it couldn't be used with
        :class:`~piquasso._simulators.gaussian.state.GaussianState`.
        Using this gate requires a high cutoff to make the more accurate simulation.
    """

    NUMBER_OF_MODES = 1

    def __init__(self, gamma: float) -> None:
        """
        Args:
            gamma (float): The Cubic Phase parameter.
        """
        super().__init__(params=dict(gamma=gamma))


class Kerr(Gate):
    r"""Kerr gate.

    The definition of the Kerr gate is

    .. math::
        K_i (\xi) = \exp \left (
            i \xi n_i n_i
        \right ).

    The Kerr gate transforms the annihilation operator as

    .. math::
        K(\xi) a K(\xi)^\dagger = a \exp(- i \xi (1 + 2 n)).

    Note:
        This is a non-linear gate, therefore it couldn't be used with
        :class:`~piquasso._simulators.gaussian.state.GaussianState`.
    """

    NUMBER_OF_MODES = 1

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
        CK_{ij}^\dagger (\xi) a_i CK_{ij} (\xi) &= a_i \exp(- i \xi n_j) \\
        CK_{ij}^\dagger (\xi) a_j CK_{ij} (\xi) &= a_j \exp(- i \xi n_i)


    Note:
        This is a non-linear gate, therefore it couldn't be used with
        :class:`~piquasso._simulators.gaussian.state.GaussianState`.
    """

    NUMBER_OF_MODES = 2

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

        super().__init__(
            params=dict(
                adjacency_matrix=adjacency_matrix,
                mean_photon_number=mean_photon_number,
            ),
        )

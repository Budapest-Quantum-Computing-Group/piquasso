#
# Copyright 2021-2025 Budapest Quantum Computing Group
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
=====

The built-in quantum gates of Piquasso.

Gates can be characterized by a unitary operator :math:`U`, which describes the
time evolution of quantum states as

.. math::
    \rho' = U \rho U^\dagger,

where :math:`\rho` is a density operator.

Gates with at most quadratic Hamiltonians are called linear gates. Evolution of the
ladder operators by linear gates could be expressed in the form

.. math::
    U^\dagger \boldsymbol{\xi} U = S_{(c)} \boldsymbol{\xi} + \boldsymbol{\beta},
    :label: linearity

where :math:`S_{(c)} \in \operatorname{Sp}(2d, \mathbb{R})` is a symplectic matrix in
the complex form,
:math:`\boldsymbol{\beta} = [\boldsymbol{\alpha}, \overline{\boldsymbol{\alpha}}]`,
:math:`\alpha \in \mathbb{C}^{d}`, and

.. math::
    \boldsymbol{\xi} := \begin{bmatrix}
        a_1, \dots, a_d, a_1^\dagger, \dots, a_d^\dagger
    \end{bmatrix}^T,
    :label: xi

where :math:`a_1^\dagger, ..., a_d^\dagger` and :math:`a_1, \dots, a_d` are the
creation and annihilation operators, respectively.

Most of the gates defined here are linear gates, which can be characterized by
:math:`S_{(c)}` and :math:`\boldsymbol{\beta}`.
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

    The unitary operator corresponding to an interferometer can be written as

    .. math::
        I(U) = \exp(i \hat{H}),

    where :math:`\hat{H}` is a quadratic Hamiltonian

    .. math::
        \hat{H} = \boldsymbol{\xi}^\dagger H \boldsymbol{\xi}, \qquad H = \frac{1}{2}
        \begin{bmatrix}
            A & \\
            & \overline{A}
        \end{bmatrix},

    where :math:`A` is self-adjoint and :math:`\boldsymbol{\xi}` is defined in Eq.
    :eq:`xi`. The interferometer is parametrized as :math:`U = e^{iA}`.

    The interpretation of the interferometer matrix :math:`U` is that it is the
    one-particle unitary matrix, i.e., the matrix which acts on single particles.
    Moreover, the evolution of the ladder operators can be described by

    .. math::
        S_{(c)} = \begin{bmatrix}
            U & 0_{d \times d} \\
            0_{d \times d} & \overline{U}
        \end{bmatrix},

    where :math:`U \in \mathbb{C}^{d \times d}` is the interferometer matrix, and
    :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.

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

    The unitary operator of the beamsplitter gate can be written as

    .. math::
        B_{ij} (\theta, \phi) = \exp \left (
            \theta e^{i \phi} a^\dagger_i a_j
            - \theta e^{- i \phi} a^\dagger_j a_i
        \right ).

    The beamsplitter transfer matrix is given by

    .. math::
        U = \begin{bmatrix}
            t & -\overline{r} \\
            r & t
        \end{bmatrix},

    where :math:`t = \cos(\theta)` and :math:`r = e^{i \phi} \sin(\theta)`.
    Moreover, the symplectic representation of the beamsplitter gate is

    .. math::
        S_{(c)} = U \oplus \overline{U} = \begin{bmatrix}
            t  & -\overline{r} &    & \\
            r & t   &    & \\
               &     & t  & -r \\
               &     & \overline{r} & t
        \end{bmatrix},

    where :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.
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
        \end{bmatrix},

    where :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.
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

    where :math:`\phi_{int}, \phi_{ext} \in \mathbb{R}` and :math:`S_{(c)}` is defined
    by Eq. :eq:`linearity`.
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
    r"""Applies a Fourier gate.

    The Fourier gate simply transforms the quadratures as follows:

    .. math::
        F^\dagger \hat{x} F &= -\hat{p} \\
        F^\dagger \hat{p} F &= \hat{x}

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
        \end{bmatrix},

    where :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.


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
            \overline{A} & \overline{P}
        \end{bmatrix},

    where :math:`P, A \in C^{d \times d}` fulfilling the relations

    .. math::
        P P^\dagger - A A^\dagger &= I_{d \times d}, \\
        P A^T &= A P^T,

    and :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.
    """

    def __init__(self, passive: np.ndarray, active: np.ndarray) -> None:
        """
        Args:
            passive (numpy.ndarray):
                The passive submatrix :math:`P` of the symplectic matrix corresponding
                to the transformation.
            active (numpy.ndarray):
                The active submatrix :math:`A`
                of the symplectic matrix corresponding to the transformation.

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
    r"""Applies the single-mode squeezing gate.

    The unitary operator corresponding to the squeezing gate is

    .. math::
        S_{i} (z) = \exp \left (
            \frac{1}{2}(\overline{z} a_i^2 - z a_i^{\dagger 2} )
        \right ).

    The symplectic representation of the squeezing gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            \cosh r & - e^{i \phi} \sinh r \\
            - e^{- i \phi} \sinh r & \cosh r
        \end{bmatrix},

    where :math:`S_{(c)}` is defined by Eq. :eq:`linearity`. The unitary operator
    corresponding to the squeezing gate is

    .. math::
        S(z) = \exp \left (
            \frac{1}{2}(\overline{z} a_i^2 - z a_i^{\dagger 2})
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
        \end{bmatrix},

    where :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.
    """

    NUMBER_OF_MODES = 1

    def __init__(self, s: float) -> None:
        """
        Args:
            s (float): The parameter quadratic phase gate.
        """
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
            \frac{1}{2}(\overline{z} a_i a_j - z a_i^\dagger a_j^\dagger )
        \right ).

    In the bosonic setting, symplectic representation of the two-mode squeezing gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            \cosh r & 0 & 0 & e^{i \phi} \sinh r \\
            0 & \cosh r & e^{i \phi} \sinh r & 0 \\
            0 & e^{- i \phi} \sinh r & \cosh r & 0 \\
            e^{- i \phi} \sinh r & 0 & 0 & \cosh r
        \end{bmatrix},

    where :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.

    In the fermionic setting, an analoguous definition is used: the bosonic ladder
    operators are just exchanged to fermionic ones. For two fermionic modes, the action
    on :math:`\ket{00}` and :math:`\ket{11}` is simply given as

    .. math::
        S_{12} (z) \ket{00} &= \cos (r/2) \ket{00} - e^{i \phi} \sin (r/2) \ket{11}, \\
        S_{12} (z) \ket{11} &= \cos (r/2) \ket{11} + e^{-i \phi} \sin (r/2) \ket{00},

    where :math:`z = r e^{i \phi}`.
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
    r"""Applies the controlled-X gate to the state.

    The unitary operator corresponding to the controlled-X gate is

    .. math::
        CX_{ij} (s) = \exp \left (
            -i \frac{s}{\hbar} x_i p_j
        \right ).

    The symplectic representation of the controlled-X gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            1            & - \frac{s}{2} & 0           & \frac{s}{2} \\
            \frac{s}{2} & 1              & \frac{s}{2} & 0            \\
            0            & \frac{s}{2} & 1            & - \frac{s}{2} \\
            \frac{s}{2} & 0            & \frac{s}{2} & 1
        \end{bmatrix},

    where :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.
    """

    NUMBER_OF_MODES = 2

    def __init__(self, s: float):
        """
        s (float): The parameter of the controlled-X gate.
        """
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
    r"""Applies the controlled-Z gate to the state.

    The unitary operator corresponding to the controlled-Z gate is

    .. math::
        CZ_{ij} (s) = \exp \left (
            -i \frac{s}{\hbar} x_i x_j
        \right ).

    The symplectic representation of the controlled Z gate is

    .. math::
        S_{(c)} = \begin{bmatrix}
            1            & i \frac{s}{2} & 0            & i \frac{s}{2}    \\
            i \frac{s}{2} & 1              & i \frac{s}{2} & 0             \\
            0            & -i \frac{s}{2} & 1            & - i \frac{s}{2} \\
            -i \frac{s}{2} & 0            & -i \frac{s}{2} & 1
        \end{bmatrix},

    where :math:`S_{(c)}` is defined by Eq. :eq:`linearity`.
    """

    NUMBER_OF_MODES = 2

    def __init__(self, s: float):
        """
        s (float): The parameter of the controlled-Z gate.
        """
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
    r"""Displacement gate.

    The displacement gate evolves the ladder operators by

    .. math::
        \boldsymbol{\xi} \mapsto \boldsymbol{\xi} + \begin{bmatrix}
            \alpha \\
            \overline{\alpha}
        \end{bmatrix},

    where :math:`\alpha \in \mathbb{C}` and :math:`\boldsymbol{\xi} = [a, a^\dagger]^T`
    is a formal vector of the ladder operators.

    One must either specify :math:`\alpha` only, or the combination of :math:`r` and
    :math:`\phi`.
    When :math:`r` and :math:`\phi` are the given parameters, :math:`\alpha` is
    calculated via:

    .. math::
        \alpha = r e^{i\phi}.

    For more intuition, one can consider transforming the quadrature operators
    :math:`\hat{x}_j` and :math:`\hat{p}_j`:

    .. math::
        D^\dagger_j(\alpha)
        \begin{bmatrix}
            \hat{x}_j \\
            \hat{p}_j
        \end{bmatrix}
        D_j(\alpha)
        = \begin{bmatrix}
            \hat{x}_j + \sqrt{2 \hbar} \mathrm{Re} \alpha I \\
            \hat{p}_j + \sqrt{2 \hbar} \mathrm{Im} \alpha I
        \end{bmatrix}.

    This means, that the parameter :math:`\alpha \in \mathbb{C}` describes the shift of
    the quadrature operators in the two-dimensional phase space, i.e., the real part is
    responsible for a shift in position, and the imaginary part for a shift in momentum.

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

    Since the Kerr gate is non-linear, the ladder operators are not evolved into a
    linear combination of ladder operators, which can be attributed to the fact that the
    commutator :math:`[a_j, \hat{n}^2_j]` is not a linear combination of the ladder
    operators, i.e., :math:`[a_j, \hat{n}^2_j] = a_j (2\hat{n}_j-I)`. Hence,
    the Kerr gate transforms the ladder operators as

    .. math::
        K^\dagger_j(\kappa) a_j K_j(\kappa) &=
            a_j \exp\left(i \kappa \left( 2 \hat{n}_j - I\right)\right),\\
        K^\dagger_j(\kappa) a_j^\dagger K_j(\kappa) &=
            a_j^\dagger \exp\left(
                -i \kappa \left(2 \hat{n}_j - I \right)
            \right).

    Note:
        This is a non-linear gate, therefore it cannot be used with
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
        \right ).

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

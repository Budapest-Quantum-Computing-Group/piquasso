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

"""
Fermionic instructions
----------------------

Extra instructions specific for Fermionic quantum computing.
"""

from piquasso.api.instruction import Preparation, Gate


class ParentHamiltonian(Preparation):
    r"""Prepares the fermionic Gaussian state with a specified parent Hamiltonian.

    The density matrix is of the form

    .. math::

        \rho = \frac{e^{\hat{H}}}{\operatorname{Tr} e^{\hat{H}}},

    where :math:`\hat{H}` is the parent Hamiltonian and has the form

    .. math::

        \hat{H} &= \mathbf{f}^\dagger H \mathbf{f}, \\\\
        H &= \begin{bmatrix}
            - \overline{A} & B \\
            - \overline{B} & A
        \end{bmatrix}. \\\\

    Here, :math:`A` is self-adjoint and :math:`B` is skew-symmetric.
    """

    def __init__(self, hamiltonian):
        super().__init__(params=dict(hamiltonian=hamiltonian))


class GaussianHamiltonian(Gate):
    r"""Applies a fermionic Gaussian transformation using its quadratic Hamiltonian.

    The gate unitary is of the form

    .. math::

        U = e^{i \hat{H}},

    where

    .. math::
        \hat{H} &= \mathbf{f} H \mathbf{f}^\dagger, \\\\
        H &= \begin{bmatrix}
            A & -\overline{B} \\
            B & -\overline{A}
        \end{bmatrix}.

    Here, :math:`A` is self-adjoint and :math:`B` is skew-symmetric.
    """

    def __init__(self, hamiltonian):
        super().__init__(params=dict(hamiltonian=hamiltonian))


class ControlledPhase(Gate):
    r"""Controlled-phase gate.

    The controlled-phase gate is defined via the unitary operator

    .. math::
        \text{CPhase}_{ij} (\phi) = \exp \left ( i \phi n_i n_j \right ).

    Note:
        This is a non-linear gate, therefore it couldn't be used with
        :class:`~piquasso.fermionic.gaussian.simulator.GaussianSimulator`.

    Note:
        This is analoguous to the :class:`~piquasso.instructions.gates.CrossKerr` gate
        in the photonic setting.
    """

    NUMBER_OF_MODES = 2

    def __init__(self, phi: float) -> None:
        """
        Args:
            phi (float): The controlled-phase angle.
        """
        super().__init__(params=dict(phi=phi))


class IsingXX(Gate):
    r"""Ising XX coupling gate.

    The Ising XX gate is defined via the unitary operator

    .. math::
        \text{XX}_{ij} (\phi) = \exp \left ( i \phi X \otimes X \right )
            = \cos \phi I + i \sin \phi (X \otimes X).

    Considering a two-mode system with Majorana operators :math:`m_1, m_2, m_3, m_4`,
    :math:`X \otimes X` can also be written as

    .. math::
        X \otimes X = -i m_2 m_3.
    """

    NUMBER_OF_MODES = 2

    def __init__(self, phi: float) -> None:
        """
        Args:
            phi (float): The rotation angle.
        """
        super().__init__(params=dict(phi=phi))

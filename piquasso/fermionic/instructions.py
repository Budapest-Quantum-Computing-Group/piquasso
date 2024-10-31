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

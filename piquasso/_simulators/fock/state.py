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

from typing import DefaultDict, Optional, Tuple, Generator, Any, Dict, List

import abc
from collections import defaultdict
import numpy as np
from piquasso.api.config import Config
from piquasso.api.state import State
from piquasso.api.connector import BaseConnector

from piquasso._math.fock import get_fock_space_basis
from piquasso.api.exceptions import InvalidModes
from piquasso._simulators.plot import plot_wigner_function


class BaseFockState(State, abc.ABC):
    def __init__(
        self, *, d: int, connector: BaseConnector, config: Optional[Config] = None
    ) -> None:
        super().__init__(connector=connector, config=config)
        self._d = d
        cutoff = self._config.cutoff
        self._space = get_fock_space_basis(d=d, cutoff=cutoff)  # type: ignore

    @property
    def d(self) -> int:
        return self._d

    @property
    def norm(self) -> float:
        return self._connector.np.sum(self.fock_probabilities)

    def _as_code(self) -> str:
        return f"pq.Q() | pq.{self.__class__.__name__}(d={self.d})"

    @abc.abstractmethod
    def _get_empty(self) -> np.ndarray:
        """Returns a full-zero representation (i.e. norm 0)."""

    @abc.abstractmethod
    def _as_mixed(self):
        """Returns the density matrix of the Fock state."""

    @property
    @abc.abstractmethod
    def nonzero_elements(self) -> Generator[Tuple[complex, tuple], Any, None]:
        """The nonzero contributions to the state representation in Fock basis."""

    @property
    @abc.abstractmethod
    def density_matrix(self) -> np.ndarray:
        """The density matrix of the state in the truncated Fock space."""

    @abc.abstractmethod
    def reduced(self, modes: Tuple[int, ...]) -> "BaseFockState":
        """Reduces the state to a subsystem corresponding to the specified modes."""

    @abc.abstractmethod
    def normalize(self) -> None:
        """Normalizes the state to have norm 1."""

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets the Fock state to a vacuum state.
        """

    @property
    def fock_amplitudes_map(self) -> DefaultDict[tuple, complex]:
        """Probability amplitudes of Fock basis states representing the state."""
        amplitude_map = defaultdict(complex)

        for elem in self.nonzero_elements:
            amplitude_map[elem[1]] = elem[0]

        return amplitude_map

    @property
    @abc.abstractmethod
    def fock_probabilities_map(self) -> Dict[Tuple[int, ...], float]:
        """The particle number detection probabilities in a map."""

    @abc.abstractmethod
    def quadratures_mean_variance(
        self, modes: Tuple[int, ...], phi: float = 0
    ) -> Tuple[float, float]:
        """Calculates the mean and the variance of the quadratures of a Fock State"""

    def wigner_function(
        self,
        positions: List[float],
        momentums: List[float],
        modes: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        r"""
        This method calculates the Wigner function values at the specified position and
        momentum vectors, according to the following equation:

        .. math::
            W(r) = \frac{1}{\pi^d \sqrt{\mathrm{det} \sigma}}
                \exp \big (
                    - (r - \mu)^T
                    \sigma^{-1}
                    (r - \mu)
                \big ).

        Note:
            The implementation is copied from
            `QuTiP <https://qutip.org/docs/latest/modules/qutip/wigner.html#wigner>`_.

        Note:
            Only single modes are supported.

        Args:
            positions (list[float]): List of position vectors.
            momentums (list[float]): List of momentum vectors.
            modes (tuple[int], optional):
                Modes where Wigner function should be calculcated.

        Returns:
            numpy.ndarray:
                The Wigner function values in the shape of a grid specified by the
                input.
        """

        # Since this code is copied from QuTiP, this copyright notice is inserted here:

        #    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
        #    All rights reserved.
        #
        #    Redistribution and use in source and binary forms, with or without
        #    modification, are permitted provided that the following conditions are
        #    met:
        #
        #    1. Redistributions of source code must retain the above copyright notice,
        #       this list of conditions and the following disclaimer.
        #
        #    2. Redistributions in binary form must reproduce the above copyright
        #       notice, this list of conditions and the following disclaimer in the
        #       documentation and/or other materials provided with the distribution.
        #
        #    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
        #       of its contributors may be used to endorse or promote products derived
        #       from this software without specific prior written permission.
        #
        #    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        #    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        #    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
        #    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        #    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        #    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        #    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        #    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        #    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        #    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        #    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        if self._config.validate and self.d != 1 and (modes is None or len(modes) != 1):
            raise InvalidModes(
                "The Wigner function can only be calculated for a single mode: "
                f"modes={modes}."
            )

        state = self if modes is None else self.reduced(modes)

        rho = state.density_matrix

        g = np.sqrt(2 / self._config.hbar)

        # QuTiP implementation starts from here

        M = np.prod(rho.shape[0])
        X, Y = np.meshgrid(positions, momentums)
        A = 0.5 * g * (X + 1.0j * Y)

        Wlist = np.array(
            [np.zeros(np.shape(A), dtype=self._config.complex_dtype) for k in range(M)]
        )
        Wlist[0] = np.exp(-2.0 * abs(A) ** 2) / np.pi

        W = np.real(rho[0, 0]) * np.real(Wlist[0])
        for n in range(1, M):
            Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
            W += 2 * np.real(rho[0, n] * Wlist[n])

        for m in range(1, M):
            temp = np.copy(Wlist[m])
            Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wlist[m - 1]) / np.sqrt(m)

            # Wlist[m] = Wigner function for |m><m|
            W += np.real(rho[m, m] * Wlist[m])

            for n in range(m + 1, M):
                temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
                temp = np.copy(Wlist[n])
                Wlist[n] = temp2

                # Wlist[n] = Wigner function for |m><n|
                W += 2 * np.real(rho[m, n] * Wlist[n])

        return 0.5 * W * g**2

    def plot_wigner(
        self,
        positions: List[float],
        momentums: List[float],
        mode: Optional[int] = None,
    ) -> None:
        r"""
        Plots the Wigner function in phase space for a single mode using the
        given positions and momentums.

        Args:
            positions (List[float]): List of position values (x-axis)
            momentums (List[float]): List of momentum values (p-axis)
            mode (int, optional):
                Mode where Wigner function should be calculcated.

        Note:
            Only a single mode is supported.

        Returns:
            None: This method generates a plot and does not return a value.

        """
        if self._config.validate and mode is not None and mode > self.d - 1:
            raise InvalidModes(
                f"Mode {mode} is out of range for the state with {self.d} modes."
            )

        if mode is not None:
            mode_tuple = (mode,)
        else:
            mode_tuple = None

        if self._config.validate and self.d != 1 and mode_tuple is None:
            raise InvalidModes(
                "The Wigner function can only be plotted for a single mode: "
                f"modes={mode_tuple} was specified."
            )

        fock_wigner_function_values = self.wigner_function(
            positions=positions, momentums=momentums, modes=mode_tuple
        )
        x, p = np.meshgrid(positions, momentums)
        xlist = x.tolist()
        plist = p.tolist()

        plot_wigner_function(
            fock_wigner_function_values,
            positions=xlist,
            momentums=plist,
        )

    def fidelity(self, state: "BaseFockState") -> float:
        r"""Calculates the state fidelity between two quantum states.

        The state fidelity :math:`F` between two density matrices
        :math:`\rho_1, \rho_2` is given by:

        .. math::
            \operatorname{F}(\rho_1, \rho_2) = \operatorname{Tr}(\sqrt{\sqrt{\rho_1}
                \rho_2\sqrt{\rho_1}})^2 = \operatorname{Tr}(\sqrt{\rho_1 \rho_2})^2

        Args:
            state:
                Either a :class:`~piquasso._simulators.fock.pure.state.PureFockState`
                or a :class:`~piquasso._simulators.fock.general.state.FockState` that
                can be used to calculate the fidelity aganist it.

        Returns:
            float: The calculated fidelity.
        """
        geometric_mean = np.sqrt(
            np.linalg.eigvals(self.density_matrix @ state.density_matrix)
        )
        return np.abs(np.sum(geometric_mean)) ** 2

    def get_marginal_fock_probabilities(self, modes: Tuple[int, ...]) -> np.ndarray:
        """Return the particle number probabilities on a given subsystem.

        Args:
            modes (tuple[int, ...]): The indices of the modes to keep.

        Returns:
            numpy.ndarray: The marginal particle number detection probabilities.
        """

        return self.reduced(modes).fock_probabilities

    @abc.abstractmethod
    def get_purity(self):
        """The purity of the Fock state."""

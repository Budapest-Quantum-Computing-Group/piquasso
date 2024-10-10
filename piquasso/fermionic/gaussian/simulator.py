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

from piquasso.instructions import gates, preparations

from ..instructions import GaussianHamiltonian, ParentHamiltonian

from .calculations import (
    state_vector,
    vacuum,
    parent_hamiltonian,
    gaussian_hamiltonian,
    passive_linear_gate,
)
from .state import GaussianState

from piquasso._simulators.simulator import BuiltinSimulator
from piquasso._simulators.connectors import NumpyConnector, JaxConnector


class GaussianSimulator(BuiltinSimulator):
    """
    Fermionic Gaussian simulator.

    Example usage::

        passive_hamiltonian = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])

        U = expm(1j * passive_hamiltonian)

        state_vector = [1, 0, 1]

        with pq.Program() as program:
            pq.Q() | pq.StateVector(state_vector)

            pq.Q() | pq.Interferometer(U)

        simulator = pq.fermionic.GaussianSimulator(d=3, connector=connector)

        state = simulator.execute(program).state

    Supported preparations:
        :class:`~piquasso.instructions.preparations.Vacuum`,
        :class:`~piquasso.instructions.preparations.StateVector`.

    Supported gates:
        :class:`~piquasso.instructions.gates.Interferometer`,
        :class:`~piquasso.instructions.gates.Beamsplitter`,
        :class:`~piquasso.instructions.gates.Phaseshifter`,
        :class:`~piquasso.fermionic.instructions.GaussianHamiltonian`.
    """

    _state_class = GaussianState

    _instruction_map = {
        preparations.Vacuum: vacuum,
        preparations.StateVector: state_vector,
        ParentHamiltonian: parent_hamiltonian,
        GaussianHamiltonian: gaussian_hamiltonian,
        gates.Interferometer: passive_linear_gate,
        gates.Beamsplitter: passive_linear_gate,
        gates.Phaseshifter: passive_linear_gate,
    }

    _default_connector_class = NumpyConnector

    _extra_builtin_connectors = [JaxConnector]

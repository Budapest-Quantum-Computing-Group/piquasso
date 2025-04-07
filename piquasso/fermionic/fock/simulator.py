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

from piquasso.instructions import gates, preparations
from .state import PureFockState

from .calculations import (
    passive_linear,
    state_vector,
    squeezing2,
    controlled_phase,
    ising_XX,
)

from piquasso._simulators.simulator import BuiltinSimulator
from piquasso._simulators.connectors import NumpyConnector, JaxConnector
from piquasso.fermionic.instructions import ControlledPhase, IsingXX


class PureFockSimulator(BuiltinSimulator):
    """
    Fermionic pure Fock state simulator.

    Example usage::

        U = scipy.stats.unitary_group.rvs(4)

        with pq.Program() as program:
            pq.Q() | pq.StateVector([0, 0, 1, 1]) / np.sqrt(2)
            pq.Q() | pq.StateVector([1, 1, 0, 0]) / np.sqrt(2)
            pq.Q() | pq.Interferometer(U)

        simulator = pq.fermionic.PureFockSimulator(
            d=4, config=pq.Config(cutoff=4 + 1)
        )

        state = simulator.execute(program).state

    Supported preparations:
        :class:`~piquasso.instructions.preparations.Vacuum`,
        :class:`~piquasso.instructions.preparations.StateVector`.

    Supported gates:
        :class:`~piquasso.instructions.gates.Interferometer`.
    """

    _state_class = PureFockState

    _instruction_map = {
        preparations.StateVector: state_vector,
        gates.Interferometer: passive_linear,
        gates.Phaseshifter: passive_linear,
        gates.Beamsplitter: passive_linear,
        gates.Beamsplitter5050: passive_linear,
        gates.MachZehnder: passive_linear,
        gates.Fourier: passive_linear,
        gates.Squeezing2: squeezing2,
        ControlledPhase: controlled_phase,
        IsingXX: ising_XX,
    }

    _default_connector_class = NumpyConnector

    _extra_builtin_connectors = [JaxConnector]

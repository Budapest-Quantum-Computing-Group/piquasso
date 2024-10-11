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
from .state import HeisenbergState

from ..instructions import GaussianHamiltonian, Correlations

from .calculations import gaussian_hamiltonian, correlations

from piquasso._simulators.simulator import BuiltinSimulator
from piquasso._simulators.connectors import NumpyConnector, JaxConnector


class HeisenbergSimulator(BuiltinSimulator):
    """ """

    _state_class = HeisenbergState

    _instruction_map = {
        GaussianHamiltonian: gaussian_hamiltonian,
        Correlations: correlations,
        # gates.Interferometer: passive_linear_gate,
        # gates.Beamsplitter: passive_linear_gate,
        # gates.Phaseshifter: passive_linear_gate,
    }

    _default_connector_class = NumpyConnector

    _extra_builtin_connectors = [JaxConnector]

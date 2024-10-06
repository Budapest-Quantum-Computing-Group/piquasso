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

from ..simulator import BuiltinSimulator
from piquasso.instructions import preparations, gates, measurements, channels

from piquasso._simulators.connectors import NumpyConnector, JaxConnector

from .state import SamplingState
from .calculations import (
    state_vector,
    passive_linear,
    particle_number_measurement,
    loss,
    lossy_interferometer,
    post_select_photons,
    imperfect_post_select_photons,
)


class SamplingSimulator(BuiltinSimulator):
    """Performs photonic simulations using Fock representation with pure states.

    The simulation (when executed) results in an instance of
    :class:`~piquasso._simulators.sampling.state.SamplingState`.

    Example usage::

        import numpy as np
        import piquasso as pq

        from scipy.stats import unitary_group

        d = 5

        U = unitary_group.rvs(d)

        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

            pq.Q() | pq.Interferometer(U)

            pq.Q(all) | pq.ParticleNumberMeasurement()

        simulator = pq.SamplingSimulator(d=5)
        result = simulator.execute(program, shots=100)

    Supported preparations:
        :class:`~piquasso.instructions.preparations.StateVector`.

    Supported gates:
        :class:`~piquasso.instructions.gates.Interferometer`,
        :class:`~piquasso.instructions.gates.Beamsplitter`,
        :class:`~piquasso.instructions.gates.Phaseshifter`,
        :class:`~piquasso.instructions.gates.MachZehnder`,
        :class:`~piquasso.instructions.gates.Fourier`,
        :class:`~piquasso.instructions.gates.Interferometer`.

    Supported measurements:
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement`.

    Supported channels:
        :class:`~piquasso.instructions.channels.Loss`.
    """

    _instruction_map = {
        preparations.StateVector: state_vector,
        gates.Beamsplitter: passive_linear,
        gates.Beamsplitter5050: passive_linear,
        gates.Phaseshifter: passive_linear,
        gates.MachZehnder: passive_linear,
        gates.Fourier: passive_linear,
        gates.Interferometer: passive_linear,
        measurements.ParticleNumberMeasurement: particle_number_measurement,
        measurements.PostSelectPhotons: post_select_photons,
        measurements.ImperfectPostSelectPhotons: imperfect_post_select_photons,
        channels.Loss: loss,
        channels.LossyInterferometer: lossy_interferometer,
    }

    _state_class = SamplingState

    _default_connector_class = NumpyConnector

    _extra_builtin_connectors = [JaxConnector]

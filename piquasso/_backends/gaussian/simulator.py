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

from piquasso.api.simulator import Simulator
from piquasso.instructions import preparations, gates, measurements, channels

from .state import GaussianState
from .calculations import (
    passive_linear,
    linear,
    displacement,
    graph,
    homodyne_measurement,
    generaldyne_measurement,
    vacuum,
    mean,
    covariance,
    particle_number_measurement,
    threshold_measurement,
    deterministic_gaussian_channel,
)


class GaussianSimulator(Simulator):
    """Performs photonic simulations using Gaussian representation.

    The simulation (when executed) results in an instance of
    :class:`~piquasso._backends.gaussian.state.GaussianState`.

    Example usage::

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

        simulator = pq.GaussianSimulator(d=5)
        result = simulator.execute(program)

    Supported preparations:
        :class:`~piquasso.instructions.preparations.Vacuum`,
        :class:`~piquasso.instructions.preparations.Mean`,
        :class:`~piquasso.instructions.preparations.Covariance`,
        :class:`~piquasso.instructions.preparations.Thermal`.

    Supported gates:
        :class:`~piquasso.instructions.gates.Interferometer`,
        :class:`~piquasso.instructions.gates.Beamsplitter`,
        :class:`~piquasso.instructions.gates.Phaseshifter`,
        :class:`~piquasso.instructions.gates.MachZehnder`,
        :class:`~piquasso.instructions.gates.Fourier`,
        :class:`~piquasso.instructions.gates.GaussianTransform`,
        :class:`~piquasso.instructions.gates.Squeezing`,
        :class:`~piquasso.instructions.gates.QuadraticPhase`,
        :class:`~piquasso.instructions.gates.Squeezing2`,
        :class:`~piquasso.instructions.gates.ControlledX`,
        :class:`~piquasso.instructions.gates.ControlledZ`,
        :class:`~piquasso.instructions.gates.Displacement`,
        :class:`~piquasso.instructions.gates.PositionDisplacement`,
        :class:`~piquasso.instructions.gates.MomentumDisplacement`,
        :class:`~piquasso.instructions.gates.Graph`.

    Supported measurements:
        :class:`~piquasso.instructions.measurements.HomodyneMeasurement`,
        :class:`~piquasso.instructions.measurements.HeterodyneMeasurement`,
        :class:`~piquasso.instructions.measurements.GeneraldyneMeasurement`,
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement`,
        :class:`~piquasso.instructions.measurements.ThresholdMeasurement`.

    Supported channels:
        :class:`~piquasso.instructions.channels.DeterministicGaussianChannel`,
        :class:`~piquasso.instructions.channels.Attenuator`.
    """

    _instruction_map = {
        preparations.Vacuum: vacuum,
        preparations.Mean: mean,
        preparations.Covariance: covariance,
        preparations.Thermal: covariance,
        gates.Interferometer: passive_linear,
        gates.Beamsplitter: passive_linear,
        gates.Phaseshifter: passive_linear,
        gates.MachZehnder: passive_linear,
        gates.Fourier: passive_linear,
        gates.GaussianTransform: linear,
        gates.Squeezing: linear,
        gates.QuadraticPhase: linear,
        gates.Squeezing2: linear,
        gates.ControlledX: linear,
        gates.ControlledZ: linear,
        gates.Displacement: displacement,
        gates.PositionDisplacement: displacement,
        gates.MomentumDisplacement: displacement,
        gates.Graph: graph,
        measurements.HomodyneMeasurement: homodyne_measurement,
        measurements.HeterodyneMeasurement: generaldyne_measurement,
        measurements.GeneraldyneMeasurement: generaldyne_measurement,
        measurements.ParticleNumberMeasurement: particle_number_measurement,
        measurements.ThresholdMeasurement: threshold_measurement,
        channels.DeterministicGaussianChannel: deterministic_gaussian_channel,
        channels.Attenuator: deterministic_gaussian_channel,
    }

    _state_class = GaussianState

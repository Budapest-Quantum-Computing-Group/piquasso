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
from piquasso.instructions import preparations, gates, measurements

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
)


class GaussianSimulator(Simulator):
    _instruction_map = {
        preparations.Vacuum: vacuum,
        preparations.Mean: mean,
        preparations.Covariance: covariance,
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
    }

    _state_class = GaussianState

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

from piquasso.api.computer import Simulator


class GaussianSimulator(Simulator):
    _instruction_map = {
        "Interferometer": passive_linear,
        "Beamsplitter": passive_linear,
        "Phaseshifter": passive_linear,
        "MachZehnder": passive_linear,
        "Fourier": passive_linear,
        "GaussianTransform": linear,
        "Squeezing": linear,
        "QuadraticPhase": linear,
        "Squeezing2": linear,
        "ControlledX": linear,
        "ControlledZ": linear,
        "Displacement": displacement,
        "PositionDisplacement": displacement,
        "MomentumDisplacement": displacement,
        "Graph": graph,
        "HomodyneMeasurement": homodyne_measurement,
        "HeterodyneMeasurement": generaldyne_measurement,
        "GeneraldyneMeasurement": generaldyne_measurement,
        "Vacuum": vacuum,
        "Mean": mean,
        "Covariance": covariance,
        "ParticleNumberMeasurement": particle_number_measurement,
        "ThresholdMeasurement": threshold_measurement,
    }

    state_class = GaussianState

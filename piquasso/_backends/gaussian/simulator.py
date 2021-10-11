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

from piquasso.api.computer import Simulator


class GaussianSimulator(Simulator):
    _instruction_map = {
        "Interferometer": "_passive_linear",
        "Beamsplitter": "_passive_linear",
        "Phaseshifter": "_passive_linear",
        "MachZehnder": "_passive_linear",
        "Fourier": "_passive_linear",
        "GaussianTransform": "_linear",
        "Squeezing": "_linear",
        "QuadraticPhase": "_linear",
        "Squeezing2": "_linear",
        "ControlledX": "_linear",
        "ControlledZ": "_linear",
        "Displacement": "_displacement",
        "PositionDisplacement": "_displacement",
        "MomentumDisplacement": "_displacement",
        "Graph": "_graph",
        "HomodyneMeasurement": "_homodyne_measurement",
        "HeterodyneMeasurement": "_generaldyne_measurement",
        "GeneraldyneMeasurement": "_generaldyne_measurement",
        "Vacuum": "_vacuum",
        "Mean": "_mean",
        "Covariance": "_covariance",
        "ParticleNumberMeasurement": "_particle_number_measurement",
        "ThresholdMeasurement": "_threshold_measurement",
    }

    state_class = GaussianState

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

from piquasso.api.computer import Simulator


class BaseFockSimulator(Simulator):
    _instruction_map = {
        "Interferometer": "_passive_linear",
        "Beamsplitter": "_passive_linear",
        "Phaseshifter": "_passive_linear",
        "MachZehnder": "_passive_linear",
        "Fourier": "_passive_linear",
        "Kerr": "_kerr",
        "CrossKerr": "_cross_kerr",
        "Squeezing": "_linear",
        "QuadraticPhase": "_linear",
        "Displacement": "_linear",
        "PositionDisplacement": "_linear",
        "MomentumDisplacement": "_linear",
        "Squeezing2": "_linear",
        "GaussianTransform": "_linear",
        "ParticleNumberMeasurement": "_particle_number_measurement",
        "Vacuum": "_vacuum",
        "Create": "_create",
        "Annihilate": "_annihilate",
    }

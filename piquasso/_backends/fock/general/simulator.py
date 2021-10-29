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

from .state import FockState

from .calculations import (
    passive_linear,
    linear,
    density_matrix_instruction,
    kerr,
    cross_kerr,
    particle_number_measurement,
    vacuum,
    create,
    annihilate,
)


class FockSimulator(Simulator):
    _instruction_map = {
        "DensityMatrix": density_matrix_instruction,
        "Interferometer": passive_linear,
        "Beamsplitter": passive_linear,
        "Phaseshifter": passive_linear,
        "MachZehnder": passive_linear,
        "Fourier": passive_linear,
        "Kerr": kerr,
        "CrossKerr": cross_kerr,
        "Squeezing": linear,
        "QuadraticPhase": linear,
        "Displacement": linear,
        "PositionDisplacement": linear,
        "MomentumDisplacement": linear,
        "Squeezing2": linear,
        "GaussianTransform": linear,
        "ParticleNumberMeasurement": particle_number_measurement,
        "Vacuum": vacuum,
        "Create": create,
        "Annihilate": annihilate,
    }

    state_class = FockState

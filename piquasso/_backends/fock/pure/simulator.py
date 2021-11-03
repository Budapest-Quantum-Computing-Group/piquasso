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

from .state import PureFockState

from .calculations import (
    state_vector_instruction,
    passive_linear,
    kerr,
    cross_kerr,
    linear,
    particle_number_measurement,
    vacuum,
    create,
    annihilate,
)

from piquasso.api.computer import Simulator
from piquasso.instructions import preparations, gates, measurements


class PureFockSimulator(Simulator):
    state_class = PureFockState

    _instruction_map = {
        preparations.Vacuum: vacuum,
        preparations.Create: create,
        preparations.Annihilate: annihilate,
        preparations.StateVector: state_vector_instruction,
        gates.Interferometer: passive_linear,
        gates.Beamsplitter: passive_linear,
        gates.Phaseshifter: passive_linear,
        gates.MachZehnder: passive_linear,
        gates.Fourier: passive_linear,
        gates.Kerr: kerr,
        gates.CrossKerr: cross_kerr,
        gates.Squeezing: linear,
        gates.QuadraticPhase: linear,
        gates.Displacement: linear,
        gates.PositionDisplacement: linear,
        gates.MomentumDisplacement: linear,
        gates.Squeezing2: linear,
        gates.GaussianTransform: linear,
        measurements.ParticleNumberMeasurement: particle_number_measurement,
    }

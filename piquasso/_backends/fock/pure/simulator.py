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
    squeezing,
    displacement,
    linear,
    particle_number_measurement,
    vacuum,
    create,
    annihilate,
)

from piquasso.api.simulator import Simulator
from piquasso.instructions import preparations, gates, measurements


class PureFockSimulator(Simulator):
    """Performs photonic simulations using Fock representation with pure states.

    The simulation (when executed) results in an instance of
    :class:`~piquasso._backends.fock.pure.state.PureFockState`.

    Example usage::

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

        simulator = pq.PureFockSimulator(d=5)
        result = simulator.execute(program)

    Supported preparations:
        :class:`~piquasso.instructions.preparations.Vacuum`,
        :class:`~piquasso.instructions.preparations.Create`,
        :class:`~piquasso.instructions.preparations.Annihilate`,
        :class:`~piquasso.instructions.preparations.StateVector`.

    Supported gates:
        :class:`~piquasso.instructions.gates.Interferometer`,
        :class:`~piquasso.instructions.gates.Beamsplitter`,
        :class:`~piquasso.instructions.gates.Phaseshifter`,
        :class:`~piquasso.instructions.gates.MachZehnder`,
        :class:`~piquasso.instructions.gates.Fourier`,
        :class:`~piquasso.instructions.gates.Kerr`,
        :class:`~piquasso.instructions.gates.CrossKerr`,
        :class:`~piquasso.instructions.gates.GaussianTransform`,
        :class:`~piquasso.instructions.gates.Squeezing`,
        :class:`~piquasso.instructions.gates.QuadraticPhase`,
        :class:`~piquasso.instructions.gates.Squeezing2`,
        :class:`~piquasso.instructions.gates.ControlledX`,
        :class:`~piquasso.instructions.gates.ControlledZ`,
        :class:`~piquasso.instructions.gates.Displacement`,
        :class:`~piquasso.instructions.gates.PositionDisplacement`,
        :class:`~piquasso.instructions.gates.MomentumDisplacement`.

    Supported measurements:
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement`.
    """

    _state_class = PureFockState

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
        gates.Squeezing: squeezing,
        gates.QuadraticPhase: linear,
        gates.Displacement: displacement,
        gates.PositionDisplacement: displacement,
        gates.MomentumDisplacement: displacement,
        gates.Squeezing2: linear,
        gates.GaussianTransform: linear,
        measurements.ParticleNumberMeasurement: particle_number_measurement,
    }

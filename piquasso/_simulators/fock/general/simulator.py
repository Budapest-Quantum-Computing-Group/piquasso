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

from ...simulator import BuiltinSimulator
from piquasso.instructions import preparations, gates, measurements, channels

from piquasso._simulators.connectors import NumpyConnector

from .state import FockState
from .calculations import (
    passive_linear,
    squeezing,
    displacement,
    linear,
    density_matrix_instruction,
    kerr,
    cross_kerr,
    cubic_phase,
    particle_number_measurement,
    vacuum,
    create,
    annihilate,
)

from ..calculations import attenuator


class FockSimulator(BuiltinSimulator):
    """Performs photonic simulations using Fock representation.

    The simulation (when executed) results in an instance of
    :class:`~piquasso._simulators.fock.general.state.FockState`.

    Example usage::

        import numpy as np
        import piquasso as pq


        with pq.Program() as program:
            pq.Q(all) | pq.Vacuum()

            pq.Q(0)   | pq.Squeezing(r=0.1)
            pq.Q(1)   | pq.Squeezing(r=0.2)

            pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

            pq.Q(0) | pq.Attenuator(theta=0.01)
            pq.Q(1) | pq.Attenuator(theta=0.02)

            pq.Q(0) | pq.Kerr(xi=0.05)

        simulator = pq.FockSimulator(d=2, config=pq.Config(cutoff=7))
        result = simulator.execute(program)

    Supported preparations:
        :class:`~piquasso.instructions.preparations.Vacuum`,
        :class:`~piquasso.instructions.preparations.Create`,
        :class:`~piquasso.instructions.preparations.Annihilate`,
        :class:`~piquasso.instructions.preparations.DensityMatrix`.

    Supported gates:
        :class:`~piquasso.instructions.gates.Interferometer`,
        :class:`~piquasso.instructions.gates.Beamsplitter`,
        :class:`~piquasso.instructions.gates.Phaseshifter`,
        :class:`~piquasso.instructions.gates.MachZehnder`,
        :class:`~piquasso.instructions.gates.Fourier`,
        :class:`~piquasso.instructions.gates.Kerr`,
        :class:`~piquasso.instructions.gates.CrossKerr`,
        :class:`~piquasso.instructions.gates.CubicPhase`,
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

    Supported channels:
        :class:`~piquasso.instructions.channels.Attenuator`
    """

    _instruction_map = {
        preparations.Vacuum: vacuum,
        preparations.Create: create,
        preparations.Annihilate: annihilate,
        preparations.DensityMatrix: density_matrix_instruction,
        gates.Interferometer: passive_linear,
        gates.Beamsplitter: passive_linear,
        gates.Beamsplitter5050: passive_linear,
        gates.Phaseshifter: passive_linear,
        gates.MachZehnder: passive_linear,
        gates.Fourier: passive_linear,
        gates.Kerr: kerr,
        gates.CrossKerr: cross_kerr,
        gates.CubicPhase: cubic_phase,
        gates.Squeezing: squeezing,
        gates.QuadraticPhase: linear,
        gates.Displacement: displacement,
        gates.PositionDisplacement: displacement,
        gates.MomentumDisplacement: displacement,
        gates.Squeezing2: linear,
        gates.GaussianTransform: linear,
        measurements.ParticleNumberMeasurement: particle_number_measurement,
        channels.Attenuator: attenuator,
    }

    _state_class = FockState

    _default_connector_class = NumpyConnector

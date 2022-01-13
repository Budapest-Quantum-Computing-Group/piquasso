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

"""The Piquasso module.

One can access all the instructions and states from here as attributes.
"""

from piquasso.api.mode import Q
from piquasso.api.config import Config
from piquasso.api.instruction import (
    Instruction,
    Preparation,
    Gate,
    Measurement,
)
from piquasso.api.program import Program
from piquasso.api.state import State
from piquasso.api.computer import Computer
from piquasso.api.simulator import Simulator
from piquasso.api.utils import as_code

from piquasso._backends.sampling import SamplingState, SamplingSimulator

from piquasso._backends.gaussian import GaussianState, GaussianSimulator
from piquasso._backends.fock import (
    FockState,
    PureFockState,
    FockSimulator,
    PureFockSimulator,
)

from .instructions.preparations import (
    Vacuum,
    Mean,
    Covariance,
    Thermal,
    StateVector,
    DensityMatrix,
    Create,
    Annihilate,
)

from .instructions.gates import (
    GaussianTransform,
    Phaseshifter,
    Beamsplitter,
    MachZehnder,
    Fourier,
    Displacement,
    PositionDisplacement,
    MomentumDisplacement,
    Squeezing,
    QuadraticPhase,
    Squeezing2,
    Kerr,
    CrossKerr,
    ControlledX,
    ControlledZ,
    Interferometer,
    Graph,
)

from .instructions.measurements import (
    ParticleNumberMeasurement,
    ThresholdMeasurement,
    HomodyneMeasurement,
    HeterodyneMeasurement,
    GeneraldyneMeasurement,
    Sampling,
)

from .instructions.channels import (
    DeterministicGaussianChannel,
    Attenuator,
    Loss,
)


__all__ = [
    # API
    "Program",
    "Q",
    "Config",
    "Instruction",
    "Preparation",
    "Gate",
    "Measurement",
    "State",
    "Computer",
    "Simulator",
    "as_code",
    # Simulators
    "GaussianSimulator",
    "SamplingSimulator",
    "FockSimulator",
    "PureFockSimulator",
    # States
    "GaussianState",
    "SamplingState",
    "FockState",
    "PureFockState",
    # Preparations
    "Vacuum",
    "Mean",
    "Covariance",
    "Thermal",
    "StateVector",
    "DensityMatrix",
    "Create",
    "Annihilate",
    # Gates
    "GaussianTransform",
    "Phaseshifter",
    "Beamsplitter",
    "MachZehnder",
    "Fourier",
    "Displacement",
    "PositionDisplacement",
    "MomentumDisplacement",
    "Squeezing",
    "QuadraticPhase",
    "Squeezing2",
    "Kerr",
    "CrossKerr",
    "ControlledX",
    "ControlledZ",
    "Interferometer",
    "Sampling",
    "Graph",
    # Measurements
    "ParticleNumberMeasurement",
    "ThresholdMeasurement",
    "HomodyneMeasurement",
    "HeterodyneMeasurement",
    "GeneraldyneMeasurement",
    # Channels
    "DeterministicGaussianChannel",
    "Attenuator",
    "Loss",
]

__version__ = "0.7.0"

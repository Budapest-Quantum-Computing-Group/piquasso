#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

from piquasso import cvqnn, dual_rail_encoding, fermionic
from piquasso._simulators.connectors import (
    JaxConnector,
    NumpyConnector,
    TensorflowConnector,
    TorchConnector,
)
from piquasso._simulators.fock import (
    BatchPureFockState,
    FockSimulator,
    FockState,
    PureFockSimulator,
    PureFockState,
)
from piquasso._simulators.gaussian import GaussianSimulator, GaussianState
from piquasso._simulators.sampling import SamplingSimulator, SamplingState
from piquasso.api.computer import Computer
from piquasso.api.config import Config
from piquasso.api.instruction import (
    Gate,
    Instruction,
    Measurement,
    Preparation,
)
from piquasso.api.mode import Q
from piquasso.api.program import Program
from piquasso.api.simulator import Simulator
from piquasso.api.state import State
from piquasso.api.utils import as_code

from .instructions.batch import (
    BatchApply,
    BatchPrepare,
)
from .instructions.channels import (
    Attenuator,
    DeterministicGaussianChannel,
    Loss,
    LossyInterferometer,
)
from .instructions.gates import (
    SNAP,
    Beamsplitter,
    Beamsplitter5050,
    ControlledX,
    ControlledZ,
    CrossKerr,
    CubicPhase,
    Displacement,
    Fourier,
    GaussianTransform,
    Graph,
    Interferometer,
    Kerr,
    MachZehnder,
    MomentumDisplacement,
    Phaseshifter,
    PositionDisplacement,
    QuadraticPhase,
    Squeezing,
    Squeezing2,
)
from .instructions.measurements import (
    GeneraldyneMeasurement,
    HeterodyneMeasurement,
    HomodyneMeasurement,
    ImperfectPostSelectPhotons,
    ParticleNumberMeasurement,
    PostSelectPhotons,
    ThresholdMeasurement,
)
from .instructions.preparations import (
    Annihilate,
    Covariance,
    Create,
    DensityMatrix,
    Mean,
    StateVector,
    Thermal,
    Vacuum,
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
    # Connectors
    "NumpyConnector",
    "TensorflowConnector",
    "JaxConnector",
    "TorchConnector",
    # States
    "GaussianState",
    "SamplingState",
    "FockState",
    "PureFockState",
    "BatchPureFockState",
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
    "Beamsplitter5050",
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
    "SNAP",
    "CubicPhase",
    "ControlledX",
    "ControlledZ",
    "Interferometer",
    "Graph",
    # Measurements
    "ParticleNumberMeasurement",
    "ThresholdMeasurement",
    "HomodyneMeasurement",
    "HeterodyneMeasurement",
    "GeneraldyneMeasurement",
    "PostSelectPhotons",
    "ImperfectPostSelectPhotons",
    # Channels
    "DeterministicGaussianChannel",
    "Attenuator",
    "Loss",
    "LossyInterferometer",
    # Batch
    "BatchPrepare",
    "BatchApply",
    # Modules
    "dual_rail_encoding",
    "cvqnn",
    "fermionic",
]

__version__ = "7.2.1"

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
from piquasso.api.instruction import Instruction, Preparation, Gate, Measurement
from piquasso.api.program import Program
from piquasso.api.state import State
from piquasso.api.plugin import Plugin

from piquasso._backends.sampling import SamplingState
from piquasso._backends.gaussian import GaussianState
from piquasso._backends.fock import FockState, PureFockState

from piquasso.core import registry

from .instructions.preparations import (
    Vacuum,
    Mean,
    Covariance,
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
    Loss,
)


_default_preparations = {
    "Vacuum": Vacuum,
    "Mean": Mean,
    "Covariance": Covariance,
    "StateVector": StateVector,
    "DensityMatrix": DensityMatrix,
    "Create": Create,
    "Annihilate": Annihilate,
}


_default_gates = {
    "GaussianTransform": GaussianTransform,
    "Phaseshifter": Phaseshifter,
    "Beamsplitter": Beamsplitter,
    "MachZehnder": MachZehnder,
    "Fourier": Fourier,
    "Displacement": Displacement,
    "PositionDisplacement": PositionDisplacement,
    "MomentumDisplacement": MomentumDisplacement,
    "Squeezing": Squeezing,
    "QuadraticPhase": QuadraticPhase,
    "Squeezing2": Squeezing2,
    "Kerr": Kerr,
    "CrossKerr": CrossKerr,
    "ControlledX": ControlledX,
    "ControlledZ": ControlledZ,
    "Interferometer": Interferometer,
    "Sampling": Sampling,
    "Graph": Graph,
}


_default_measurements = {
    "ParticleNumberMeasurement": ParticleNumberMeasurement,
    "ThresholdMeasurement": ThresholdMeasurement,
    "HomodyneMeasurement": HomodyneMeasurement,
    "HeterodyneMeasurement": HeterodyneMeasurement,
    "GeneraldyneMeasurement": GeneraldyneMeasurement,
}


_default_channels = {
    "Loss": Loss,
}


class _DefaultPlugin(Plugin):
    classes = {
        "SamplingState": SamplingState,
        "GaussianState": GaussianState,
        "FockState": FockState,
        "PureFockState": PureFockState,
        **_default_preparations,
        **_default_gates,
        **_default_measurements,
        **_default_channels,
    }


registry.use_plugin(_DefaultPlugin)


__all__ = [
    # General
    "Program",
    "Plugin",
    "Q",
    "Config",
    "Instruction",
    "Preparation",
    "Gate",
    "Measurement",
    "State",
    "Circuit",
    *_default_preparations.keys(),
    *_default_gates.keys(),
    *_default_measurements.keys(),
    *_default_channels.keys(),
]

__version__ = "0.2.1"

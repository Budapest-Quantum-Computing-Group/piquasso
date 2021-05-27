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

Important:
    It is preferred to access every Piquasso object from this module directly if
    possible, especially when you're using a plugin.
"""

import sys

from piquasso.api import constants
from piquasso.api.mode import Q
from piquasso.api.state import State
from piquasso.api.plugin import Plugin
from piquasso.api.program import Program
from piquasso.api.circuit import Circuit
from piquasso.api.instruction import Instruction

from piquasso._backends.sampling import SamplingState
from piquasso._backends.gaussian import GaussianState
from piquasso._backends.fock import FockState, PureFockState, PNCFockState

from piquasso.core import _registry

from .instructions.preparations import (
    Vacuum,
    Mean,
    Covariance,
    StateVector,
    DensityMatrix,
    Create,
    Annihilate,
    OccupationNumbers,
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
    Sampling,
    Graph,
)

from .instructions.measurements import (
    ParticleNumberMeasurement,
    ThresholdMeasurement,
    HomodyneMeasurement,
    HeterodyneMeasurement,
    GeneraldyneMeasurement,
)

from .instructions.channels import (
    Loss,
)


constants.seed()


_default_preparations = {
    "Vacuum": Vacuum,
    "Mean": Mean,
    "Covariance": Covariance,
    "StateVector": StateVector,
    "DensityMatrix": DensityMatrix,
    "Create": Create,
    "Annihilate": Annihilate,
    "OccupationNumbers": OccupationNumbers,
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


def use(plugin):
    _registry.use_plugin(plugin, override=True)


class _DefaultPlugin(Plugin):
    classes = {
        "SamplingState": SamplingState,
        "GaussianState": GaussianState,
        "FockState": FockState,
        "PureFockState": PureFockState,
        "PNCFockState": PNCFockState,
        **_default_preparations,
        **_default_gates,
        **_default_measurements,
        **_default_channels,
    }


_registry.use_plugin(_DefaultPlugin)


class Piquasso:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, attribute):
        try:
            return _registry.items[attribute]
        except KeyError:
            return getattr(self._module, attribute)

    def __dir__(self):
        return dir(self._module)


Piquasso.__doc__ = sys.modules[__name__].__doc__
sys.modules[__name__] = Piquasso(sys.modules[__name__])


__all__ = [
    # General
    "Program",
    "Plugin",
    "Q",
    "Instruction",
    "State",
    "Circuit",
    *_default_preparations.keys(),
    *_default_gates.keys(),
    *_default_measurements.keys(),
    *_default_channels.keys(),
]

__version__ = "0.1.3"

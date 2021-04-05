#
# Copyright (C) 2020 by TODO - All rights reserved.
#

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

from piquasso.core.registry import _use_plugin, _retrieve_class

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
    Sampling,
    Graph,
)

from .instructions.measurements import (
    MeasureParticleNumber,
    MeasureThreshold,
    MeasureHomodyne,
    MeasureHeterodyne,
    MeasureDyne,
)


constants.seed()


def use(plugin):
    _use_plugin(plugin, override=True)


class _DefaultPlugin(Plugin):
    classes = {
        "SamplingState": SamplingState,
        "GaussianState": GaussianState,
        "FockState": FockState,
        "PureFockState": PureFockState,
        "PNCFockState": PNCFockState,
    }


_use_plugin(_DefaultPlugin)


class Piquasso:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, attribute):
        try:
            return _retrieve_class(attribute)
        except NameError:
            return getattr(self._module, attribute)


sys.modules[__name__] = Piquasso(sys.modules[__name__])


__all__ = [
    # General
    "Program",
    "Plugin",
    "Q",
    "Instruction",
    "State",
    "Circuit",
    # Preparations
    "Vacuum",
    "Mean",
    "Covariance",
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
    "MeasureParticleNumber",
    "MeasureThreshold",
    "MeasureHomodyne",
    "MeasureHeterodyne",
    "MeasureDyne",
]

__version__ = "0.1.3"

#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import sys

from piquasso.api.plugin import Plugin
from piquasso.api.program import Program
from piquasso.api.mode import Q

from piquasso._backends.sampling import SamplingState
from piquasso._backends.gaussian import GaussianState
from piquasso._backends.fock import FockState
from piquasso._backends.purefock import PureFockState

from piquasso.core.registry import _use_plugin, _retrieve_class

from .operations.preparations import (
    Number,
    DMNumber,
    Create,
    Annihilate,
)

from .operations.gates import (
    PassiveTransform,
    GaussianTransform,
    R,
    B,
    MZ,
    F,
    D,
    X,
    Z,
    S,
    P,
    S2,
    K,
    CK,
    Interferometer,
    Sampling,
)

from .operations.measurements import (
    MeasureParticleNumber,
    MeasureHomodyne,
    MeasureHeterodyne,
    MeasureDyne,
)


def use(plugin):
    _use_plugin(plugin, override=True)


class _DefaultPlugin(Plugin):
    classes = {
        "SamplingState": SamplingState,
        "GaussianState": GaussianState,
        "FockState": FockState,
        "PureFockState": PureFockState,
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
    "Program",
    "Plugin",
    "PassiveTransform",
    "GaussianTransform",
    "Q",
    "R",
    "B",
    "MZ",
    "F",
    "D",
    "X",
    "Z",
    "S",
    "P",
    "S2",
    "K",
    "CK",
    "Interferometer",
    "Sampling",
    "Number",
    "DMNumber",
    "Create",
    "Annihilate",
    "MeasureParticleNumber",
    "MeasureHomodyne",
    "MeasureHeterodyne",
    "MeasureDyne",
]

__version__ = "0.1.3"

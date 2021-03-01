#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import sys

from .sampling import SamplingState
from .gaussian import GaussianState

from .program import Program
from .mode import Q
from . import registry
from .operations import (
    PassiveTransform,
    GaussianTransform,
    R,
    B,
    D,
    S,
    P,
    S2,
    Interferometer,
    Sampling,
)


def use(plugin):
    for name, class_ in plugin.classes.items():
        registry.set_class(name, class_)


class Piquasso:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, attribute):
        try:
            return registry.ClassRecorder.records[attribute]
        except KeyError:
            return getattr(self._module, attribute)


sys.modules[__name__] = Piquasso(sys.modules[__name__])


__all__ = [
    "Program",
    "PassiveTransform",
    "GaussianTransform",
    "Q",
    "R",
    "B",
    "D",
    "S",
    "P",
    "S2",
    "Interferometer",
    "Sampling",
    "GaussianState",
    "SamplingState",
]

__version__ = "0.1.1"

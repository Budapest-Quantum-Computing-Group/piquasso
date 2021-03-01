#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import sys

from .plugin import Plugin, DefaultPlugin

from .program import Program
from .mode import Q
from .registry import use_plugin
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
    use_plugin(plugin, override=True)


use_plugin(DefaultPlugin)


class Piquasso:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, attribute):
        try:
            return registry.retrieve_class(attribute)
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
    "D",
    "S",
    "P",
    "S2",
    "Interferometer",
    "Sampling",
]

__version__ = "0.1.1"

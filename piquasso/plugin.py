#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc

from piquasso.sampling import SamplingState
from piquasso.gaussian import GaussianState
from piquasso.pncfock import PNCFockState


class Plugin(abc.ABC):
    classes: dict


class DefaultPlugin(Plugin):
    classes = {
        "SamplingState": SamplingState,
        "GaussianState": GaussianState,
        "PNCFockState": PNCFockState,
    }

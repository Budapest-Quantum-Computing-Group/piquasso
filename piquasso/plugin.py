#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc

from piquasso.sampling import SamplingState
from piquasso.gaussian import GaussianState


class Plugin(abc.ABC):
    classes: dict


class DefaultPlugin(Plugin):
    classes = {
        "SamplingState": SamplingState,
        "GaussianState": GaussianState,
    }

#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.core.registry import _register
from piquasso.api.operation import Operation


@_register
class MeasureParticleNumber(Operation):
    """Particle number measurement.

    # TODO: Measure only certain modes!
    """

    def __init__(self):
        pass

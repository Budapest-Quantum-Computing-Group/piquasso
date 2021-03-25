#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.core.registry import _register
from piquasso.api.instruction import Instruction


@_register
class MeasureParticleNumber(Instruction):
    """Particle number measurement."""

    def __init__(self, cutoff=3, shots=1):
        super().__init__(cutoff=cutoff, shots=shots)


@_register
class MeasureDyne(Instruction):
    """General-dyne measurement."""

    def __init__(self, detection_covariance, *, shots=1):
        super().__init__(detection_covariance=detection_covariance, shots=shots)


@_register
class MeasureHomodyne(MeasureDyne):
    """Homodyne measurement."""

    def __init__(self, *, z=1e-4, shots=1):
        super().__init__(
            detection_covariance=np.array(
                [
                    [z ** 2, 0],
                    [0, (1 / z) ** 2],
                ]
            ),
            shots=shots,
        )


@_register
class MeasureHeterodyne(MeasureDyne):
    """Heterodyne measurement."""

    def __init__(self, shots=1):
        super().__init__(
            detection_covariance=np.identity(2),
            shots=shots,
        )

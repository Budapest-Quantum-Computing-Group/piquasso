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

import numpy as np

from piquasso.api.instruction import Instruction


class ParticleNumberMeasurement(Instruction):
    """Particle number measurement."""

    def __init__(self, cutoff=5, shots=1):
        super().__init__(cutoff=cutoff, shots=shots)


class ThresholdMeasurement(Instruction):
    """Threshold measurement."""

    def __init__(self, shots=1):
        super().__init__(shots=shots)


class GeneraldyneMeasurement(Instruction):
    """General-dyne measurement."""

    def __init__(self, detection_covariance, *, shots=1):
        super().__init__(detection_covariance=detection_covariance, shots=shots)


class HomodyneMeasurement(Instruction):
    """Homodyne measurement."""

    def __init__(self, phi=0.0, z=1e-4, shots=1):
        super().__init__(
            phi=phi,
            detection_covariance=np.array(
                [
                    [z ** 2, 0],
                    [0, (1 / z) ** 2],
                ]
            ),
            shots=shots,
        )


class HeterodyneMeasurement(GeneraldyneMeasurement):
    """Heterodyne measurement."""

    def __init__(self, shots=1):
        super().__init__(
            detection_covariance=np.identity(2),
            shots=shots,
        )

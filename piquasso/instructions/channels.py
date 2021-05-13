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

from piquasso.core.mixins import _ScalingMixin

from piquasso.api.errors import InvalidParameter
from piquasso.api.instruction import Instruction


class Loss(Instruction, _ScalingMixin):
    """Applies a loss channel to the state.

    Note:
        Currently, this instruction can only be used along with
        `~piquasso._backends.sampling.state.SamplingState`.

    Args:
        transmissivity (numpy.ndarray): The transmissivity array.
    """

    def __init__(self, transmissivity):
        super().__init__(transmissivity=transmissivity)

        self._transmissivity = np.atleast_1d(transmissivity)

    def _autoscale(self):
        if (
            self._transmissivity is None
            or len(self.modes) == len(self._transmissivity)
        ):
            pass
        elif len(self._transmissivity) == 1:
            self._transmissivity = np.array(
                [self._transmissivity[0]] * len(self.modes),
                dtype=complex,
            )
        else:
            raise InvalidParameter(
                f"The channel {self} is not applicable to modes {self.modes} with the "
                "specified parameters."
            )

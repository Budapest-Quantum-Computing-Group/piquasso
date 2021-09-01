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

from piquasso.core import _mixins

from piquasso.api.errors import InvalidParameter
from piquasso.api.instruction import Instruction


class Loss(Instruction, _mixins.ScalingMixin):
    """Applies a loss channel to the state.

    Note:
        Currently, this instruction can only be used along with
        :class:`~piquasso._backends.sampling.state.SamplingState`.

    Args:
        transmissivity (numpy.ndarray): The transmissivity array.
    """

    def __init__(self, transmissivity: np.ndarray) -> None:
        super().__init__(
            params=dict(transmissivity=transmissivity),
            extra_params=dict(
                transmissivity=np.atleast_1d(transmissivity),
            ),
        )

    def _autoscale(self) -> None:
        transmissivity = self._extra_params["transmissivity"]
        if (
            transmissivity is None
            or len(self.modes) == len(transmissivity)
        ):
            pass
        elif len(transmissivity) == 1:
            self._extra_params["transmissivity"] = np.array(
                [transmissivity[0]] * len(self.modes),
                dtype=complex,
            )
        else:
            raise InvalidParameter(
                f"The channel {self} is not applicable to modes {self.modes} with the "
                "specified parameters."
            )

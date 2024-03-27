#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

from typing import Tuple, TYPE_CHECKING

from piquasso.api.instruction import Gate

if TYPE_CHECKING:
    import numpy as np


class PostSelectPhotons(Gate):
    """Post-selection on detected photon numbers."""

    def __init__(
        self, postselect_modes: Tuple[int, ...], photon_counts: Tuple[int, ...]
    ):
        """
        Args:
            postselect_modes (Tuple[int, ...]): The modes to post-select on.
            photon_counts (Tuple[int, ...]):
                The desired photon numbers on the specified modes.
        """

        super().__init__(
            params=dict(postselect_modes=postselect_modes, photon_counts=photon_counts)
        )


class ImperfectPostSelectPhotons(Gate):
    def __init__(
        self,
        postselect_modes: Tuple[int, ...],
        photon_counts: Tuple[int, ...],
        detector_efficiency_matrix: "np.ndarray",
    ):
        super().__init__(
            params=dict(
                postselect_modes=postselect_modes,
                photon_counts=photon_counts,
                detector_efficiency_matrix=detector_efficiency_matrix,
            )
        )

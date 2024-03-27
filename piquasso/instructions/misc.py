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

from typing import Tuple

from piquasso.api.instruction import Gate


class PostSelectPhotons(Gate):
    """Post-selection on detected photon numbers.

    Example usage::

        with pq.Program() as program:
            pq.Q(all) | pq.StateVector([0, 1, 0]) * np.sqrt(0.2)
            pq.Q(all) | pq.StateVector([1, 1, 0]) * np.sqrt(0.3)
            pq.Q(all) | pq.StateVector([2, 1, 0]) * np.sqrt(0.5)

            pq.Q(0) | pq.Phaseshifter(np.pi)

            pq.Q(1, 2) | pq.Beamsplitter(np.pi / 8)
            pq.Q(0, 1) | pq.Beamsplitter(1.1437)
            pq.Q(1, 2) | pq.Beamsplitter(-np.pi / 8)

            pq.Q(all) | pq.PostSelectPhotons(
                postselect_modes=(1, 2), photon_counts=(1, 0)
            )

        simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=4))

        state = simulator.execute(program).state


    Note:
        The resulting state is not normalized. To normalize it, use
        :meth:`~piquasso._backends.fock.pure.state.PureFockState.normalize`.
    """

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

#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

import random

from typing import List, Tuple, Union, Optional, cast

import numpy as np

from .branch import Branch
from .config import Config
from .state import State
from .exceptions import NotImplementedCalculation


class Result:
    r"""Collects the results of a quantum computation.

    The `Result` class contains the results of a quantum computation. If no measurements
    were performed, then it contains the final quantum state, accessible through the
    property :meth:`state`. If measurements were performed, then it contains the
    measurement outcomes (samples), accessible through the property :meth:`samples`.
    Also, the post-measurement quantum states are stored in the :meth:`branches`
    property, containing a list of :class:`~piquasso.api.branch.Branch` instances.

    If the program was executed with a finite number of shots, then the samples can be
    accessed through the :meth:`samples` property. If the program was executed with
    `shots=None`, then the exact probability distribution was calculated, and the
    :meth:`samples` property is not available. In this case, the post-measurement states
    and their probabilities can be accessed through the :meth:`branches` property.
    """

    def __init__(self, branches: List[Branch], config: Config, shots: Union[int, None]):
        """
        Args:
            branches: The branches containing all the simulation results.
            config: The config object.
            shots: The number of times the circuit was executed. If it is `None`, the
                exact probability distribution was calculated, instead of samples.
        """

        self._branches = branches
        self._config = config
        self._shots = shots

    def __repr__(self) -> str:
        return f"Result(branches={self.branches}, config={self._config}, shots={self._shots})"  # noqa: E501

    @property
    def outcome_map(self) -> dict:
        """Returns a mapping from measurement outcomes to their frequencies and states.

        Returns:
            dict: A dictionary where the keys are the measurement outcomes, and the
                values are dictionaries containing the frequency and the
                post-measurement state corresponding to the outcome.
        """
        return {
            branch.outcome: {"frequency": branch.frequency, "state": branch.state}
            for branch in self.branches
        }

    @property
    def branches(self) -> List[Branch]:
        """Returns the list of branches obtained in the calculation.

        The :class:`~piquasso.api.branch.Branch` instances contain the post-measurement
        quantum states, the corresponding measurement outcomes (samples), and their
        frequencies, i.e., how many times the outcome appeared divided by the total
        number of shots. If the program was executed with `shots=None`, then the
        frequency corresponds to the exact probability of the outcome.

        Returns:
            List[Branch]: The list of branches containing the post-measurement states,
            samples, and their frequencies.
        """

        return self._branches

    @property
    def state(self) -> Optional[State]:
        """The resulting quantum state."""
        if len(self.branches) == 1:
            return self.branches[0].state

        raise NotImplementedCalculation(
            "This feature is not yet implemented. However, you can access the "
            "post-measurement state in the `Result.branches` attribute."
        )

    @property
    def samples(self) -> List[Tuple[Union[int, float], ...]]:
        """The list of measurement outcomes.

        A measurement outcome (or sample) is a tuple of integers or floating point
        numbers, depending on the type of measurements used. A sample is sorted by the
        ordering of the measurements in the executed program.

        Note, that a sample can contain both integers and floats at the same type, e.g.,
        both :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement`
        and :class:`~piquasso.instructions.measurements.HomodyneMeasurement` is used.
        """

        if self._shots is None:
            raise NotImplementedCalculation(
                "The 'Result.samples' property is not available with the exact "
                "probability distribution (i.e., when 'shots=None' is used in the "
                "simulation). Please execute the program with a finite number of shots "
                "to obtain samples."
            )

        shots = cast(int, self._shots)

        _samples = []

        for branch in self.branches:
            _samples.extend([tuple(branch.outcome)] * int(branch.frequency * shots))

        # NOTE: the samples need to be shuffled with the same seed to ensure
        # consistency.
        r = random.Random(self._config.seed_sequence)
        r.shuffle(_samples)

        return _samples

    def get_counts(self) -> dict:
        """Returns the samples binned according to their frequency.

        Raises:
            NotImplementedError: If the samples contain non-integer data.

        Returns:
            dict:
                The binned samples in a dictionary format, where the keys are the
                samples and the values are the number of occurrences.
        """

        if (
            isinstance(self.samples, np.ndarray)
            and self.samples.dtype not in (np.int32, np.int64)
        ) or (self.samples and not isinstance(self.samples[0][0], (int, np.integer))):
            raise NotImplementedError(
                "The 'Result.get_counts' method only supports samples that contain "
                "integers (e.g., samples from 'ParticleNumberMeasurement')."
            )

        shots = cast(int, self._shots)

        ret = {}
        for branch in self.branches:
            ret[branch.outcome] = int(branch.frequency * shots)

        return ret

    def to_subgraph_nodes(self) -> List[List[int]]:
        """Convert samples to subgraph modes.

        Assuming that a graph's adjacency matrix is embedded into the circuit, the
        samples from particle number measurement naturally favor dense subgraphs. The
        samples could be converted to subgraph modes using this method.

        The resulting subgraph node indices correspond to the indices in the adjacency
        matrix.

        Only meaningful for discrete samples generated by
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement`.

        References:
            - `Using Gaussian Boson Sampling to Find Dense Subgraphs <https://arxiv.org/abs/1803.10731>`_
            - `Quantum approximate optimization with Gaussian boson sampling <https://arxiv.org/abs/1803.10730>`_

        Returns:
            list[list[int]]: The list of subgraph node indices.
        """  # noqa: E501

        subgraphs = []

        for sample in self.samples:
            modes = []
            for index, count in enumerate(sample):
                modes += [index] * int(count)

            subgraphs.append(sorted(modes))

        return subgraphs

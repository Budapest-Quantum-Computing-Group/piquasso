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

import numpy as np

from fractions import Fraction


def get_counts(samples):
    counts_dct = {}
    for sample in samples:
        if sample not in counts_dct:
            counts_dct[sample] = 1
        else:
            counts_dct[sample] += 1

    return counts_dct


def sample_from_probability_map(probability_map, shots):
    """Generate samples from a probability map.

    The samples are returned in a dict, where the keys are the possible outcomes, and
    the values are the fraction of their occurrence divided by the total number of
    samples taken.

    If shots is None, return the full probability map as a frequency map, filtered to
    non-zero probabilities.

    Args:
        probability_map (Dict[Tuple[int], float]): Mapping from samples to their
            probabilities.
        shots (Optional[int]): Number of samples to generate. If None, return the full
            probability map as a frequency map, filtered to non-zero probabilities.

    Returns:
        Dict[Tuple[int], Fraction]: Mapping from samples to their frequencies.
    """
    if shots is None:
        return {
            sample: probability
            for sample, probability in probability_map.items()
            if not np.isclose(probability, 0.0)
        }

    samples = random.choices(
        population=list(probability_map.keys()),
        weights=list(probability_map.values()),
        k=shots,
    )

    binned_samples = get_counts(samples)

    return {
        sample: Fraction(multiplicity, shots)
        for sample, multiplicity in binned_samples.items()
    }

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

import random
from typing import Sequence

from piquasso.api.errors import PiquassoException


def choose_from_cumulated_probabilities(cumulated_probabilities: Sequence) -> int:
    """
    Choses an element from the given cumulatad probability distribution.

    Args:
        cumulated_probabilities:
            Monotone increasing sequance of floats.
    """
    guess = random.uniform(0, cumulated_probabilities[-1])

    for first, second in zip(cumulated_probabilities, cumulated_probabilities[1:]):
        if first < guess <= second:
            return cumulated_probabilities.index(first)

    raise PiquassoException(
        f"The cumulatad probabilities {cumulated_probabilities} are "
        "not monotone increasing."
    )

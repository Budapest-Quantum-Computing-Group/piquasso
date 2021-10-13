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

import piquasso as pq


def test_Config_seed_generates_same_output():
    seed_sequence = 42

    mean = np.array([1, 2])
    covariance = np.array(
        [
            [2, -1],
            [-1, 2],
        ]
    )

    config1 = pq.Config(seed_sequence=seed_sequence)
    config2 = pq.Config(seed_sequence=seed_sequence)

    sample = config1.rng.multivariate_normal(mean=mean, cov=covariance)
    reproduced_sample = config2.rng.multivariate_normal(mean=mean, cov=covariance)

    assert np.allclose(sample, reproduced_sample)

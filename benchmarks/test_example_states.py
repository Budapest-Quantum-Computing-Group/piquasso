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


def test_example_programs_result_in_the_same_state(
    example_pq_gaussian_state,
    example_sf_gaussian_state,
):
    # NOTE: While in SF they use the xp-ordered mean and covariance by default,
    # we access it by the `xp_` prefixes.
    assert np.allclose(
        example_pq_gaussian_state.xp_mean,
        example_sf_gaussian_state.means(),
    )

    # NOTE: We use a different definition for the covariance in piquasso, that is the
    # reason for the scaling by 2.
    assert np.allclose(
        example_pq_gaussian_state.xp_cov / 2,
        example_sf_gaussian_state.cov(),
    )

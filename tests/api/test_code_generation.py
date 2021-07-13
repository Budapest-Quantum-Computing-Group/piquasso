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


def test_code_generation():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(0) | pq.Fourier()
        pq.Q(0, 1) | pq.Beamsplitter(0.1, 0.3)

        pq.Q(all) | pq.Fourier()

    code = program.as_code()

    assert code == (
        """\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q() | pq.GaussianState(d=3)

    pq.Q(0) | pq.Fourier()
    pq.Q(0, 1) | pq.Beamsplitter(theta=0.1, phi=0.3)
    pq.Q() | pq.Fourier()
"""
    )

    exec(code)


def test_Program_as_code_with_numpy_ndarray_parameter():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=3) | pq.Vacuum()

        pq.Q(0, 1) | pq.GeneraldyneMeasurement(
            detection_covariance=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ),
        )

    code = program.as_code()

    assert code == (
        """\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q() | pq.GaussianState(d=3)

    pq.Q() | pq.Vacuum()
    pq.Q(0, 1) | pq.GeneraldyneMeasurement(detection_covariance=np.array([[1., 0.],
       [0., 1.]]))
"""
    )

    exec(code)

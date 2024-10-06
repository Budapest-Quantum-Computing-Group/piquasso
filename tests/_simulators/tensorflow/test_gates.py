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

import piquasso as pq
import pytest
import tensorflow as tf


@pytest.mark.monkey
def test_Interferometer_numpy_array_as_parameter(generate_unitary_matrix):
    r = tf.Variable(0.01)
    d = 5
    interferometer = generate_unitary_matrix(d)

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=3), connector=pq.TensorflowConnector()
    )

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=r)

        pq.Q(all) | pq.Interferometer(interferometer)

    simulator.execute(program)

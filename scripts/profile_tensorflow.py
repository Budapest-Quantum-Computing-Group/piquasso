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
import tensorflow as tf
import time
from scipy.stats import unitary_group


alpha = 0.01
r = 0.01
xi = 0.3
d = 2

interferometer = unitary_group.rvs(d)

alpha_ = tf.Variable(alpha, dtype=tf.float32)

with pq.Program() as program:
    pq.Q(all) | pq.Vacuum()

    pq.Q(all) | pq.Displacement(r=alpha_)
    pq.Q(all) | pq.Squeezing(r)
    pq.Q(all) | pq.Interferometer(interferometer)
    pq.Q(all) | pq.Kerr(xi)

simulator_fock = pq.PureFockSimulator(
    d=d, config=pq.Config(cutoff=d), connector=pq.TensorflowConnector()
)

with tf.GradientTape() as tape:
    state = simulator_fock.execute(program).state
    fock_probabilities = state.fock_probabilities


profiler_options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
)

tf.profiler.experimental.start("logdir", options=profiler_options)

start_time = time.time()
gradient = tape.jacobian(fock_probabilities, [alpha_])
print("JACOBIAN CALCULATION TIME: ", time.time() - start_time)

tf.profiler.experimental.stop()

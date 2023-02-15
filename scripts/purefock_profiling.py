#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

from scipy.stats import unitary_group


alpha = 0.01
r = 0.01
xi = 0.3
d = 5
interferometer = unitary_group.rvs(d)

with pq.Program() as program:
    pq.Q(all) | pq.Vacuum()

    pq.Q(all) | pq.Displacement(alpha=alpha)
    pq.Q(all) | pq.Squeezing(r)
    pq.Q(all) | pq.Interferometer(interferometer)
    pq.Q(all) | pq.Kerr(xi)

simulator_fock = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=d))

options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=1, python_tracer_level=1, device_tracer_level=1
)

tf.profiler.experimental.start("logdir", options=options)

state = simulator_fock.execute(program).state
mean_photon_number = state.mean_photon_number()

tf.profiler.experimental.stop()

print("MEAN_PHOTON_NUM: ", mean_photon_number)

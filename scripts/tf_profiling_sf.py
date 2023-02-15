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

import strawberryfields as sf
import tensorflow as tf
import time
from scipy.stats import unitary_group


alpha = 0.01
r = 0.01
xi = 0.3
d = 1
# interferometer = unitary_group.rvs(d)

program = sf.Program(d)

mapping = {}

alpha_ = tf.Variable(alpha)
param = program.params("alpha")
mapping["alpha"] = alpha_

engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": d})

with program.context as q:
    for i in range(d):
        sf.ops.Dgate(param) | q[i]
        # sf.ops.Sgate(r) | q[i]

    # sf.ops.Interferometer(interferometer) | tuple(q[i] for i in range(d))

    # for i in range(d):
        # sf.ops.Kgate(xi) | q[i]

with tf.GradientTape(persistent=True) as tape:
    result = engine.run(program, args=mapping)
    state = result.state
    fock_probabilities = state.all_fock_probs()

options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
)
tf.profiler.experimental.start("logdir", options=options)
start_time = time.time()

tape.jacobian(fock_probabilities, [alpha_], experimental_use_pfor=False)

tf.profiler.experimental.stop()

print("EXEC_TIME: ", time.time() - start_time)

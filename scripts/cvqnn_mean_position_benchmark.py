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


tf.get_logger().setLevel("ERROR")


def measure_graph_size(f, *args):
    if not hasattr(f, "get_concrete_function"):
        return 0

    g = f.get_concrete_function(*args).graph

    return len(g.as_graph_def().node)


def calculate_mean_position(weights, cutoff, d, connector):
    simulator = pq.PureFockSimulator(
        d,
        pq.Config(cutoff=cutoff),
        connector=connector,
    )

    with tf.GradientTape() as tape:
        cvqnn_layers = pq.cvqnn.create_layers(weights)

        preparation = [pq.Vacuum()]

        program = pq.Program(instructions=preparation + cvqnn_layers.instructions)

        final_state = simulator.execute(program).state

        mean_position = final_state.mean_position(0)

    print("_FORWARD FINISH:", time.time() - start_time)

    mean_position_grad = tape.gradient(mean_position, weights)

    print("_BACK FINISH:", time.time() - start_time)

    return mean_position, mean_position_grad


if __name__ == "__main__":
    d = 2
    layer_count = 5
    cutoff = 10

    NUMBER_OF_ITERATIONS = 5

    weights = tf.Variable(
        pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count, d=d)
    )

    decorator = tf.function(jit_compile=True)

    connector = pq.TensorflowConnector(decorate_with=decorator)

    enhanced_calculate_mean_position = decorator(calculate_mean_position)

    print("START")
    start_time = time.time()

    enhanced_calculate_mean_position(weights, cutoff, d, connector)

    print("COMPILATION TIME:", time.time() - start_time)

    print(
        "GRAPH SIZE:",
        measure_graph_size(
            enhanced_calculate_mean_position, weights, cutoff, d, connector
        ),
    )

    sum_ = 0.0

    for i in range(NUMBER_OF_ITERATIONS):
        weights = tf.Variable(
            pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count, d=d)
        )

        start_time = time.time()
        result = enhanced_calculate_mean_position(weights, cutoff, d, connector)
        end_time = time.time()

        runtime = end_time - start_time

        print(f"{i}. runtime={runtime},\t result={result[0].numpy()}")

        sum_ += runtime

    print("AVERAGE RUNTIME:", sum_ / NUMBER_OF_ITERATIONS)

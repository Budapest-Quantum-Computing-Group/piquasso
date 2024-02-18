import piquasso as pq
import tensorflow as tf

import numpy as np


d = 2
cutoff = 5


def calculate_mean_position(weights):
    simulator = pq.PureFockSimulator(
        d,
        pq.Config(cutoff=cutoff, normalize=False),
        calculator=pq.TensorflowCalculator(),
    )

    with tf.GradientTape() as tape:
        cvqnn_layers = pq.cvqnn.create_layers(weights)

        preparation = [pq.Vacuum()] + [
            pq.Displacement(r=0.1).on_modes(i) for i in range(d)
        ]

        program = pq.Program(instructions=preparation + cvqnn_layers.instructions)

        simulator.execute(program)

        final_state = simulator.execute(program).state

        mean_position = final_state.mean_position(0)

    print("FORWARD FINISH:", time.time() - start_time)

    mean_position_grad = tape.gradient(mean_position, weights)

    print("BACK FINISH:", time.time() - start_time)

    return mean_position, mean_position_grad


weigths = tf.Variable(
    pq.cvqnn.generate_random_cvqnn_weights(layer_count=3, d=d)
)

decorator = tf.function(jit_compile=True, reduce_retracing=True)

enhanced_calculate_mean_position = decorator(calculate_mean_position)

import time

print("START")
start_time = time.time()

enhanced_calculate_mean_position(weigths)

print("COMPILATION TIME:", time.time() - start_time)

weigths = tf.Variable(
    pq.cvqnn.generate_random_cvqnn_weights(layer_count=3, d=d)
)

start_time = time.time()

enhanced_calculate_mean_position(weigths)

print("SECOND RUNTIME:", time.time() - start_time)


# graph = enhanced_calculate_mean_position.get_concrete_function(weigths).graph
# for node in graph.as_graph_def().node:
#    print(f'{node.input} -> {node.name}')

import piquasso as pq
import tensorflow as tf

import numpy as np


def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph

    return len(g.as_graph_def().node)


def calculate_mean_position(input_, weights, cutoff, d):
    simulator = pq.PureFockSimulator(
        d,
        pq.Config(cutoff=cutoff, normalize=False),
        calculator=pq.TensorflowCalculator(),
    )

    with tf.GradientTape() as tape:
        cvqnn_layers = pq.cvqnn.create_layers(weights)

        preparation = [pq.Vacuum()] + [
            pq.Displacement(r=input_).on_modes(i) for i in range(d)
        ]

        program = pq.Program(instructions=preparation + cvqnn_layers.instructions)

        simulator.execute(program)

        final_state = simulator.execute(program).state

        mean_position = final_state.mean_position(0)

    print("FORWARD FINISH:", time.time() - start_time)

    mean_position_grad = tape.gradient(mean_position, weights)

    print("BACK FINISH:", time.time() - start_time)

    return mean_position, mean_position_grad


if __name__ == "__main__":
    d = 2
    cutoff = 4

    weigths = tf.Variable(pq.cvqnn.generate_random_cvqnn_weights(layer_count=3, d=d))

    decorator = tf.function(jit_compile=True, reduce_retracing=True)

    enhanced_calculate_mean_position = decorator(calculate_mean_position)

    input_ = tf.Variable(0.1)

    import time

    print("START")
    start_time = time.time()

    enhanced_calculate_mean_position(input_, weigths, cutoff, d)

    print("COMPILATION TIME:", time.time() - start_time)

    weigths = tf.Variable(
        pq.cvqnn.generate_random_cvqnn_weights(layer_count=3, d=d)
    )
    input_ = tf.Variable(0.2)

    start_time = time.time()

    enhanced_calculate_mean_position(input_, weigths, cutoff, d)

    print("SECOND RUNTIME:", time.time() - start_time)

    print("NUMBER OF NODES:", measure_graph_size(enhanced_calculate_mean_position, input_, weigths, cutoff, d))

    # graph = enhanced_calculate_mean_position.get_concrete_function(weigths).graph
    # for node in graph.as_graph_def().node:
    #    print(f'{node.input} -> {node.name}')

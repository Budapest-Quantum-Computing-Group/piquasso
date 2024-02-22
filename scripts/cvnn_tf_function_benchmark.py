import piquasso as pq
import tensorflow as tf

import time


tf.get_logger().setLevel('ERROR')


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

    print("_FORWARD FINISH:", time.time() - start_time)

    mean_position_grad = tape.gradient(mean_position, weights)

    print("_BACK FINISH:", time.time() - start_time)

    return mean_position, mean_position_grad


if __name__ == "__main__":
    d = 2
    cutoff = 3
    layer_count = 1

    NUMBER_OF_ITERATIONS = 10

    weights = tf.Variable(
        pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count, d=d)
    )

    decorator = tf.function

    enhanced_calculate_mean_position = decorator(calculate_mean_position)
    input_ = tf.Variable(0.1)

    start_time = time.time()

    calculate_mean_position(input_, weights, cutoff, d)

    print("VANILLA TIME:", time.time() - start_time)
    start_time = time.time()

    print("START")
    start_time = time.time()

    enhanced_calculate_mean_position(input_, weights, cutoff, d)

    print("COMPILATION TIME:", time.time() - start_time)
    start_time = time.time()

    sum_ = 0.0

    for _ in range(NUMBER_OF_ITERATIONS):
        input_ = tf.Variable(0.1)

        weigths = tf.Variable(
           pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count, d=d)
        )

        start_time = time.time()
        enhanced_calculate_mean_position(input_, weights, cutoff, d)
        end_time = time.time()

        sum_ += end_time - start_time

    print("AVG RUNTIME:", sum_ / NUMBER_OF_ITERATIONS)
    start_time = time.time()

    print(
        "NUMBER OF NODES:",
        measure_graph_size(
            enhanced_calculate_mean_position, input_, weights, cutoff, d
        ),
    )


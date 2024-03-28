from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

from piquasso.decompositions.clements import Clements

import piquasso as pq

import tensorflow as tf
import numpy as np
import tyro
import pandas as pd
import matplotlib.pyplot as plt
import json

from tensorflow.python.keras.optimizers import Optimizer

from typing import Any, Dict, Callable


np.set_printoptions(suppress=True, linewidth=200)


@dataclass
class Args:
    output_dir: str

    learning_rate: int = 0.00025

    starting_prob: float = 1 / 16

    iterations: int = 5000

    enhanced: bool = False
    use_jit: bool = False


tf.get_logger().setLevel("ERROR")


def _calculate_loss(
    weights: tf.Tensor,
    expected_state_vector: tf.Tensor,
    P: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    state_vector: tf.Tensor,
    cutoff: int,
    prob: tf.Tensor
):
    d = 6
    config = pq.Config(cutoff=cutoff, normalize=False)
    np = calculator.np

    modes = (0, 1, 2, 3)
    ancilla_modes = (4, 5)
    acting_modes = (0,)

    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    ancilla_state = [1, 1]

    preparation = pq.Program(instructions=[
        pq.StateVector(state_00 + ancilla_state, coefficient=state_vector[0]),
        pq.StateVector(state_01 + ancilla_state, coefficient=state_vector[1]),
        pq.StateVector(state_10 + ancilla_state, coefficient=state_vector[2]),
        pq.StateVector(state_11 + ancilla_state, coefficient=state_vector[3]),
    ])

    inverse_params = weights[:12]
    diags = weights[12:18]
    direct_params = weights[18:]

    inverse_ops = pq.Program(
        instructions=[
            pq.Phaseshifter(phi=inverse_params[1]).on_modes(1),
            pq.Beamsplitter(theta=inverse_params[0], phi=0.0).on_modes(1, 2),

            pq.Phaseshifter(phi=inverse_params[3]).on_modes(0),
            pq.Beamsplitter(theta=inverse_params[2], phi=0.0).on_modes(0, 1),

            pq.Phaseshifter(phi=inverse_params[5]).on_modes(3),
            pq.Beamsplitter(theta=inverse_params[4], phi=0.0).on_modes(3, 4),

            pq.Phaseshifter(phi=inverse_params[7]).on_modes(2),
            pq.Beamsplitter(theta=inverse_params[6], phi=0.0).on_modes(2, 3),

            pq.Phaseshifter(phi=inverse_params[9]).on_modes(1),
            pq.Beamsplitter(theta=inverse_params[8], phi=0.0).on_modes(1, 2),

            pq.Phaseshifter(phi=inverse_params[11]).on_modes(0),
            pq.Beamsplitter(theta=inverse_params[10], phi=0.0).on_modes(0, 1),
        ]
    )

    diag_ops = pq.Program(
        instructions=[
            pq.Phaseshifter(phi=diags[0]).on_modes(0),
            pq.Phaseshifter(phi=diags[1]).on_modes(1),
            pq.Phaseshifter(phi=diags[2]).on_modes(2),
            pq.Phaseshifter(phi=diags[3]).on_modes(3),
            pq.Phaseshifter(phi=diags[4]).on_modes(4),
            pq.Phaseshifter(phi=diags[5]).on_modes(5),

        ]
    )

    direct_ops = pq.Program(
        instructions=[
            pq.Beamsplitter(theta=direct_params[0], phi=np.pi).on_modes(4, 5),
            pq.Phaseshifter(phi=-direct_params[1]).on_modes(4),

            pq.Beamsplitter(theta=direct_params[2], phi=np.pi).on_modes(3, 4),
            pq.Phaseshifter(phi=-direct_params[3]).on_modes(3),

            pq.Beamsplitter(theta=direct_params[4], phi=np.pi).on_modes(2, 3),
            pq.Phaseshifter(phi=-direct_params[5]).on_modes(2),

            pq.Beamsplitter(theta=direct_params[6], phi=np.pi).on_modes(1, 2),
            pq.Phaseshifter(phi=-direct_params[7]).on_modes(1),

            pq.Beamsplitter(theta=direct_params[8], phi=np.pi).on_modes(0, 1),
            pq.Phaseshifter(phi=-direct_params[9]).on_modes(0),

            pq.Beamsplitter(theta=direct_params[10], phi=np.pi).on_modes(4, 5),
            pq.Phaseshifter(phi=-direct_params[11]).on_modes(4),

            pq.Beamsplitter(theta=direct_params[12], phi=np.pi).on_modes(3, 4),
            pq.Phaseshifter(phi=-direct_params[13]).on_modes(3),

            pq.Beamsplitter(theta=direct_params[14], phi=np.pi).on_modes(2, 3),
            pq.Phaseshifter(phi=-direct_params[15]).on_modes(2),

            pq.Beamsplitter(theta=direct_params[16], phi=np.pi).on_modes(4, 5),
            pq.Phaseshifter(phi=-direct_params[17]).on_modes(4),
        ]
    )

    program = pq.Program(instructions=[
        *preparation.instructions,

        *inverse_ops.instructions,
        *diag_ops.instructions,
        *direct_ops.instructions,

        pq.ImperfectPostSelectPhotons(
            postselect_modes=ancilla_modes,
            photon_counts=ancilla_state,
            detector_efficiency_matrix=np.eye(cutoff)
        )
    ])

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    state = simulator.execute(program).state
    reduced_state = state.reduced(modes)

    density_matrix = reduced_state.density_matrix
    success_prob = tf.math.real(tf.linalg.trace(density_matrix))
    normalized_density_matrix = density_matrix / success_prob

    F = tf.math.real(np.conj(expected_state_vector) @ normalized_density_matrix @ expected_state_vector)
    loss = 1 - F + np.exp(-1000 * (success_prob - prob))

    return loss, success_prob, F

def train_step(
    weights: tf.Tensor,
    P: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    expected_state_vector: tf.Tensor,
    state_vector: tf.Tensor,
    cutoff: int,
    prob: tf.Tensor
):
    with tf.GradientTape() as tape:
        loss, success_prob, fidelity = _calculate_loss(
            weights=weights,
            P=P,
            calculator=calculator,
            expected_state_vector=expected_state_vector,
            state_vector=state_vector,
            cutoff=cutoff,
            prob=prob
        )

    grad = tape.gradient(loss, weights)

    return loss, success_prob, fidelity, grad


def train(
    iterations: int,
    opt: Optimizer,
    _train_step: Callable,
    weights: tf.Tensor,
    P: tf.Tensor,
    state_vector: tf.Tensor,
    expected_state_vector: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    cutoff: int,
    prob: tf.Tensor,
    lossfile: Path
):

    with open(lossfile, "w") as f:
        f.write("iteration,loss,success_prob,fidelity\n")

    for i in tqdm(range(iterations)):
        loss, success_prob, fidelity, grad = _train_step(
            weights=weights,
            P=P,
            calculator=calculator,
            expected_state_vector=expected_state_vector,
            state_vector=state_vector,
            cutoff=cutoff,
            prob=prob
        )

        opt.apply_gradients(zip([grad], [weights]))

        with open(lossfile, "a+") as f:
            f.write(f"{i},{loss},{success_prob},{fidelity}\n")


def get_expected_state_vector(
    state_vector: np.ndarray,
    cutoff: int,
    calculator: pq.TensorflowCalculator
) -> np.ndarray:
    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    config = pq.Config(normalize=False, cutoff=cutoff)
    expected_program = pq.Program(instructions=[
        pq.StateVector(state_00, coefficient=state_vector[0]),
        pq.StateVector(state_01, coefficient=state_vector[1]),
        pq.StateVector(state_10, coefficient=state_vector[2]),
        pq.StateVector(state_11, coefficient=-state_vector[3]),
    ])

    simulator2 = pq.PureFockSimulator(d=4, config=config, calculator=calculator)
    expected_state = simulator2.execute(expected_program).state
    return expected_state.state_vector


def get_ideal_weights():
    U = np.array([
        [-1 / 3, -np.sqrt(2) / 3, np.sqrt(2) / 3, 2 / 3],
        [np.sqrt(2) / 3, -1 / 3, -2 / 3, np.sqrt(2) / 3],
        [-np.sqrt(3 + np.sqrt(6)) / 3, np.sqrt(3 - np.sqrt(6)) / 3, -np.sqrt((3 + np.sqrt(6))/2) / 3, np.sqrt(1/6-1/(3*np.sqrt(6)))],
        [-np.sqrt(3-np.sqrt(6)) / 3, -np.sqrt(3+np.sqrt(6)) / 3, -np.sqrt(1/6-1/(3*np.sqrt(6))), -np.sqrt((3 + np.sqrt(6))/2) / 3]
    ])

    I = np.eye(6)

    I[[0, 2, 4, 5], 0] = U[0, :]
    I[[0, 2, 4, 5], 2] = U[1, :]
    I[[0, 2, 4, 5], 4] = U[2, :]
    I[[0, 2, 4, 5], 5] = U[3, :]

    decomp = Clements(I)

    # inverse:
    #   (1, 2), (0, 1), (3, 4), (2, 3), (1, 2), (0, 1)
    # direct:
    #   (4, 5), (3, 4), (2, 3), (1, 2), (0, 1), (4, 5), (3, 4), (2, 3), (4, 5)
    return np.array(
        [param for op in decomp.inverse_operations for param in op['params']]+
        [op for op in np.angle(decomp.diagonals)] +
        [param for op in reversed(decomp.direct_operations) for param in op['params']]
    )


def main():
    args = tyro.cli(Args)

    decorator = tf.function(jit_compile=args.use_jit) if args.enhanced else None

    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    calculator = pq.TensorflowCalculator(decorate_with=decorator)
    np = calculator.np
    fallback_np = calculator.fallback_np

    cutoff = 5

    P = fallback_np.array([
        [1.0, 0.1050, 0.0110, 0.0012, 0.001, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.8950, 0.2452, 0.0513, 0.0097, 0.0017, 0.0003, 0.0001, 0.0],
        [0.0, 0.0, 0.7438, 0.3770, 0.1304, 0.0384, 0.0104, 0.0027, 0.0007],
        [0.0, 0.0, 0.0, 0.5706, 0.4585, 0.2361, 0.0996, 0.0375, 0.0132],
        [0.0, 0.0, 0.0, 0.0, 0.4013, 0.4672, 0.3346, 0.1907, 0.0952],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2565, 0.4076, 0.3870, 0.2862],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1476, 0.3066, 0.3724],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0755, 0.1985],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9338]
    ])

    P = P[:cutoff, :cutoff]
    P = tf.convert_to_tensor(P)
    ideal_weights = fallback_np.array([
        np.pi, 0.0, np.pi, 0.0, 0.0,
        np.pi / 8, 65.5302 * 2 * np.pi / 360,
        - np.pi / 8, 0,
        0, 0
    ])
    ideal_weights_tf = tf.Variable(ideal_weights, dtype=tf.float64)

    ideal_weights = get_ideal_weights()
    # ideal_weights = ideal_weights_tf.numpy()

    weights_np = ideal_weights.copy()
    weights = tf.Variable(weights_np, dtype=tf.float64)

    checkpoint = tf.train.Checkpoint(weights=weights)
    # checkpoint.restore(tf.train.latest_checkpoint(str(args.cz_input_dir)))

    state_vector = np.sqrt([1 / 4, 1 / 4, 1 / 4, 1 / 4])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _train_step = decorator(train_step) if decorator is not None else train_step

    expected_state_vector = get_expected_state_vector(
        state_vector=state_vector,
        cutoff=cutoff,
        calculator=calculator
    )
    expected_state_vector = tf.convert_to_tensor(expected_state_vector)

    train(
        iterations=args.iterations,
        opt=opt,
        _train_step=_train_step,
        weights=weights,
        P=P,
        expected_state_vector=expected_state_vector,
        state_vector=state_vector,
        calculator=calculator,
        cutoff=cutoff,
        prob=tf.convert_to_tensor(args.starting_prob),
        lossfile=output_dir / "lossfile.csv"
    )

    with open(str(output_dir / "args.json"), "w") as f:
        json.dump(args.toJson(), f)

    for plot_info in plot_infos:
        plot(**plot_info)

    checkpoint.save(str(output_dir / "weights"))


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

import piquasso as pq

import tensorflow as tf
import tyro
import pandas as pd
import matplotlib.pyplot as plt
import json

from tensorflow.python.keras.optimizers import Optimizer

from typing import Any, Dict, Callable


@dataclass
class Args:
    output_dir: str

    learning_rate: float = 0.00025

    starting_prob: float = 0.185
    ending_prob: float = 0.315

    step_size: float = 0.005

    iterations: int = 5000

    continued: bool = False
    use_softplus: bool = False
    enhanced: bool = False
    use_jit: bool = False



tf.get_logger().setLevel("ERROR")


def get_expected_density_matrix(state_vector, d, cutoff, calculator):
    config = pq.Config(cutoff=cutoff)
    expected_program = pq.Program(instructions=[
        pq.StateVector([0, 1, 0], coefficient=state_vector[0]),
        pq.StateVector([1, 1, 0], coefficient=state_vector[1]),
        pq.StateVector([2, 1, 0], coefficient=-state_vector[2]),
    ])

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    expected_density_matrix = simulator.execute(expected_program).state.density_matrix

    return expected_density_matrix



def _calculate_loss(
    weights: tf.Tensor,
    P: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    state_vector: tf.Tensor,
    cutoff: int,
    prob: tf.Tensor
):
    d = 3
    config = pq.Config(cutoff=cutoff, normalize=False)
    np = calculator.np
    phase_shifter_phis = weights[:3]
    thetas = weights[3:6]
    phis = weights[6:]

    program = pq.Program(instructions=[
        pq.StateVector([0, 1, 0], coefficient=state_vector[0]),
        pq.StateVector([1, 1, 0], coefficient=state_vector[1]),
        pq.StateVector([2, 1, 0], coefficient=state_vector[2]),

        pq.Phaseshifter(phase_shifter_phis[0]).on_modes(0),
        pq.Phaseshifter(phase_shifter_phis[1]).on_modes(1),
        pq.Phaseshifter(phase_shifter_phis[2]).on_modes(2),

        pq.Beamsplitter(theta=thetas[0], phi=phis[0]).on_modes(1, 2),
        pq.Beamsplitter(theta=thetas[1], phi=phis[1]).on_modes(0, 1),
        pq.Beamsplitter(theta=thetas[2], phi=phis[2]).on_modes(1, 2),

        pq.ImperfectPostSelectPhotons(
            postselect_modes=(1, 2),
            photon_counts=(1, 0),
            detector_efficiency_matrix=P
        )
    ])

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    reduced_state = simulator.execute(program).state.reduced((0, ))

    density_matrix = reduced_state.density_matrix[:3, :3]
    success_prob = tf.math.real(tf.linalg.trace(density_matrix))
    normalized_density_matrix = density_matrix / success_prob

    expected_state = state_vector
    expected_state = calculator.assign(expected_state, 2, -expected_state[2])

    F = tf.math.real(np.conj(expected_state) @ normalized_density_matrix @ expected_state)
    loss = 1 - F + np.exp(-1000 * ((success_prob - prob)))

    return loss, success_prob, F


def _calculate_loss_softplus(
    weights: tf.Tensor,
    P: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    state_vector: tf.Tensor,
    expected_density_matrix: tf.Tensor,
    cutoff: int,
    prob: tf.Tensor
):
    d = 3
    config = pq.Config(cutoff=cutoff, normalize=False)
    np = calculator.np
    phase_shifter_phis = weights[:3]
    thetas = weights[3:6]
    phis = weights[6:]

    program = pq.Program(instructions=[
        pq.StateVector([0, 1, 0], coefficient=state_vector[0]),
        pq.StateVector([1, 1, 0], coefficient=state_vector[1]),
        pq.StateVector([2, 1, 0], coefficient=state_vector[2]),

        pq.Phaseshifter(phase_shifter_phis[0]).on_modes(0),
        pq.Phaseshifter(phase_shifter_phis[1]).on_modes(1),
        pq.Phaseshifter(phase_shifter_phis[2]).on_modes(2),

        pq.Beamsplitter(theta=thetas[0], phi=phis[0]).on_modes(1, 2),
        pq.Beamsplitter(theta=thetas[1], phi=phis[1]).on_modes(0, 1),
        pq.Beamsplitter(theta=thetas[2], phi=phis[2]).on_modes(1, 2),

        pq.ImperfectPostSelectPhotons(
            postselect_modes=(1, 2),
            photon_counts=(1, 0),
            detector_efficiency_matrix=P
        )
    ])

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    # reduced_state = simulator.execute(program).state.reduced((0, ))
    state = simulator.execute(program).state

    # density_matrix = reduced_state.density_matrix# [:3, :3]
    density_matrix = state.density_matrix
    success_prob = tf.math.real(tf.linalg.trace(density_matrix))
    normalized_density_matrix = density_matrix / success_prob

    # expected_state = state_vector
    # expected_state = calculator.assign(expected_state, 2, -expected_state[2])
    # expected_state = np.append(expected_state, 0)

    # F = tf.math.real(np.conj(expected_state) @ normalized_density_matrix @ expected_state)
    F = tf.math.real(tf.linalg.trace(normalized_density_matrix @ expected_density_matrix))
    loss = 1 - F + 1 / 1000 * np.log(1 + np.exp(-1000 * ((success_prob - prob))))

    return loss, success_prob, F


def train_step(
    weights: tf.Tensor,
    P: tf.Tensor,
    loss_fn: Callable,
    calculator: pq.TensorflowCalculator,
    state_vector: tf.Tensor,
    expected_density_matrix: tf.Tensor,
    cutoff: int,
    prob: tf.Tensor
):
    with tf.GradientTape() as tape:
        loss_fn, success_prob, fidelity = loss_fn(
            weights=weights,
            P=P,
            calculator=calculator,
            state_vector=state_vector,
            expected_density_matrix=expected_density_matrix,
            cutoff=cutoff,
            prob=prob
        )

    grad = tape.gradient(loss_fn, weights)

    return loss_fn, success_prob, fidelity, grad


def train(
    start: int,
    iterations: int,
    opt: Optimizer,
    train_step_: Callable,
    loss_fn: Callable,
    weights: tf.Tensor,
    P: tf.Tensor,
    state_vector: tf.Tensor,
    expected_density_matrix: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    cutoff: int,
    prob: tf.Tensor,
    lossfile: Path
):

    if start == 0:
        with open(lossfile, "w") as f:
            f.write("iteration,loss,success_prob,fidelity,tradeoff\n")

    for i in tqdm(range(start, start + iterations)):
        loss, success_prob, fidelity, grad = train_step_(
            weights=weights,
            loss_fn=loss_fn,
            P=P,
            calculator=calculator,
            state_vector=state_vector,
            expected_density_matrix=expected_density_matrix,
            cutoff=cutoff,
            prob=prob
        )

        opt.apply_gradients(zip([grad], [weights]))

        with open(lossfile, "a") as f:
            f.write(f"{i},{loss},{success_prob},{fidelity},{prob}\n")


def _get_last_iter(output_dir: Path) -> int:
    df = pd.read_csv(str(output_dir / "losses.csv"))

    return df['iteration'].iloc[-1] + 1


def main():
    args = tyro.cli(Args)

    decorator = tf.function(jit_compile=args.use_jit) if args.enhanced else None

    calculator = pq.TensorflowCalculator(decorate_with=decorator)
    np = calculator.np
    fallback_np = calculator.fallback_np

    cutoff = 4

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

    ideal_weights = fallback_np.array([np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, - np.pi / 8, 0, 0, 0])

    state_vector = np.sqrt([0.2, 0.3, 0.5])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_step_ = decorator(train_step) if decorator is not None else train_step

    prob = args.starting_prob

    loss_fn = _calculate_loss_softplus if args.use_softplus else _calculate_loss

    with open(str(Path(args.output_dir) / "args.json"), "w") as f:
        json.dump(args.__dict__, f)

    expected_density_matrix = get_expected_density_matrix(state_vector, d=3, cutoff=cutoff, calculator=calculator)

    while prob < args.ending_prob:
        output_dir = Path(args.output_dir) / f"{prob:.4f}"
        output_dir.mkdir(exist_ok=True)

        weights = tf.Variable(ideal_weights.copy(), dtype=tf.float64)
        checkpoint = tf.train.Checkpoint(weights=weights)
        start = 0

        if args.continued:
            checkpoint.restore(tf.train.latest_checkpoint(str(output_dir)))
            start = _get_last_iter(output_dir)

        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        train(
            start=start,
            iterations=args.iterations,
            loss_fn=loss_fn,
            opt=opt,
            train_step_=train_step_,
            weights=weights,
            P=P,
            state_vector=state_vector,
            expected_density_matrix=expected_density_matrix,
            calculator=calculator,
            cutoff=cutoff,
            prob=tf.convert_to_tensor(prob, dtype=tf.float64),
            lossfile=output_dir / f"losses.csv"
        )

        checkpoint.save(str(output_dir / "weights"))

        prob += args.step_size

    train(
        iterations=args.iterations,
        opt=opt,
        _train_step=_train_step,
        weights=weights,
        P=P,
        state_vector=state_vector,
        calculator=calculator,
        cutoff=cutoff,
        tradeoff_coeff=tf.constant(args.starting_tradeoff),
        lossfile=plot_infos[1]["lossfile"]
    )
    with open(str(args.output_dir / "args.json"), "w") as f:
        json.dump(args.toJson(), f)




if __name__ == "__main__":
    main()

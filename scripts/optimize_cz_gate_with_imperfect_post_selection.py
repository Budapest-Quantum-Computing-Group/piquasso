from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

import piquasso as pq

import tensorflow as tf
import numpy as np
import tyro
import pandas as pd
import matplotlib.pyplot as plt
import json

from tensorflow.python.keras.optimizers import Optimizer

from typing import Any, Dict, Callable


@dataclass
class Args:
    output_dir: Path

    ns_input_dir: Path
    # cz_input_dir: Path

    ns_input_dir: Path
    # cz_input_dir: Path

    learning_rate: int = 0.00025

    starting_tradeoff: float = 0.001
    "Starting tradeoff after the initial learning"

    iterations: int = 5000

    enhanced: bool = False
    use_jit: bool = False

    def toJson(self) -> Dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "learning_rate": self.learning_rate,
            "startnig_tradeoff": self.starting_tradeoff,
            "iterations": self.iterations,
            "enhanced": self.enhanced,
            "use_jit": self.use_jit
        }

tf.get_logger().setLevel("ERROR")


def _calculate_loss(
    weights: tf.Tensor,
    expected_state_vector: tf.Tensor,
    P: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    state_vector: tf.Tensor,
    cutoff: int,
    tradeoff_coeff: tf.Tensor
):
    d = 8
    config = pq.Config(cutoff=cutoff, normalize=False)
    np = calculator.np

    modes = (0, 1, 2, 3)
    ancilla_modes = (4, 5, 6, 7)

    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    ancilla_state = [1, 0, 1, 0]

    preparation = pq.Program(instructions=[
        pq.StateVector(state_00 + ancilla_state, coefficient=state_vector[0]),
        pq.StateVector(state_01 + ancilla_state, coefficient=state_vector[1]),
        pq.StateVector(state_10 + ancilla_state, coefficient=state_vector[2]),
        pq.StateVector(state_11 + ancilla_state, coefficient=state_vector[3]),
    ])

    phase_shifter_phis = weights[0][:3]
    thetas = weights[0][3:6]
    phis = weights[0][6:]

    ns_0 = pq.Program(
        instructions=[
            pq.Phaseshifter(phase_shifter_phis[0]).on_modes(modes[0]),
            pq.Phaseshifter(phase_shifter_phis[1]).on_modes(ancilla_modes[0]),
            pq.Phaseshifter(phase_shifter_phis[2]).on_modes(ancilla_modes[1]),

            pq.Beamsplitter(theta=thetas[0], phi=phis[0]).on_modes(ancilla_modes[0], ancilla_modes[1]),
            pq.Beamsplitter(theta=thetas[1], phi=phis[1]).on_modes(modes[0], ancilla_modes[0]),
            pq.Beamsplitter(theta=thetas[2], phi=phis[2]).on_modes(ancilla_modes[0], ancilla_modes[1]),
        ]
    )

    phase_shifter_phis = weights[1][:3]
    thetas = weights[1][3:6]
    phis = weights[1][6:]

    ns_1 = pq.Program(
        instructions=[
            pq.Phaseshifter(phase_shifter_phis[0]).on_modes(modes[2]),
            pq.Phaseshifter(phase_shifter_phis[1]).on_modes(ancilla_modes[2]),
            pq.Phaseshifter(phase_shifter_phis[2]).on_modes(ancilla_modes[3]),

            pq.Beamsplitter(theta=thetas[0], phi=phis[0]).on_modes(ancilla_modes[2], ancilla_modes[3]),
            pq.Beamsplitter(theta=thetas[1], phi=phis[1]).on_modes(modes[2], ancilla_modes[2]),
            pq.Beamsplitter(theta=thetas[2], phi=phis[2]).on_modes(ancilla_modes[2], ancilla_modes[3]),
        ]
    )

    program = pq.Program(instructions=[
        *preparation.instructions,

        pq.Beamsplitter(theta=np.pi / 4).on_modes(0, 2),

        *ns_0.instructions,
        *ns_1.instructions,

        pq.Beamsplitter(theta=-np.pi / 4).on_modes(0, 2),

        pq.ImperfectPostSelectPhotons(
            postselect_modes=ancilla_modes,
            photon_counts=ancilla_state,
            detector_efficiency_matrix=P
        )
    ])

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    state = simulator.execute(program).state
    reduced_state = state.reduced(modes)

    density_matrix = reduced_state.density_matrix
    success_prob = tf.math.real(tf.linalg.trace(density_matrix))
    normalized_density_matrix = density_matrix / success_prob

    expected_program = pq.Program(instructions=[
        pq.StateVector(state_00, coefficient=state_vector[0]),
        pq.StateVector(state_01, coefficient=state_vector[1]),
        pq.StateVector(state_10, coefficient=state_vector[2]),
        pq.StateVector(state_11, coefficient=-state_vector[3]),
    ])

    simulator2 = pq.PureFockSimulator(d=4, config=config, calculator=calculator)
    expected_state = simulator2.execute(expected_program).state
    expected_state_vector = expected_state.state_vector

    breakpoint()

    F = tf.math.real(np.conj(expected_state_vector) @ normalized_density_matrix @ expected_state_vector)
    # loss = 1 - F + tradeoff_coeff * ((1 / 16 - success_prob)**2)
    loss = 1 - F + np.exp(tradeoff_coeff * (1 / 16 - success_prob))

    return loss, success_prob, F

def train_step(
    weights: tf.Tensor,
    P: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    expected_state_vector: tf.Tensor,
    state_vector: tf.Tensor,
    cutoff: int,
    tradeoff_coeff: tf.Tensor
):
    with tf.GradientTape() as tape:
        loss, success_prob, fidelity = _calculate_loss(
            weights=weights,
            P=P,
            calculator=calculator,
            expected_state_vector=expected_state_vector,
            state_vector=state_vector,
            cutoff=cutoff,
            tradeoff_coeff=tradeoff_coeff
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
    tradeoff_coeff: tf.Tensor,
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
            tradeoff_coeff=tradeoff_coeff
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
    ideal_weights = fallback_np.array([np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, - np.pi / 8, 0, 0, 0])
    weights_np = np.array([
        ideal_weights,
        ideal_weights
    ])
    weights = tf.Variable(weights_np, dtype=tf.float64)

    checkpoint = tf.train.Checkpoint(weights=weights)

    state_vector = np.sqrt([1 / 4, 1 / 4, 1 / 4, 1 / 4])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_infos = [
        {
            "lossfile": args.output_dir / "losses_0.csv",
            "export_file": args.output_dir / "cz_0.pdf",
            "name": "CZ gate with imperfect post selection (without success prob in loss)"
        },
        # {
        #     "lossfile": args.output_dir / "losses_1.csv",
        #     "export_file": args.output_dir / "cz_1.pdf",
        #     "name": "CZ gate with imperfect post selection iteration 0 (with success prob in loss)"
        # }
    ]

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
        state_vector=state_vector,
        calculator=calculator,
        cutoff=cutoff,
        tradeoff_coeff=tf.convert_to_tensor(args.starting_tradeoff),
        lossfile=plot_infos[0]["lossfile"]
    )

    # train(
    #     iterations=args.iterations,
    #     opt=opt,
    #     _train_step=_train_step,
    #     weights=weights,
    #     P=P,
    #     expected_state_vector=expected_state_vector,
    #     state_vector=state_vector,
    #     calculator=calculator,
    #     cutoff=cutoff,
    #     tradeoff_coeff=tf.convert_to_tensor(args.starting_tradeoff, dtype=tf.float64),
    #     lossfile=plot_infos[1]["lossfile"]
    # )

    with open(str(args.output_dir / "args.json"), "w") as f:
        json.dump(args.toJson(), f)

    checkpoint.save(str(args.output_dir / "weights"))


if __name__ == "__main__":
    main()

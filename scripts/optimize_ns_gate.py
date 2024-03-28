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
    output_dir: Path

    seed: int = 123
    learning_rate: int = 0.00025

    starting_tradeoff: float = 0.001
    "Starting tradeoff after the initial learning"

    iterations: int = 5000

    enhanced: bool = False
    use_jit: bool = False

    def toJson(self) -> Dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "startnig_tradeoff": self.starting_tradeoff,
            "iterations": self.iterations,
            "enhanced": self.enhanced,
            "use_jit": self.use_jit
        }



tf.get_logger().setLevel("ERROR")


def _calculate_loss(
    weights: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    state_vector: tf.Tensor,
    tradeoff_coeff: float
):
    d = 3
    config = pq.Config(cutoff=4, normalize=False)
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

        pq.PostSelectPhotons(postselect_modes=(1, 2), photon_counts=(1, 0))
    ])

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    reduced_state = simulator.execute(program).state.reduced((0, ))

    density_matrix = reduced_state.density_matrix[:3, :3]
    success_prob = tf.math.real(tf.linalg.trace(density_matrix))
    normalized_density_matrix = density_matrix / success_prob

    expected_state = state_vector
    expected_state = calculator.assign(expected_state, 2, -expected_state[2])

    F = tf.math.real(np.conj(expected_state) @ normalized_density_matrix @ expected_state)
    loss = 1 - F + tradeoff_coeff * ((1 / 4 - success_prob)**2)

    return loss, success_prob, F



def train_step(
    weights: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    state_vector: tf.Tensor,
    tradeoff_coeff: float
):
    with tf.GradientTape() as tape:
        loss, success_prob, fidelity = _calculate_loss(
            weights=weights,
            calculator=calculator,
            state_vector=state_vector,
            tradeoff_coeff=tradeoff_coeff
        )

    grad = tape.gradient(loss, weights)

    return loss, success_prob, fidelity, grad


def train(
    iterations: int,
    opt: Optimizer,
    _train_step: Callable,
    weights: tf.Tensor,
    state_vector: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    tradeoff_coeff: float,
    lossfile: Path
):
    with open(lossfile, "w") as f:
        f.write("iteration,loss,success_prob,fidelity\n")

    for i in tqdm(range(iterations)):
        loss, success_prob, fidelity, grad = _train_step(
            weights=weights,
            calculator=calculator,
            state_vector=state_vector,
            tradeoff_coeff=tradeoff_coeff
        )

        opt.apply_gradients(zip([grad], [weights]))

        with open(lossfile, "a+") as f:
            f.write(f"{i},{loss},{success_prob},{fidelity}\n")


def plot(lossfile: Path, name: str, export_file: Path):
    df = pd.read_csv(str(lossfile))

    iterations = df['iteration']
    losses = df['loss']

    plt.figure()
    plt.title(name)

    plt.plot(iterations, losses)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.savefig(str(export_file))


def main():
    args = tyro.cli(Args)

    decorator = tf.function(jit_compile=args.use_jit) if args.enhanced else None

    calculator = pq.TensorflowCalculator(decorate_with=decorator)
    np = calculator.np
    fallback_np = calculator.fallback_np
    fallback_np.random.seed(args.seed)

    ideal_weights = fallback_np.array([np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, - np.pi / 8, 0, 0, 0])

    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    errors = fallback_np.random.normal(0, 0.1, size=9)
    # error = fallback_np.random.normal(0, 0.1)
    weights_np = ideal_weights.copy()
    weights_np += errors
    weights = tf.Variable(weights_np, dtype=tf.float64)

    checkpoint = tf.train.Checkpoint(weights=weights)

    state_vector = np.sqrt([0.2, 0.3, 0.5])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_infos = [
        {
            "lossfile": args.output_dir / "losses_0.csv",
            "export_file": args.output_dir / "ns_0.pdf",
            "name": "NS gate with imperfect post selection (without success prob in loss)"
        },
        {
            "lossfile": args.output_dir / "losses_1.csv",
            "export_file": args.output_dir / "ns_1.pdf",
            "name": "NS gate with imperfect post selection iteration 0 (with success prob in loss)"
        }
    ]

    _train_step = decorator(train_step) if decorator is not None else train_step

    train(
        iterations=args.iterations,
        opt=opt,
        _train_step=_train_step,
        weights=weights,
        state_vector=state_vector,
        calculator=calculator,
        tradeoff_coeff=0,
        lossfile=plot_infos[0]["lossfile"]
    )

    train(
        iterations=args.iterations,
        opt=opt,
        _train_step=_train_step,
        weights=weights,
        state_vector=state_vector,
        calculator=calculator,
        tradeoff_coeff=0.001,
        lossfile=plot_infos[1]["lossfile"]
    )

    with open(str(args.output_dir / "args.json"), "w") as f:
        json.dump(args.toJson(), f)

    for plot_info in plot_infos:
        plot(**plot_info)
<<<<<<< HEAD

=======
    
>>>>>>> beab4140 (Add commandline arguments to NS/CZ gate optimizations)
    checkpoint.save(str(args.output_dir / "weights"))


if __name__ == "__main__":
    main()

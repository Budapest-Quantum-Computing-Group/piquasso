from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json


from scipy import stats


@dataclass
class Args:
    input_dir: str



def get_trained_losses(input_dir: Path):
    directories = [entry for entry in input_dir.iterdir() if entry.is_dir()]
    n = len(directories)
    success_probs = np.zeros((n+1, ))
    fids = np.zeros((n+1, ))
    for i, f in enumerate(directories):
        if not f.is_dir():
            continue
        df = pd.read_csv(str(f / "losses.csv"))
        success_probs[i] = df['success_prob'].iloc[-1]
        fids[i] = df['fidelity'].iloc[-1]
        plot_losses(f, df['iteration'], df['loss'], df['success_prob'], df['fidelity'])
        print(f"{f.name}: {success_probs[i]}, {fids[i]}")

    df = pd.read_csv(str(directories[0] / "losses.csv"))
    success_probs[n] = df['success_prob'].iloc[0]
    fids[n] = df['fidelity'].iloc[0]


    return success_probs, fids


def plot_losses(dir: Path, iterations, losses, probs, fids):

    iterations = iterations[:1000]
    losses = losses[:1000]
    probs = probs[:1000]
    fids = fids[:1000]

    rc_fonts = {
    "font.family": "serif",
    "font.size": 20,
    "text.usetex": True,
    'text.latex.preamble': r"""
        \usepackage[tt=false, type1=true]{libertine}
        \usepackage[libertine]{newtxmath}
        """,
    }

    mpl.rcParams.update(rc_fonts)

    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 3)

    axis = fig.add_subplot(gs[0, 0])
    fig.tight_layout()
    axis.grid()
    axis.plot(iterations, losses)
    axis.set_xlim(left=iterations.iloc[0])
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Loss')

    axis = fig.add_subplot(gs[0, 1])
    axis.grid()
    axis.plot(iterations, probs)
    axis.set_xlim(left=iterations.iloc[0])
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Success Probability')

    axis = fig.add_subplot(gs[0, 2])
    axis.grid()
    axis.plot(iterations, fids)
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Fidelity')
    axis.set_xlim(left=iterations.iloc[0])
    plt.tight_layout()

    plt.savefig(str(dir / 'losses.pdf'))

    plt.close()
    # plt.plot(iterations[-1000:], losses[-1000:])
    # plt.plot(iterations, losses)
    # plt.xlabel('Iterations')
    # plt.ylabel('Losses')

    # plt.savefig(str(dir / 'losses.pdf'))
    # plt.clf()

    # plt.plot(iterations, probs)
    # plt.xlabel('Iterations')
    # plt.ylabel('Success Probabilites')

    # plt.savefig(str(dir / 'probs.pdf'))
    # plt.clf()

    # plt.plot(iterations, fids)
    # plt.xlabel('Iterations')
    # plt.ylabel('Fidelities')

    # plt.savefig(str(dir / 'fidelities.pdf'))
    # plt.clf()


def main():
    args = tyro.cli(Args)

    input_dir = Path(args.input_dir)

    success_probs, fids = get_trained_losses(input_dir)

    slope, intercept, r_value, p_value, std_err = stats.linregress(success_probs[:-1], fids[:-1])
    line = slope * success_probs[:-1] + intercept

    with open(str(input_dir / "lin-regress.json"), "w") as f:
        d = {
            "m": slope,
            "b": intercept
        }
        json.dump(d, f)
    # print(slope)
    # print(intercept)

    rc_fonts = {
    "font.family": "serif",
    "font.size": 16,
    "text.usetex": True,
    'text.latex.preamble': r"""
        \usepackage[tt=false, type1=true]{libertine}
        \usepackage[libertine]{newtxmath}
        """,
    }

    mpl.rcParams.update(rc_fonts)

    plt.rc('grid', linestyle="--")

    plt.clf()
    plt.scatter(success_probs[:-1], fids[:-1], label="With state learning")
    plt.scatter(success_probs[-1], fids[-1], color='red', marker='x',
                label="Without state learning")
    # plt.plot(success_probs[:-1], line)
    plt.grid(True)
    plt.xlabel('Success probability')
    plt.ylabel('Fidelity')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(input_dir / "plot.pdf")


if __name__ == "__main__":
    main()

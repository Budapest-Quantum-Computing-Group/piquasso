"""Benchmark the JAX permanent pipeline against the perm_boost C++/CUDA backend.

Three backends are compared:

* ``piquasso-jax``  -- current Piquasso JAX permanent
  (:func:`piquasso._math.jax.permanent.permanent_with_reduction`).
* ``boost-cpu``     -- perm_boost C++ backend via the JAX FFI
  (:func:`piquasso.jax_extensions.permanent.perm`, dispatched to CPU).
* ``boost-gpu``     -- the same FFI on CUDA.

Two tasks are timed at a sweep of matrix sizes ``n`` (with row/column
multiplicities fixed at ``--multiplicity``, default 3, matching the upstream
permanent-boost benchmark notebook):

1. Permanent value -- ``perm(A, rows, cols)``.
2. Real/imag-split Jacobian -- ``jax.jacobian`` of a wrapper that returns
   ``(res.real, res.imag)``. This is the non-holomorphic path used by the
   perm_boost test suite (see ``tests/perm_boost/test_grad_perm.py``) and
   is the form that downstream ML loss functions hit in practice -- a
   typical loss reaches the permanent through real-valued quantities like
   ``|perm|^2`` or expectation values, not through a holomorphic chain.

For each (backend, task, n) the function is warmed up once (covers JIT/FFI
compile + first kernel launch) and then timed ``--repeats`` times; the median
wall time is recorded. There is no runtime timeout -- backends are protected
instead by hard per-backend ``max_n`` caps tuned to the algorithm's scaling
(see ``Backend.max_n_perm`` / ``Backend.max_n_grad``). The baseline scales as
2^(n*multiplicity) and is unusable past those caps; an attempted run there
would not finish in any reasonable wall-clock and signal-based timeouts can't
interrupt XLA-compiled or CUDA C calls.

Outputs land next to this script:

* ``results_permanent.csv``, ``results_gradient.csv`` -- raw timings.
* ``permanent_benchmark.png``, ``gradient_benchmark.png`` -- log-y plots.

Run ``python benchmark_perm_jax.py --help`` for tuneables. Default settings
match the ranges in the upstream notebook (``figures.ipynb``).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import unitary_group

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from piquasso._math.jax.permanent import permanent_with_reduction  # noqa: E402
from piquasso.jax_extensions.permanent import perm as boost_perm  # noqa: E402


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Backend plumbing
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    """One row in the sweep table: a permanent implementation on a device.

    Attributes:
        label: Human-readable name used in console output, CSV headers, and
            plot legends.
        device: JAX device the inputs are placed on and the function is
            dispatched against.
        perm_fn: The permanent callable; signature ``(matrix, rows, cols)``.
        style: Matplotlib kwargs (color, marker, linestyle, ...) forwarded
            to ``ax.plot`` for this backend.
        max_n_perm: Hard cap on ``n`` for the permanent-value sweep.
        max_n_grad: Hard cap on ``n`` for the Jacobian sweep.
        jit: If True, the task builders wrap ``perm_fn`` with ``jax.jit``.
    """

    label: str
    device: jax.Device
    perm_fn: Callable[..., jnp.ndarray]
    style: dict
    # Hard cap on ``n`` for each task. SIGALRM-based timeouts cannot interrupt
    # C-level CUDA/XLA calls, so we refuse to launch past these. piquasso-jax
    # scales as O(n * (n*multiplicity) * 2^(n*multiplicity-1)); perm_boost is
    # capped by the kernel's ``MAX_IDX_MAX = 1e8`` (4^n grows past it at n=14
    # for multiplicity 3).
    max_n_perm: int
    max_n_grad: int
    # When True, the task builders wrap perm_fn (and the jacobian thereof)
    # with jax.jit. The warmup call absorbs compile time; subsequent timed
    # calls measure post-compile execution.
    jit: bool = False


def _build_backends() -> list[Backend]:
    """Probe for CPU/GPU and build the list of backends to sweep.

    Returns:
        Backends in the order they should appear in plots and CSVs. Always
        includes ``piquasso-jax (CPU)`` and ``perm_boost CPU`` (eager + jit);
        ``perm_boost GPU`` is appended only when ``jax.devices("cuda")``
        succeeds.
    """
    # ``jax.devices()`` triggers init of every registered backend; a wedged
    # CUDA driver (e.g. after a SIGKILL'd process) will then error out even
    # if the caller only asked for CPU. Probe CUDA first in a sub-process so
    # a broken driver gets isolated to that probe.
    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("cuda")[0]
    except RuntimeError as exc:
        print(f"  (skipping GPU backend: {str(exc).splitlines()[-1][:120]})")
        gpu = None

    def _pair(label, device, perm_fn, base_style, max_n_perm, max_n_grad):
        """Build matched eager and jit-compiled variants of one backend."""
        eager = Backend(
            label=label,
            device=device,
            perm_fn=perm_fn,
            style=base_style,
            max_n_perm=max_n_perm,
            max_n_grad=max_n_grad,
            jit=False,
        )
        # Same colour as the eager counterpart so they group visually in the
        # legend; marker overridden to '*' (and alpha lowered) to distinguish.
        jit_style = {**base_style, "marker": "*", "alpha": 0.65, "linewidth": 2}
        jitted = Backend(
            label=f"{label} (jit)",
            device=device,
            perm_fn=perm_fn,
            style=jit_style,
            max_n_perm=max_n_perm,
            max_n_grad=max_n_grad,
            jit=True,
        )
        return [eager, jitted]

    backends: list[Backend] = []
    # No jit variant for piquasso-jax: `permanent_with_reduction` calls
    # `assym_reduce`, whose output shape depends on the runtime values of
    # rows/cols, so XLA can't trace it (see permanent.py:25-30). The inner
    # Glynn loop is already @jit-decorated, so the eager call gets JIT anyway.
    backends.append(
        Backend(
            label="piquasso-jax (CPU)",
            device=cpu,
            perm_fn=permanent_with_reduction,
            style=dict(linestyle=":", marker="^", color="tab:gray"),
            max_n_perm=7,
            max_n_grad=6,
        )
    )
    backends.extend(
        _pair(
            "perm_boost CPU",
            cpu,
            boost_perm,
            dict(linestyle="--", marker="s", color="tab:blue"),
            max_n_perm=13,
            max_n_grad=11,
        )
    )
    if gpu is not None:
        backends.extend(
            _pair(
                "perm_boost GPU",
                gpu,
                boost_perm,
                dict(linestyle="-", marker="o", color="tab:red"),
                max_n_perm=13,
                max_n_grad=11,
            )
        )
    return backends


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _block(out) -> None:
    """Force XLA to materialize results before we stop the clock.

    Args:
        out: A JAX array, or an arbitrarily nested tuple/list of them.
    """
    if isinstance(out, (tuple, list)):
        for o in out:
            _block(o)
    elif hasattr(out, "block_until_ready"):
        out.block_until_ready()


def _time_calls(fn: Callable[[], object], repeats: int) -> tuple[float, str]:
    """Warm up once, then time ``repeats`` invocations of ``fn``.

    Args:
        fn: Zero-arg callable that produces JAX arrays.
        repeats: Number of timed calls after the warmup.

    Returns:
        ``(median_wall_seconds, status)``. ``status`` is ``"ok"`` on success
        or ``"error: <type>: <msg>"`` if any call raised (e.g. the FFI
        rejecting ``idx_max > 1e8``); in that case the time is ``nan``.
    """
    try:
        _block(fn())  # warmup (JIT compile / first launch)
    except Exception as exc:  # noqa: BLE001 -- backend may reject (e.g. idx_max cap)
        return (
            float("nan"),
            f"error: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}",
        )

    samples = []
    for _ in range(repeats):
        try:
            t0 = time.perf_counter()
            _block(fn())
            samples.append(time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            return (
                float("nan"),
                f"error: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}",
            )
    return float(np.median(samples)), "ok"


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def _make_inputs(backend: Backend, n: int, multiplicity: int, seed: int):
    """Generate inputs and place them on the backend's device.

    All backends consume the same dtypes (complex128 matrix, uint64 row/col
    multiplicities): perm_boost's FFI validates ``rows`` / ``cols`` as
    uint32 or uint64, and ``permanent_with_reduction`` accepts any integer
    dtype, so uint64 works for both.

    Args:
        backend: Target backend; its ``device`` field selects placement.
        n: Side length of the (square) unitary matrix.
        multiplicity: Constant value used for every row and column.
        seed: RNG seed passed to ``scipy.stats.unitary_group.rvs``.

    Returns:
        ``(matrix, rows, cols)`` -- all placed on ``backend.device``.
    """
    matrix_np = unitary_group.rvs(n, random_state=seed).astype(np.complex128)
    rows_np = np.full(n, multiplicity, dtype=np.uint64)
    cols_np = np.full(n, multiplicity, dtype=np.uint64)

    matrix = jax.device_put(jnp.asarray(matrix_np), backend.device)
    rows = jax.device_put(jnp.asarray(rows_np), backend.device)
    cols = jax.device_put(jnp.asarray(cols_np), backend.device)
    return matrix, rows, cols


def _perm_task(
    backend: Backend, n: int, multiplicity: int, seed: int
) -> Callable[[], object]:
    """Build a zero-arg callable that computes ``perm(A, rows, cols)``.

    Args:
        backend: Backend providing the permanent function and device.
        n: Matrix side length.
        multiplicity: Row/column multiplicity used for the inputs.
        seed: RNG seed for the unitary matrix.

    Returns:
        Closure ready to be timed by :func:`_time_calls`. Inputs are
        prebuilt; ``jax.jit`` is applied if ``backend.jit`` is True.
    """
    matrix, rows, cols = _make_inputs(backend, n, multiplicity, seed)
    fn = backend.perm_fn
    if backend.jit:
        fn = jax.jit(fn)
    device = backend.device

    def _call():
        # `jax.lax.platform_dependent` dispatches on the default JAX backend,
        # not the input device -- so pin the backend per call.
        with jax.default_device(device):
            return fn(matrix, rows, cols)

    return _call


def _split_real_imag(
    perm_fn: Callable[..., jnp.ndarray],
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    """Wrap ``perm_fn`` so its complex output is exposed as ``(real, imag)``.

    Lets ``jax.jacobian`` produce the real-part and imag-part Jacobians
    separately without the holomorphic shortcut. This is the form used in
    ``tests/perm_boost/test_grad_perm.py``.

    Args:
        perm_fn: Callable returning a complex scalar.

    Returns:
        Wrapped callable returning ``(res.real, res.imag)``.
    """

    def wrapped(primal, rows, cols):
        res = perm_fn(primal, rows, cols)
        return res.real, res.imag

    return wrapped


def _grad_task(
    backend: Backend, n: int, multiplicity: int, seed: int
) -> Callable[[], object]:
    """Build a zero-arg callable that computes the real/imag-split Jacobian.

    Args:
        backend: Backend providing the permanent function and device.
        n: Matrix side length.
        multiplicity: Row/column multiplicity used for the inputs.
        seed: RNG seed for the unitary matrix.

    Returns:
        Closure that evaluates ``jax.jacobian`` of the split wrapper,
        ready to be timed by :func:`_time_calls`.
    """
    matrix, rows, cols = _make_inputs(backend, n, multiplicity, seed)
    jacobian_fn = jax.jacobian(_split_real_imag(backend.perm_fn), argnums=0)
    if backend.jit:
        jacobian_fn = jax.jit(jacobian_fn)
    device = backend.device

    def _call():
        with jax.default_device(device):
            return jacobian_fn(matrix, rows, cols)

    return _call


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _sweep(
    label: str,
    n_values: range,
    task_builder: Callable[[Backend, int, int, int], Callable[[], object]],
    backends: list[Backend],
    multiplicity: int,
    repeats: int,
    seed: int,
    task_kind: str,
) -> dict[str, list[float]]:
    """Run a single (task, backends, n-range) sweep.

    For each ``n`` and backend, builds the task with ``task_builder`` and
    delegates timing to :func:`_time_calls`. Backends are skipped when
    ``n`` exceeds their per-task cap, or sticky-skipped after the first
    failure at a smaller ``n`` (subsequent ``n`` would only get slower).

    Args:
        label: Header printed before the sweep starts.
        n_values: Matrix sizes to iterate over.
        task_builder: Factory returning a zero-arg callable to be timed
            (typically :func:`_perm_task` or :func:`_grad_task`).
        backends: Backends to sweep.
        multiplicity: Row/column multiplicity forwarded to ``task_builder``.
        repeats: Timed calls per (backend, n).
        seed: RNG seed forwarded to ``task_builder``.
        task_kind: ``"perm"`` or ``"grad"``; selects which ``max_n_*`` cap
            to apply.

    Returns:
        Mapping from backend label to a list of median wall times aligned
        with ``n_values``. Skipped or failed entries are ``nan``.
    """
    print(f"\n=== {label} (multiplicity={multiplicity}, repeats={repeats}) ===")
    results: dict[str, list[float]] = {b.label: [] for b in backends}
    skip = {b.label: False for b in backends}

    def _cap(backend: Backend) -> int:
        return backend.max_n_perm if task_kind == "perm" else backend.max_n_grad

    for n in n_values:
        print(f"  n={n}")
        for backend in backends:
            if n > _cap(backend):
                results[backend.label].append(float("nan"))
                print(f"    {backend.label:>22}: SKIP (n>{_cap(backend)} cap)")
                continue
            if skip[backend.label]:
                results[backend.label].append(float("nan"))
                print(f"    {backend.label:>22}: SKIP (failed at smaller n)")
                continue
            fn = task_builder(backend, n, multiplicity, seed)
            t, status = _time_calls(fn, repeats=repeats)
            results[backend.label].append(t)
            if status == "ok":
                print(f"    {backend.label:>22}: {t * 1e3:9.3f} ms")
            else:
                skip[backend.label] = True
                print(f"    {backend.label:>22}: {status.upper()}")
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_csv(path: Path, n_values: range, results: dict[str, list[float]]) -> None:
    """Write a sweep's median wall times to a CSV file.

    Args:
        path: Destination CSV path; overwritten if it exists.
        n_values: Matrix sizes; written to the first column.
        results: Mapping from backend label to per-``n`` wall times.
    """
    headers = ["n"] + list(results.keys())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i, n in enumerate(n_values):
            w.writerow([n] + [results[label][i] for label in results])
    print(f"  wrote {path}")


def _plot(
    path: Path,
    title: str,
    n_values: range,
    results: dict[str, list[float]],
    backends: list[Backend],
    multiplicity: int,
) -> None:
    """Render a log-y plot of sweep results to ``path``.

    Args:
        path: Destination image path (extension picked up by Matplotlib).
        title: Plot title.
        n_values: Matrix sizes used for the x-axis.
        results: Mapping from backend label to per-``n`` wall times.
        backends: Backends in legend/style order; supplies ``style`` kwargs.
        multiplicity: Multiplicity reported in the x-axis label.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale("log")
    ax.set_xlabel(f"matrix size n  (row & column multiplicity = {multiplicity})")
    ax.set_ylabel("median wall time per call (s, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    style_by_label = {b.label: b.style for b in backends}
    for label, times in results.items():
        ax.plot(list(n_values), times, label=label, **style_by_label[label])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Argument list (default: ``sys.argv[1:]``).

    Returns:
        The populated ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--max-n-perm",
        type=int,
        default=14,
        help="largest n for the permanent-value sweep (default 14)",
    )
    p.add_argument(
        "--max-n-grad",
        type=int,
        default=11,
        help="largest n for the gradient sweep (default 11)",
    )
    p.add_argument("--min-n", type=int, default=2, help="smallest n (default 2)")
    p.add_argument(
        "--multiplicity",
        type=int,
        default=3,
        help="row & column multiplicity (default 3)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timed runs per (backend, n) after warmup (default 5)",
    )
    p.add_argument("--seed", type=int, default=42, help="unitary RNG seed (default 42)")
    p.add_argument("--quick", action="store_true", help="small sweep for smoke testing")
    p.add_argument(
        "--out",
        type=Path,
        default=HERE,
        help="output directory (default: alongside this script)",
    )
    return p.parse_args(argv)


def main(argv=None):
    """Run both sweeps and write CSV + PNG outputs.

    Args:
        argv: CLI argument list forwarded to :func:`_parse_args`.

    Returns:
        Process exit code (always ``0`` on a completed run).
    """
    args = _parse_args(argv)
    if args.quick:
        args.max_n_perm = 8
        args.max_n_grad = 6
        args.repeats = 3

    args.out.mkdir(parents=True, exist_ok=True)
    backends = _build_backends()
    print(f"Backends: {[b.label for b in backends]}")
    print(f"JAX devices: {jax.devices()}")

    perm_results = _sweep(
        "Permanent value",
        n_values=range(args.min_n, args.max_n_perm + 1),
        task_builder=_perm_task,
        backends=backends,
        multiplicity=args.multiplicity,
        repeats=args.repeats,
        seed=args.seed,
        task_kind="perm",
    )
    grad_results = _sweep(
        "Real/imag-split Jacobian",
        n_values=range(args.min_n, args.max_n_grad + 1),
        task_builder=_grad_task,
        backends=backends,
        multiplicity=args.multiplicity,
        repeats=args.repeats,
        seed=args.seed,
        task_kind="grad",
    )

    print("\nWriting outputs:")
    _write_csv(
        args.out / "results_permanent.csv",
        range(args.min_n, args.max_n_perm + 1),
        perm_results,
    )
    _write_csv(
        args.out / "results_gradient.csv",
        range(args.min_n, args.max_n_grad + 1),
        grad_results,
    )
    _plot(
        args.out / "permanent_benchmark.png",
        f"Permanent benchmark -- JAX pipeline (multiplicity {args.multiplicity})",
        range(args.min_n, args.max_n_perm + 1),
        perm_results,
        backends,
        args.multiplicity,
    )
    _plot(
        args.out / "gradient_benchmark.png",
        "Real/imag-split Jacobian benchmark -- JAX pipeline "
        f"(multiplicity {args.multiplicity})",
        range(args.min_n, args.max_n_grad + 1),
        grad_results,
        backends,
        args.multiplicity,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

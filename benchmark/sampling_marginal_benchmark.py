#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

"""
Benchmark of ParticleNumberMeasurement on a strict subset of modes (issue #499):
the efficient marginal algorithm vs. full-sampling-then-discarding.

This justifies the switch implemented in
``piquasso._simulators.sampling.marginal.prefer_marginal_sampling``: for small
numbers of measured modes ``k`` the marginal algorithm is orders of magnitude
faster, while for large ``k`` the discard approach wins (and the heuristic falls
back to it).

Run with: ``python benchmark/sampling_marginal_benchmark.py``
"""

import time

import numpy as np
from scipy.stats import unitary_group

import piquasso as pq
import piquasso._simulators.sampling.simulation_steps as steps


def _time_execution(d, occupation, modes, shots, force_discard):
    U = unitary_group.rvs(d, random_state=3)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(occupation)
        pq.Q(*range(d)) | pq.Interferometer(U)
        pq.Q(*modes) | pq.ParticleNumberMeasurement()

    original = steps.prefer_marginal_sampling
    if force_discard:
        steps.prefer_marginal_sampling = lambda *args, **kwargs: False
    try:
        start = time.perf_counter()
        pq.SamplingSimulator(d=d).execute(program, shots=shots)
        return time.perf_counter() - start
    finally:
        steps.prefer_marginal_sampling = original


if __name__ == "__main__":
    shots = 2000
    print(f"shots = {shots}\n")
    print(f"{'d':>3} {'k':>3} {'efficient (ms)':>15} {'discard (ms)':>14} {'speedup':>9}")
    for d, k in [(6, 1), (6, 2), (6, 3), (8, 2), (10, 2)]:
        occupation = [1] * min(4, d) + [0] * (d - min(4, d))
        modes = tuple(range(k))
        efficient = _time_execution(d, occupation, modes, shots, force_discard=False)
        discard = _time_execution(d, occupation, modes, shots, force_discard=True)
        print(
            f"{d:>3} {k:>3} {efficient * 1e3:>15.2f} {discard * 1e3:>14.2f} "
            f"{discard / efficient:>8.1f}x"
        )

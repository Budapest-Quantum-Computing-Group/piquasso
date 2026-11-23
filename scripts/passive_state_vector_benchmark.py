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

"""Black-box benchmark of the state vector calculation in `PassiveSimulator`.

Only the public API is used: a program is executed with `PassiveSimulator`, and the
time of the `state_vector` property of the resulting state is measured for three
kinds of postselection:

- none,
- weak: the last mode is postselected on vacuum,
- strong: all but the first mode are postselected on one photon each.

Usage::

    python scripts/passive_state_vector_benchmark.py
"""

import statistics
import time

import scipy.stats

import piquasso as pq

# (title, postselection, [(modes, photons), ...])
BENCHMARKS = [
    ("not postselected", "none", [(8, 6), (10, 6), (12, 6), (14, 6), (16, 6)]),
    ("weakly postselected", "weak", [(8, 6), (10, 6), (12, 6), (14, 6), (16, 6)]),
    ("strongly postselected", "strong", [(8, 8), (12, 12), (16, 16), (20, 20)]),
]
REPEAT = 10


def build_program(interferometer, photons, postselection):
    d = len(interferometer)

    occupation_numbers = [0] * d
    for i in range(photons):
        occupation_numbers[i % d] += 1

    with pq.Program() as program:
        pq.Q(all) | pq.NumberState(occupation_numbers)
        pq.Q(all) | pq.Interferometer(interferometer)

        if postselection == "weak":
            pq.Q(d - 1) | pq.PostSelectPhotons(photon_counts=(0,))
        elif postselection == "strong":
            pq.Q(*range(1, d)) | pq.PostSelectPhotons(photon_counts=(1,) * (d - 1))

    return program


def benchmark(d, photons, postselection):
    interferometer = scipy.stats.unitary_group.rvs(d, random_state=0)

    simulator = pq.PassiveSimulator(d=d, config=pq.Config(cutoff=photons + 1))
    program = build_program(interferometer, photons, postselection)
    state = simulator.execute(program).state

    timings = []

    for _ in range(REPEAT + 1):
        start = time.perf_counter()
        state_vector = state.state_vector
        timings.append(time.perf_counter() - start)

    return timings[1:], len(state_vector)


def main():
    for title, postselection, cases in BENCHMARKS:
        print(title)
        print(
            f"{'modes':>5} {'photons':>7} {'dim':>9} {'median [s]':>12} {'min [s]':>12}"
        )

        for d, photons in cases:
            timings, dim = benchmark(d, photons, postselection)

            print(
                f"{d:>5} {photons:>7} {dim:>9} "
                f"{statistics.median(timings):>12.6f} {min(timings):>12.6f}"
            )


if __name__ == "__main__":
    main()

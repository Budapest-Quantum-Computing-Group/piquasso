#
# Copyright 2021-2025 Budapest Quantum Computing Group
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
Benchmarking the Piquasso vs. TheWalrus loop hafnian with repetitions.
"""

import time

import json

from piquasso._math.hafnian import loop_hafnian_with_reduction
from thewalrus import loop_hafnian as loop_hafnian_repeated

import matplotlib.pyplot as plt

import numpy as np

np.set_printoptions(suppress=True, linewidth=200)


if __name__ == "__main__":
    x = []
    y = []
    z = []

    ITER = 100

    FILENAME = f"loop_hafnian_benchmark_{int(time.time())}.json"

    # Compilation
    d = 2
    A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    A = A + A.T
    diag = np.random.rand(d) + 1j * np.random.rand(d)
    occupation_numbers = 2 * np.ones(d, dtype=int)
    loop_hafnian_with_reduction(A, diag, occupation_numbers)
    loop_hafnian_repeated(A, diag, occupation_numbers)
    ###

    for d in range(6, 24, 1):
        print(d)
        x.append(d)
        A = np.random.rand(d, d) + 1j * np.random.rand(d, d)

        A = A + A.T

        diag = np.random.rand(d) + 1j * np.random.rand(d)

        occupation_numbers = 2 * np.ones(d, dtype=int)

        sum_ = 0.0

        for _ in range(ITER):
            print("|", end="", flush=True)
            start_time = time.time()
            loop_hafnian_with_reduction(A, diag, occupation_numbers)
            sum_ += time.time() - start_time

        y.append(sum_ / ITER)
        print()

        sum_ = 0.0

        for _ in range(ITER):
            print("-", end="", flush=True)
            start_time = time.time()
            loop_hafnian_repeated(A, diag, occupation_numbers)
            sum_ += time.time() - start_time

        z.append(sum_ / ITER)
        print()

        with open(FILENAME, "w") as f:
            json.dump(dict(x=x, y=y, z=z), f, indent=4)

    plt.scatter(x, y, label="Piquasso")
    plt.scatter(x, z, label="TheWalrus")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("d [-]")
    plt.ylabel("Execution time [s]")

    plt.show()

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
Benchmarking the Piquasso vs. TheWalrus hafnian with repetitions.
"""

import time
import json
import numpy as np
from scipy.linalg import schur
import matplotlib.pyplot as plt

from piquasso._math.pfaffian import pfaffian


def pfaffian_naive(matrix):
    if matrix.shape[0] == 0:
        return 1.0

    if matrix.shape[0] % 2 == 1:
        return 0.0

    blocks, O = schur(matrix)
    a = np.diag(blocks, 1)[::2]

    return np.prod(a) * np.linalg.det(O)


np.set_printoptions(suppress=True, linewidth=200)

if __name__ == "__main__":
    x = []
    y = []
    z = []

    ITER = 100

    FILENAME = f"pfaffian_benchmark_{int(time.time())}.json"

    # Compilation
    d = 2
    A = np.random.rand(d, d)
    A = A - A.T

    pfaffian(A)

    for d in range(2, 100 + 1, 2):
        print(d)
        x.append(d)
        A = np.random.rand(d, d)

        A = A - A.T

        sum_ = 0.0
        for _ in range(ITER):
            print("|", end="", flush=True)
            start_time = time.time()
            pfaffian(A)
            sum_ += time.time() - start_time

        y.append(sum_ / ITER)
        print()

        sum_ = 0.0
        for _ in range(ITER):
            print("/", end="", flush=True)
            start_time = time.time()
            pfaffian_naive(A)
            sum_ += time.time() - start_time

        z.append(sum_ / ITER)
        print()

        with open(FILENAME, "w") as f:
            json.dump(dict(x=x, y=y, z=z), f, indent=4)

    plt.scatter(x, y, label="Piquasso C++")
    plt.scatter(x, z, label="Piquasso Python")
    plt.legend()
    plt.xlabel("d [-]")
    plt.ylabel("Execution time [s]")

    plt.show()

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

from typing import List
import numpy as np


def plot_wigner_function(
    vals: np.ndarray,
    positions: List[List[float]],
    momentums: List[List[float]],
    levels: int = 50,
) -> None:
    """
    Plots the Wigner function using matplotlib.

    Args:
        vals: Values to plot.
        positions: List of lists of positions.
        momentums: List of lists of momentums.
        levels: Number of contour levels.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("The visualization feature requires matplotlib.")

    plt.contourf(positions, momentums, vals, levels=levels, cmap="RdBu")
    plt.colorbar()
    plt.xlabel("Position")
    plt.ylabel("Momentum")
    plt.title("Wigner Function")
    plt.show()

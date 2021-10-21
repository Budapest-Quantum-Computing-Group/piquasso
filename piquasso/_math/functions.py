#
# Copyright 2021 Budapest Quantum Computing Group
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


def gaussian_wigner_function(
    positions: List[List[float]],
    momentums: List[List[float]],
    *,
    d: int,
    mean: np.ndarray,
    cov: np.ndarray
) -> np.ndarray:
    return np.array(
        [
            [
                gaussian_wigner_function_for_scalar(
                    [*position, *momentum], d=d, mean=mean, cov=cov
                )
                for position in positions
            ]
            for momentum in momentums
        ]
    )


def gaussian_wigner_function_for_scalar(
    X: List[float], *, d: int, mean: np.ndarray, cov: np.ndarray
) -> float:
    return (
        (1 / (np.pi ** d))
        * np.sqrt((1 / np.linalg.det(cov)))
        * np.exp(-(X - mean) @ np.linalg.inv(cov) @ (X - mean))
    ).real

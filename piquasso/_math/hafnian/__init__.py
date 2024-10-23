#
# Copyright 2021-2024 Budapest Quantum Computing Group
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
Package containing hafnian and loop hafnian calculations.

Most of the code in this package are translated versions of the PiquassoBoost C++ code
from https://github.com/Budapest-Quantum-Computing-Group/piquassoboost.

Moreover, the algorithms are enhanced by factoring in repetitions according to
https://arxiv.org/abs/2108.01622.
"""

from .plain_hafnian import hafnian_with_reduction, hafnian_with_reduction_batch

from .loop_hafnian import (
    loop_hafnian_with_reduction,
    loop_hafnian_with_reduction_batch,
)


__all__ = [
    "hafnian_with_reduction",
    "hafnian_with_reduction_batch",
    "loop_hafnian_with_reduction",
    "loop_hafnian_with_reduction_batch",
]

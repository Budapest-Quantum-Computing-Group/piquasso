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
Fermionic Gaussian Simulations
******************************

This is a package for Fermionic Linear Optics (FLO) or Fermionic Gaussian states.

Note:
    For consistency, we tried to the conventions from the SciPost Physics Lecture Note
    "Fermionic Gaussian states: an introduction to numerical approaches" from J. Surace
    and L. Tagliacozzo, `arXiv:2111.08343 <https://arxiv.org/pdf/2111.08343>`_.
    However, this article uses anti-lexicographic ordering (i.e. :math:`| 1 \langle` is
    before :math:`| 0 \langle` in order), and we prefer lexicographic ordering,
    therefore we changed some calculations accordingly.
"""

from .simulator import GaussianSimulator
from .state import GaussianState

__all__ = [
    "GaussianSimulator",
    "GaussianState",
]

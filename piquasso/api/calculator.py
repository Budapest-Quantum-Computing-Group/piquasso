#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

from piquasso._math.permanent import glynn_gray_permanent
from piquasso._math.hafnian import hafnian_with_reduction, loop_hafnian_with_reduction

from piquasso.api.typing import PermanentFunction, HafnianFunction, LoopHafnianFunction


class Calculator:
    """The customizable calculations for a simulation."""

    def __init__(
        self,
        permanent_function: PermanentFunction = glynn_gray_permanent,
        hafnian_function: HafnianFunction = hafnian_with_reduction,
        loop_hafnian_function: LoopHafnianFunction = loop_hafnian_with_reduction,
    ):
        self.permanent_function = permanent_function
        self.hafnian_function = hafnian_function
        self.loop_hafnian_function = loop_hafnian_function

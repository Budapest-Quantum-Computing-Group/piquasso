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

import abc

from piquasso.api.result import Result

from piquasso.api.program import Program


class Computer(abc.ABC):
    """Base class for all quantum computers or simulators supported by Piquasso."""

    @abc.abstractmethod
    def execute(
        self,
        program: Program,
        shots: int = 1,
    ) -> Result:
        """Executes the program `shots` times.

        Args:
            program (Program): The program to be executes.
            shots (int, optional):
                The number of times the program should be executed. Defaults to 1.

        Returns:
            Result: The result of the execution containing `shots` number of samples.
        """
        pass

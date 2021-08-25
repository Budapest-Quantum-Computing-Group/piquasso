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

from piquasso.api.errors import InvalidProgram
from piquasso.api.instruction import Instruction
from piquasso.api.mode import Q


class Measurement(Instruction):
    def _apply_to_program_on_register(self, program, register: Q) -> None:
        if any(
            isinstance(instruction, type(self))
            for instruction in program.instructions
        ):
            raise InvalidProgram("Measurement already registered.")

        super()._apply_to_program_on_register(program, register)

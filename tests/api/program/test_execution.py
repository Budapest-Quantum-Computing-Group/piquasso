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

import piquasso as pq


def test_instruction_execution(program):
    instruction = pq.DummyInstruction(param=420)
    with program:
        pq.Q(0, 1) | instruction

    program.execute()

    program._circuit.dummy_instruction.assert_called_once_with(
        instruction, program.state
    )


def test_register_instruction_from_left_hand_side(program):
    instruction = pq.DummyInstruction(param=420)
    with program:
        instruction | pq.Q(0, 1)

    program.execute()

    program._circuit.dummy_instruction.assert_called_once_with(
        instruction, program.state
    )

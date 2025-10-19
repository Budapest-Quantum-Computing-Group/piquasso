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

import piquasso as pq


def test_program_copy():
    program = pq.Program()

    program_copy = program.copy()

    assert program_copy is not program


def test_program_repr():
    program = pq.Program()

    assert repr(program) == "Program(instructions=[])"


def test_program_get_number_of_modes_with_two_modes():
    """Test that get_number_of_modes correctly infers d=2."""
    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector((2, 0))
        pq.Q(0, 1) | pq.Beamsplitter5050()

    assert program.get_number_of_modes() == 2


def test_program_get_number_of_modes_with_single_mode():
    """Test that get_number_of_modes correctly infers d=1."""
    with pq.Program() as program:
        pq.Q(0) | pq.StateVector((2,))

    assert program.get_number_of_modes() == 1


def test_program_get_number_of_modes_with_sparse_modes():
    """Test that get_number_of_modes infers based on max mode index."""
    with pq.Program() as program:
        pq.Q(0, 2, 5) | pq.StateVector((1, 0, 1))

    # Mode 5 is the highest, so we need d=6
    assert program.get_number_of_modes() == 6


def test_program_get_number_of_modes_empty_program():
    """Test that get_number_of_modes returns None for empty program."""
    with pq.Program() as program:
        pass

    assert program.get_number_of_modes() is None


def test_program_get_number_of_modes_with_multiple_instructions():
    """Test that get_number_of_modes considers all instructions."""
    with pq.Program() as program:
        pq.Q(0) | pq.StateVector((1,))
        pq.Q(1) | pq.StateVector((1,))
        pq.Q(2, 3) | pq.Beamsplitter5050()

    # Mode 3 is the highest, so we need d=4
    assert program.get_number_of_modes() == 4

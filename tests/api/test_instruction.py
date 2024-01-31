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

import pytest

import piquasso as pq

from piquasso.api.exceptions import PiquassoException


def test_registering_instruction_by_subclassing():
    class OtherBeamsplitter(pq.Instruction):
        pass

    assert pq.Instruction.get_subclass("OtherBeamsplitter") is OtherBeamsplitter


def test_subclassing_instruction_with_existing_name_is_successful():
    class Beamsplitter(pq.Instruction):
        pass

    assert pq.Instruction.get_subclass("Beamsplitter") is Beamsplitter
    assert pq.Beamsplitter is not Beamsplitter

    # Teardown
    pq.Instruction.set_subclass(pq.Beamsplitter)
    assert pq.Instruction.get_subclass("Beamsplitter") is pq.Beamsplitter


def test_set_subclass_with_no_subclass():
    any_other_class = object

    with pytest.raises(PiquassoException):
        pq.Instruction.set_subclass(any_other_class)

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

from fractions import Fraction

from piquasso.api.branch import Branch


def test_Branch_state(FakeState):
    state = FakeState(d=3, connector=None)
    branch = Branch(state=state, outcome=(0, 1), frequency=Fraction(42, 100))

    assert branch.state is state


def test_Branch_outcome(FakeState):
    state = FakeState(d=3, connector=None)
    branch = Branch(state=state, outcome=(0, 1), frequency=Fraction(42, 100))

    assert branch.outcome == (0, 1)


def test_Branch_frequency(FakeState):
    state = FakeState(d=3, connector=None)
    branch = Branch(state=state, outcome=(0, 1), frequency=Fraction(42, 100))

    assert branch.frequency == Fraction(42, 100)


def test_Branch_repr(FakeState):
    state = FakeState(d=3, connector=None)
    branch = Branch(state=state, outcome=(0, 1), frequency=Fraction(42, 100))

    assert (
        repr(branch)
        == "Branch(state=FakeState(d=3, config=Config(), connector=None), outcome=(0, 1), frequency=21/50)"  # noqa: E501
    )

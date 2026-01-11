#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

import re

import pytest

import piquasso as pq

from piquasso.api.result import Result
from piquasso.api.branch import Branch

from fractions import Fraction


def test_Result_branches(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    branch1 = Branch(state=state1, outcome=(0,), frequency=Fraction(498, 1000))
    branch2 = Branch(state=state2, outcome=(1,), frequency=Fraction(502, 1000))

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=1000)

    assert result.branches == [branch1, branch2]


def test_Result_state_with_single_branch(FakeState):
    state = FakeState(d=3, connector=pq.NumpyConnector())

    branch = Branch(state=state, outcome=(0,), frequency=1000)

    branches = [branch]

    result = Result(branches=branches, config=pq.Config(), shots=1000)

    assert result.state is state


def test_Result_state_raises_NotImplementedCalculation_for_multiple_branches(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    branch1 = Branch(state=state1, outcome=(0,), frequency=Fraction(498, 1000))
    branch2 = Branch(state=state2, outcome=(1,), frequency=Fraction(502, 1000))

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=1000)

    with pytest.raises(
        pq.api.exceptions.NotImplementedCalculation,
        match=(
            "This feature is not yet implemented. However, you can access the "
            "post-measurement state in the `Result.branches` attribute."
        ),
    ):
        _ = result.state


def test_Result_samples(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    shots = 1000

    f0 = Fraction(498, 1000)
    f1 = Fraction(502, 1000)

    branch1 = Branch(state=state1, outcome=(0,), frequency=f0)
    branch2 = Branch(state=state2, outcome=(1,), frequency=f1)

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=shots)

    samples = result.samples

    assert samples.count((0,)) == int(f0 * shots)
    assert samples.count((1,)) == int(f1 * shots)


def test_Result_get_counts(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    shots = 1000

    f0 = Fraction(498, 1000)
    f1 = Fraction(502, 1000)

    branch1 = Branch(state=state1, outcome=(0,), frequency=f0)
    branch2 = Branch(state=state2, outcome=(1,), frequency=f1)

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=shots)

    counts = result.get_counts()

    assert counts == {(0,): int(f0 * shots), (1,): int(f1 * shots)}


def test_Result_to_subgraph_nodes(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    shots = 2

    f0 = Fraction(1, 2)
    f1 = Fraction(1, 2)

    branch1 = Branch(state=state1, outcome=(0, 1, 1), frequency=f0)
    branch2 = Branch(state=state2, outcome=(1, 2, 0), frequency=f1)

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=shots)

    nodes = result.to_subgraph_nodes()

    assert [1, 2] in nodes
    assert [0, 1, 1] in nodes


def test_Result_repr(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    branch1 = Branch(state=state1, outcome=(0,), frequency=498)
    branch2 = Branch(state=state2, outcome=(1,), frequency=502)

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=1000)

    assert (
        repr(result)
        == "Result(branches=[Branch(state=FakeState(d=3, config=Config(), connector=NumpyConnector()), outcome=(0,), frequency=498), Branch(state=FakeState(d=3, config=Config(), connector=NumpyConnector()), outcome=(1,), frequency=502)], config=Config(), shots=1000)"  # noqa: E501
    )


def test_Result_samples_with_shots_None(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    f0 = 1 / 3
    f1 = 2 / 3

    branch1 = Branch(state=state1, outcome=(0,), frequency=f0)
    branch2 = Branch(state=state2, outcome=(1,), frequency=f1)

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=None)

    with pytest.raises(
        pq.api.exceptions.NotImplementedCalculation,
        match=re.escape(
            "The 'Result.samples' property is not available with the exact probability "
            "distribution (i.e., when 'shots=None' is used in the simulation). Please "
            "execute the program with a finite number of shots to obtain samples."
        ),
    ):
        _ = result.samples


def test_Result_get_counts_with_shots_None(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    f0 = 1 / 3
    f1 = 2 / 3

    branch1 = Branch(state=state1, outcome=(0,), frequency=f0)
    branch2 = Branch(state=state2, outcome=(1,), frequency=f1)

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=None)

    with pytest.raises(
        pq.api.exceptions.NotImplementedCalculation,
        match=re.escape(
            "The 'Result.samples' property is not available with the exact probability "
            "distribution (i.e., when 'shots=None' is used in the simulation). Please "
            "execute the program with a finite number of shots to obtain samples."
        ),
    ):
        _ = result.get_counts()


def test_result_outcome_map(FakeState):
    state1 = FakeState(d=3, connector=pq.NumpyConnector())
    state2 = FakeState(d=3, connector=pq.NumpyConnector())

    f0 = 1 / 3
    f1 = 2 / 3

    branch1 = Branch(state=state1, outcome=(0, 1), frequency=f0)
    branch2 = Branch(state=state2, outcome=(1, 0), frequency=f1)

    branches = [branch1, branch2]

    result = Result(branches=branches, config=pq.Config(), shots=None)

    outcome_map = result.outcome_map

    assert outcome_map == {
        (0, 1): {"frequency": f0, "state": state1},
        (1, 0): {"frequency": f1, "state": state2},
    }

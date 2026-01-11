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

from typing import Optional, Tuple, Union

from .state import State

from fractions import Fraction


class Branch:
    r"""Represents a branching in the quantum computation caused by a measurement.

    Describes a branch in the computation tree that arises due to a measurement
    operation. Upon each measurement, the quantum state can collapse into multiple
    possible outcomes, each leading to a different branch in the computation.
    Roughly, a quantum state :math:`|\psi\rangle` undergoing a measurement with possible
    outcomes :math:`\{m_i\}` will collapse into one of the post-measurement states
    :math:`\{|\psi_i\rangle\}` with corresponding probabilities :math:`\{p_i\}` (i.e.,
    an ensemble of quantum states). Each of these possible outcomes define a separate
    branch in the computation, and a `Branch` object is holding momentary information
    about these branches.

    Each `Branch` instance contains the post-measurement quantum state, the outcomes
    of the measurements that led to this branch, and the frequency (or weight) of
    this branch in the overall ensemble. The frequency indicates how often this
    particular outcome occurred relative to the total number of shots (simulations)
    performed, which is an approximation of the probability of observing this
    outcome.

    If the simulation was run with `shots=None`, indicating that the exact probability
    distribution was calculated instead of sampling, the frequency corresponds to the
    exact probability of the outcome.

    The data in the branch is illustrated as follows. Consider the following setup,
    where a measurement is performed with two possible outcomes, 0 and 1:

    .. raw:: html
        :class: only-light

        <div style="text-align: center;">
            <img
                src="../_static/branch_illustration.svg"
                style="width: 35%; max-width: 500px; height: auto;"
                alt="Branch illustration"
            >
        </div>

    .. raw:: html
        :class: only-dark

        <div style="text-align: center;">
            <img
                src="../_static/branch_illustration_dark.svg"
                style="width: 35%; max-width: 500px; height: auto;"
                alt="Branch illustration"
            >
        </div>

    In this case, two branches are created, one for each possible outcome of the
    measurement. The branches contain the post-measurement quantum states
    :math:`|\psi_0\rangle` and :math:`|\psi_1\rangle`, the outcomes 0 and 1, and
    their corresponding probabilities :math:`p_0` and :math:`p_1`,
    corresponding to each outcome. Since the circuit is executed `shots` times, instead
    of the probabilities, the frequencies are stored in the branches, which are
    calculated as :math:`f_i = n_i / \text{shots}`, where :math:`n_i` is the number of
    times the outcome :math:`i` was observed.

    :ivar state:
        The post-measurement quantum state corresponding to the branch. If the branch
        has no associated quantum state (e.g., all the modes are measured), then
        this is set to `None`.
    :ivar outcome:
        The accumulated outcome corresponding to the measurement(s) that caused the
        branching. If no measurements were performed, then this is is set to `None`.
    :ivar frequency:
        The contribution of the state in the final ensemble. If the branch is the result
        of a measurement with multiple possible outcomes, then this is set to the
        ratio of the number of times `outcome` appears in the simulation, divided by
        `shots` (which is the total number of samples taken). If the branch is not the
        result of a measurement (i.e., the branch was not created after a branching
        event), then this value is simply `1`.
    """

    def __init__(
        self,
        state: Optional[State] = None,
        outcome: Optional[Tuple[Union[int, float], ...]] = None,
        frequency: Optional[Fraction] = None,
    ):
        self.state = state if state is not None and state.d != 0 else None
        self.outcome = outcome if outcome is not None else tuple()
        self.frequency = frequency if frequency is not None else Fraction(1)

    def __repr__(self):
        strings = []
        if self.state is not None:
            strings.append(f"state={self.state}")
        if self.outcome is not tuple():
            strings.append(f"outcome={self.outcome}")

        strings.append(f"frequency={self.frequency}")

        return f"Branch({', '.join(strings)})"

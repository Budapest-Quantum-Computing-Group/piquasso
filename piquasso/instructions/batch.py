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

from piquasso.api.instruction import Preparation, Gate, BatchInstruction


class BatchPrepare(Preparation, BatchInstruction):
    r"""Allows for batch processing of multiple states.

    NOTE: This feature is experimental.
    """

    def __init__(self, subprograms):
        super().__init__(params=dict(subprograms=subprograms))


class BatchApply(Gate, BatchInstruction):
    r"""Applies subprograms to each element in the batch.

    NOTE: This feature is experimental.
    """

    def __init__(self, subprograms):
        super().__init__(params=dict(subprograms=subprograms))

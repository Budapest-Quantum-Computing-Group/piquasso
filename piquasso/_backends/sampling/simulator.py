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

from .state import SamplingState

from piquasso.api.computer import Simulator


class SamplingSimulator(Simulator):
    _instruction_map = {
        "StateVector": "_state_vector",
        "Beamsplitter": "_passive_linear",
        "Phaseshifter": "_passive_linear",
        "MachZehnder": "_passive_linear",
        "Fourier": "_passive_linear",
        "Sampling": "_sampling",
        "Interferometer": "_passive_linear",
        "Loss": "_loss",
    }

    state_class = SamplingState

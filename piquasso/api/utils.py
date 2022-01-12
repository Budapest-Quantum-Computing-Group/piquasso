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

from piquasso.api.program import Program
from piquasso.api.simulator import Simulator


def as_code(program: Program, simulator: Simulator, shots: int = 1) -> str:
    return (
        "import numpy as np\n"
        "import piquasso as pq\n"
        "\n"
        "\n"
        f"{program._as_code()}\n"
        "\n"
        f"simulator = {simulator._as_code()}\n"
        "\n"
        f"result = simulator.execute(program, shots={shots})\n"
    )

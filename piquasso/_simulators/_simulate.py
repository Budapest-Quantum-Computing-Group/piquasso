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

from typing import Optional, Union

from .gaussian import GaussianSimulator
from .fock import FockSimulator, PureFockSimulator
from .sampling import SamplingSimulator

from ..api.exceptions import NotImplementedCalculation
from ..api.connector import BaseConnector
from ..api.program import Program
from ..api.config import Config
from ..api.result import Result


def _resolve_simulator_class(program, connector):
    simulator_classes = (
        GaussianSimulator,
        SamplingSimulator,
        PureFockSimulator,
        FockSimulator,
    )

    possible_simulator_classes = []

    for simulator_class in simulator_classes:
        supported_instruction_classes = set(simulator_class._instruction_map.keys())

        if all(
            instruction.__class__ in supported_instruction_classes
            for instruction in program.instructions
        ):
            possible_simulator_classes.append(simulator_class)

    if not possible_simulator_classes:
        raise NotImplementedCalculation(
            "The program cannot be simulated by any of the built-in simulators in "
            "Piquasso. Please verify that the program's instructions are supported by "
            "at least one built-in simulator. For requests to implement a new "
            "simulation pathway, please open an issue on the Piquasso GitHub "
            "repository: "
            "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
        )

    if connector is None:
        return possible_simulator_classes[0]

    for simulator_class in possible_simulator_classes:
        if isinstance(connector, simulator_class._supported_connector_classes()):
            return simulator_class

    possible_connectors = {
        connector_class
        for simulator_class in possible_simulator_classes
        for connector_class in simulator_class._supported_connector_classes()
    }

    possible_connector_names = ", ".join(
        sorted(connector_class.__name__ for connector_class in possible_connectors)
    )

    connector_name = type(connector).__name__

    raise NotImplementedCalculation(
        f"The specified program cannot be simulated by the connector "
        f"'{connector_name}'. Please consider using another connector from "
        f"'{possible_connector_names}', or open an issue on the Piquasso GitHub "
        "repository: "
        "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
    )


def simulate(
    program: Program,
    number_of_modes: Optional[int] = None,
    *,
    connector: Optional[BaseConnector] = None,
    config: Optional[Config] = None,
    shots: Union[int, None] = 1,
) -> Result:
    """Performs a simulation according to the prescribed program.

    Args:
        program (Program): The quantum program to simulate.
        number_of_modes (Optional[int]): The number of modes in the simulation.
            If ``None``, the simulator determines the number of modes from the
            program where supported.
        connector (BaseConnector): The connector to use for the simulation.
        config (Config): The configuration to use for the simulation.
        shots (Union[int, None]): The number of shots to simulate. If ``None``,
            the simulation is executed using the exact probability distribution
            instead of finite-shot sampling.

    Returns:
        Result: The result of the simulation.
    """

    simulator_class = _resolve_simulator_class(program, connector)

    simulator_kwargs = {
        key: value
        for key, value in {
            "d": number_of_modes,
            "connector": connector,
            "config": config,
        }.items()
        if value is not None
    }

    simulator = simulator_class(**simulator_kwargs)

    return simulator.execute(program, shots=shots)

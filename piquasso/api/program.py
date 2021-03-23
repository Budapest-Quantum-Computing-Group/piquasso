#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import json
import blackbird
import inspect
import collections

from piquasso import operations
from piquasso.core import _context
from piquasso.core.registry import _create_instance_from_mapping


class Program:
    r"""The representation for a quantum program.

    This also specifies a context in which all the operations should be
    specified.

    Attributes:
        state (State): The initial quantum state.
        circuit (Circuit):
            The circuit on which the quantum program should run.
        operations (list):
            The set of operations, e.g. quantum gates and measurements.
    """

    def __init__(
        self,
        state=None,
        operations=None,
    ):
        self._register_state(state)
        self.operations = operations or []
        self.results = []

    def _register_state(self, state):
        self.state = state

        self.circuit = state.circuit_class(state, program=self) if state else None

    @property
    def d(self):
        """The number of modes, on which the state of the program is defined.

        Returns:
            int: The number of modes.
        """
        return self.state.d

    @classmethod
    def from_properties(cls, properties):
        """Creates a `Program` instance from a mapping.

        The currently supported format is

        .. code-block:: python

            {
                "state": {
                    "type": <STATE_CLASS_NAME>,
                    "properties": {
                        ...
                    }
                },
                "operations": [
                    {
                        "type": <OPERATION_CLASS_NAME>,
                        "properties": {
                            ...
                        }
                    }
                ]
            }

        Note:
            Numeric arrays and complex numbers are not yet supported.

        Args:
            properties (collections.Mapping):
                The desired `Program` instance in the format of a mapping.

        Returns:
            Program: A `Program` initialized using the specified mapping.
        """

        return cls(
            state=_create_instance_from_mapping(properties["state"]),
            operations=list(
                map(
                    _create_instance_from_mapping,
                    properties["operations"],
                )
            )
        )

    @classmethod
    def from_json(cls, json_):
        """Creates a `Program` instance from JSON.

        Almost the same as :meth:`from_properties`, but with JSON parsing.

        Args:
            json_ (str): The JSON formatted program.

        Returns:
            Program: A program initialized with the JSON data.
        """
        properties = json.loads(json_)

        return cls.from_properties(properties)

    def execute(self):
        """Executes the collected operations on the circuit."""

        self.circuit.execute_operations(self.operations)

        return self.results

    def __enter__(self):
        _context.current_program = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _context.current_program = None

    def _blackbird_operation_to_operation(self, blackbird_operation):
        """
        Maps one element of the `operations` of a `BlackbirdProgram` into an
        element of `self.operations`.

        Args:
            blackbird_operation (dict): An element of the `BlackbirdProgram.operations`

        Returns:
            Operation:
                Instance of :class:`Operation` corresponding to the operation defined
                in Blackbird.
        """

        pq_operation_class = {
            "Dgate": operations.gates.D,
            "Xgate": operations.gates.X,
            "Zgate": operations.gates.Z,
            "Sgate": operations.gates.S,
            "Pgate": operations.gates.P,
            "Vgate": None,
            "Kgate": operations.gates.K,
            "Rgate": operations.gates.R,
            "BSgate": operations.gates.B,
            "MZgate": operations.gates.MZ,
            "S2gate": operations.gates.S2,
            "CXgate": operations.gates.CX,
            "CZgate": operations.gates.CZ,
            "CKgate": operations.gates.CK,
            "Fouriergate": operations.gates.F,
        }.get(blackbird_operation["op"])

        params = self._get_operation_params(
            pq_operation_class=pq_operation_class, bb_operation=blackbird_operation
        )

        operation = pq_operation_class(**params)

        operation.modes = blackbird_operation["modes"]

        return operation

    @staticmethod
    def _get_operation_params(pq_operation_class, bb_operation):
        bb_params = bb_operation.get("args", None)

        if bb_params is None:
            return {}

        parameters = inspect.signature(pq_operation_class).parameters

        operation_params = collections.OrderedDict()

        for param_name, param in parameters.items():
            if param_name == "self":
                continue

            operation_params[param_name] = param.default

        for pq_param_name, bb_param in zip(operation_params.keys(), bb_params):
            operation_params[pq_param_name] = bb_param

        return operation_params

    def from_blackbird(self, bb):
        """
        Loads the gates to apply into `self.operations` from a
        :class:`blackbird.BlackbirdProgram`

        Args:
            bb (blackbird.BlackbirdProgram): the BlackbirdProgram to use
        """
        self.operations = [*map(self._blackbird_operation_to_operation, bb.operations)]

    def load_blackbird(self, filename: str):
        """
        Loads the gates to apply into `self.operations` from a BlackBird file
        (.xbb).

        Args:
            filename (str): file location of a valid Blackbird program
        """
        bb = blackbird.load(filename)
        return self.from_blackbird(bb)

    def loads_blackbird(self, string):
        """
        Loads the gates to apply into `self.operations` from a string
        representing a :class:`blackbird.BlackbirdProgram`.

        Args:
            string (str): string containing a valid Blackbird Program
        """
        bb = blackbird.loads(string)
        return self.from_blackbird(bb)

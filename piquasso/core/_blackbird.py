#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import inspect
from collections import OrderedDict

from . import registry


def load_operations(blackbird_program):
    """
    Loads the gates to apply into :attr:`Program.operations` from a
    :class:`blackbird.BlackbirdProgram`

    Args:
        blackbird_program (blackbird.BlackbirdProgram): The BlackbirdProgram to use.
    """

    operation_map = {
        "Dgate": registry._retrieve_class("D"),
        "Xgate": registry._retrieve_class("X"),
        "Zgate": registry._retrieve_class("Z"),
        "Sgate": registry._retrieve_class("S"),
        "Pgate": registry._retrieve_class("P"),
        "Vgate": None,
        "Kgate": registry._retrieve_class("K"),
        "Rgate": registry._retrieve_class("R"),
        "BSgate": registry._retrieve_class("B"),
        "MZgate": registry._retrieve_class("MZ"),
        "S2gate": registry._retrieve_class("S2"),
        "CXgate": registry._retrieve_class("CX"),
        "CZgate": registry._retrieve_class("CZ"),
        "CKgate": registry._retrieve_class("CK"),
        "Fouriergate": registry._retrieve_class("F"),
    }

    return [
        _blackbird_operation_to_operation(operation_map, operation)
        for operation in blackbird_program.operations
    ]


def _blackbird_operation_to_operation(operation_map, blackbird_operation):
    pq_operation_class = operation_map.get(blackbird_operation["op"])

    params = _get_operation_params(
        pq_operation_class=pq_operation_class, bb_operation=blackbird_operation
    )

    operation = pq_operation_class(**params)

    operation.modes = tuple(blackbird_operation["modes"])

    return operation


def _get_operation_params(pq_operation_class, bb_operation):
    bb_params = bb_operation.get("args", None)

    if bb_params is None:
        return {}

    parameters = inspect.signature(pq_operation_class).parameters

    operation_params = OrderedDict()

    for param_name, param in parameters.items():
        if param_name == "self":
            continue

        operation_params[param_name] = param.default

    for pq_param_name, bb_param in zip(operation_params.keys(), bb_params):
        operation_params[pq_param_name] = bb_param

    return operation_params

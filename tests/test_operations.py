#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import piquasso as pq

from piquasso.api.operation import Operation
from piquasso.api.errors import InvalidParameter


def test_operation_initialization_from_properties():
    properties = {
        "params": {
            "first_param": "first_param_value",
            "second_param": "second_param_value"
        },
        "modes": ["some", "modes"],
    }

    class DummyOperation(Operation):
        def __init__(self, first_param, second_param):
            super().__init__(first_param=first_param, second_param=second_param)

    operation = DummyOperation.from_properties(properties)

    assert operation.params == {
        "first_param": "first_param_value",
        "second_param": "second_param_value"
    }
    assert operation.modes == ["some", "modes"]


def test_displacement_raises_InvalidParameter_for_redundant_parameters():
    with pytest.raises(InvalidParameter):
        pq.Displacement(alpha=1, r=2, phi=3)

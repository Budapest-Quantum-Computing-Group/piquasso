#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.api.operation import Operation


class TestOperation:
    def test_operation_initialization_from_properties(self):
        properties = {
            "params": {
                "first_param": "first_param_value",
                "second_param": "second_param_value"
            },
            "modes": ["some", "modes"],
        }

        class DummyOperation(Operation):
            def __init__(self, first_param, second_param):
                super().__init__(first_param, second_param)

        operation = DummyOperation.from_properties(properties)

        assert operation.params == ("first_param_value", "second_param_value")
        assert operation.modes == ["some", "modes"]

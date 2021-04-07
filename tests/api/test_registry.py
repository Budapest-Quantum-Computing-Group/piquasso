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

import pytest

from piquasso.core.registry import (
    _register,
    _retrieve_class,
    _create_instance_from_mapping,
)


class TestRegistry:

    @pytest.fixture
    def SomeClass(self):

        @_register
        class SomeClass:
            def __init__(self, foo, bar):
                self.foo = foo
                self.bar = bar

            @classmethod
            def from_properties(cls, properties):
                return cls(**properties)

        return SomeClass

    @pytest.fixture
    def SomeSubClass(self, SomeClass):

        @_register
        class SomeSubClass(SomeClass):
            pass

        return SomeSubClass

    @pytest.fixture
    def SomeSubSubClass(self, SomeSubClass):

        @_register
        class SomeSubSubClass(SomeSubClass):
            pass

        return SomeSubSubClass

    def test_retrieving_existing_subclasses(self, SomeSubClass, SomeSubSubClass):

        assert _retrieve_class("SomeSubClass") == SomeSubClass
        assert _retrieve_class("SomeSubSubClass") == SomeSubSubClass

    def test_retrieving_nonexistent_class_raises_NameError(self):
        with pytest.raises(NameError):
            _retrieve_class("nonexistent_class")

    def test_creating_instance_from_properties(self, SomeClass):
        instance = _create_instance_from_mapping(
            {
                "type": "SomeClass",
                "properties": {
                    "foo": "fee",
                    "bar": "beer",
                },
            }
        )

        assert instance.__class__.__name__ == "SomeClass"

        assert instance.foo == "fee"
        assert instance.bar == "beer"

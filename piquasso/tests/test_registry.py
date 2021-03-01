#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

from piquasso import registry


class TestRegistry:

    @pytest.fixture
    def SomeClass(self):

        @registry.add
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

        @registry.add
        class SomeSubClass(SomeClass):
            pass

        return SomeSubClass

    @pytest.fixture
    def SomeSubSubClass(self, SomeSubClass):

        @registry.add
        class SomeSubSubClass(SomeSubClass):
            pass

        return SomeSubSubClass

    def test_retrieving_existing_subclasses(self, SomeSubClass, SomeSubSubClass):

        assert registry.retrieve_class("SomeSubClass") == SomeSubClass
        assert registry.retrieve_class("SomeSubSubClass") == SomeSubSubClass

    def test_retrieving_nonexistent_class_raises_NameError(self):
        with pytest.raises(NameError):
            registry.retrieve_class("nonexistent_class")

    def test_creating_instance_from_properties(self, SomeClass):
        instance = registry.create_instance_from_mapping(
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

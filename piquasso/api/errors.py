#
# Copyright (C) 2020 by TODO - All rights reserved.
#

class InvalidState(Exception):
    """Raised when an invalid state is encountered or being prepared."""


class InvalidParameter(Exception):
    """Raised when an invalid parameter is specified."""


class InvalidModes(Exception):
    """Raised when invalid set of modes are encountered."""

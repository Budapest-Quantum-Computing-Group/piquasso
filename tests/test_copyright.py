#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

import os
import glob

from itertools import islice
from pathlib import Path
from datetime import date


COPYRIGHT = f"""\
#
# Copyright 2021-{date.today().year} Budapest Quantum Computing Group
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
"""

COPYRIGHT_CPP = f"""\
/*
 * Copyright 2021-{date.today().year} Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""

COPYRIGHT_MAP = {
    "*.py": COPYRIGHT,
    "*.cpp": COPYRIGHT_CPP,
    "*.hpp": COPYRIGHT_CPP,
}


def test_copyrights():
    root = Path(__file__).parents[1]

    exceptions = [
        root / ".venv",
        root / "docs",
        root / "dist",
        root / "build",
    ]

    extensions = COPYRIGHT_MAP.keys()

    for extension in extensions:
        copyright_text = COPYRIGHT_MAP[extension]
        for filename in glob.iglob(str(root / "**" / extension), recursive=True):
            if (
                _is_current_file(filename)
                or _is_empty_file(filename)
                or _in_exceptions(filename, exceptions)
            ):
                continue

            with open(filename, "r+") as file:
                first_few_lines = "".join(
                    islice(file.readlines(), copyright_text.count("\n"))
                )

            assert (
                copyright_text in first_few_lines
            ), f"Invalid or no copyright found in {filename}"


def _is_current_file(filename):
    return filename == os.path.abspath(__file__)


def _is_empty_file(filename):
    return os.stat(filename).st_size == 0


def _in_exceptions(filename, exceptions):
    for exception in exceptions:
        if Path(exception) in Path(filename).parents:
            return True

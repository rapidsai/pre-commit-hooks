# Copyright (c) 2024, NVIDIA CORPORATION.
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

import io
from unittest.mock import patch

import pytest

from rapids_pre_commit_hooks.lint import Linter
from rapids_pre_commit_hooks.rapids_metadata import check_rapids_metadata


class TestRAPIDSMetadata:
    def mock_files(self, files):
        def new_open(filename, mode="r"):
            assert mode == "r"

            try:
                content = files[filename]
            except KeyError:
                raise FileNotFoundError

            return io.StringIO(content)

        return patch("builtins.open", new_open)

    def test_mock_files(self):
        with self.mock_files({"file.txt": "Hello"}):
            with open("file.txt") as f:
                assert f.read() == "Hello"
            with pytest.raises(FileNotFoundError):
                open("nonexistent.txt")
            with pytest.raises(AssertionError):
                open("file.txt", "rb")
            with pytest.raises(AssertionError):
                open("file.txt", "w")

    def test_template_file(self):
        FILES = {
            "VERSION": "24.04.00\n",
            "file.txt.rapids_metadata_template": """This file contains RAPIDS metadata
Full version is {RAPIDS_VERSION}
Major-minor version is {RAPIDS_VERSION_MAJOR_MINOR}
Major version is {RAPIDS_VERSION_MAJOR}
Minor version is {RAPIDS_VERSION_MINOR}
Patch version is {RAPIDS_VERSION_PATCH}
""",
        }

        CORRECT_CONTENT = CONTENT = """This file contains RAPIDS metadata
Full version is 24.04.00
Major-minor version is 24.04
Major version is 24
Minor version is 04
Patch version is 00
"""

        linter = Linter("file.txt", CONTENT)
        with self.mock_files(FILES):
            checker = check_rapids_metadata()
            checker(linter, None)
        assert linter.warnings == []

        CONTENT = """This file contains RAPIDS metadata
Full version is 24.02.00
Major-minor version is 24.02
Major version is 24
Minor version is 02
Patch version is 00
"""
        expected_linter = Linter("file.txt", CONTENT)
        expected_linter.add_warning(
            (0, len(CONTENT)),
            "file does not match template replacement from "
            '"file.txt.rapids_metadata_template"',
        ).add_replacement((0, len(CONTENT)), CORRECT_CONTENT)

        linter = Linter("file.txt", CONTENT)
        with self.mock_files(FILES):
            checker = check_rapids_metadata()
            checker(linter, None)
        assert linter.warnings == expected_linter.warnings

    def test_no_template_file(self):
        FILES = {
            "VERSION": "24.04.00\n",
        }

        CONTENT = """This file contains RAPIDS metadata
Full version is 24.04.00
Major-minor version is 24.04
Major version is 24
Minor version is 04
Patch version is 00
"""
        expected_linter = Linter("file.txt", CONTENT)
        expected_linter.add_warning(
            (51, 59),
            "do not hard-code RAPIDS version; dynamically read from VERSION file or "
            'write a "file.txt.rapids_metadata_template" file',
        )
        expected_linter.add_warning(
            (83, 88),
            "do not hard-code RAPIDS version; dynamically read from VERSION file or "
            'write a "file.txt.rapids_metadata_template" file',
        )

        linter = Linter("file.txt", CONTENT)
        with self.mock_files(FILES):
            checker = check_rapids_metadata()
            checker(linter, None)
        assert linter.warnings == expected_linter.warnings

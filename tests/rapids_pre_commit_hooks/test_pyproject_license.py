# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

from textwrap import dedent
from unittest.mock import Mock

import pytest
import tomlkit

from rapids_pre_commit_hooks import pyproject_license
from rapids_pre_commit_hooks.lint import Linter


@pytest.mark.parametrize(
    ["key", "append", "loc"],
    [
        (
            ("table", "key1"),
            False,
            (15, 22),
        ),
        (
            ("table", "key2"),
            False,
            (30, 32),
        ),
        (
            ("table", "key3"),
            False,
            (40, 60),
        ),
        (
            ("table", "key3", "nested"),
            False,
            (51, 58),
        ),
        (
            ("table",),
            True,
            (61, 61),
        ),
    ],
)
def test_find_value_location(key, append, loc):
    CONTENT = dedent(
        """\
        [table]
        key1 = "value"
        key2 = 42
        key3 = { nested = "value" }

        [table2]
        key = "value"
        """
    )
    parsed_doc = tomlkit.loads(CONTENT)
    assert (
        pyproject_license.find_value_location(parsed_doc, key, append) == loc
    )
    assert parsed_doc.as_string() == CONTENT


@pytest.mark.parametrize(
    ["document", "loc", "message", "replacement_loc", "replacement_text"],
    [
        (
            dedent(
                """\
                [project]
                license = { text = "Apache-2.0" }
                """
            ),
            (29, 41),
            'license should be "Apache 2.0"',
            None,
            None,
        ),
        (
            dedent(
                """\
                [project]
                license = { text = "BSD" }
                """
            ),
            (29, 34),
            'license should be "Apache 2.0"',
            None,
            None,
        ),
        *(
            (
                dedent(
                    f"""\
                    [project]
                    license = {{ text = {
                        tomlkit.string(license).as_string()
                    } }}
                    """
                ),
                None,
                None,
                None,
                None,
            )
            for license in pyproject_license.ACCEPTABLE_LICENSES
        ),
        (
            dedent(
                """\
                [project]
                license = { text = 'Apache 2.0' }  # Single quotes are fine
                """
            ),
            None,
            None,
            None,
            None,
        ),
        (
            dedent(
                """\
                [build-system]
                requires = ["scikit-build-core"]
                """
            ),
            (48, 48),
            'add project.license with value { text = "Apache 2.0" }',
            (48, 48),
            '[project]\nlicense = { text = "Apache 2.0" }\n',
        ),
        (
            dedent(
                """\
                [project]
                name = "test-project"

                [build-system]
                requires = ["scikit-build-core"]
                """
            ),
            (32, 32),
            'add project.license with value { text = "Apache 2.0" }',
            (32, 32),
            'license = { text = "Apache 2.0" }\n',
        ),
        (
            dedent(
                """\
                [project]
                name = "test-project"

                [project.optional-dependencies]
                test = ["pytest"]
                """
            ),
            (32, 32),
            'add project.license with value { text = "Apache 2.0" }',
            (32, 32),
            'license = { text = "Apache 2.0" }\n',
        ),
        (
            dedent(
                """\
                [project.optional-dependencies]
                test = ["pytest"]
                """
            ),
            (50, 50),
            'add project.license with value { text = "Apache 2.0" }',
            (50, 50),
            '[project]\nlicense = { text = "Apache 2.0" }\n',
        ),
    ],
)
def test_check_pyproject_license(
    document,
    loc,
    message,
    replacement_loc,
    replacement_text,
):
    linter = Linter("pyproject.toml", document)
    pyproject_license.check_pyproject_license(linter, Mock())

    expected_linter = Linter("pyproject.toml", document)
    if loc and message:
        w = expected_linter.add_warning(loc, message)
        if replacement_loc and replacement_text:
            w.add_replacement(replacement_loc, replacement_text)
    assert linter.warnings == expected_linter.warnings

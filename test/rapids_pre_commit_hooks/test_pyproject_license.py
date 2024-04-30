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

from textwrap import dedent

import pytest
import tomlkit

from rapids_pre_commit_hooks import pyproject_license
from rapids_pre_commit_hooks.lint import Linter


@pytest.mark.parametrize(
    ["key", "loc"],
    [
        (
            ("table", "key1"),
            (15, 22),
        ),
        (
            ("table", "key2"),
            (30, 32),
        ),
        (
            ("table", "key3"),
            (40, 60),
        ),
        (
            ("table", "key3", "nested"),
            (51, 58),
        ),
    ],
)
def test_find_value_location(key, loc):
    CONTENT = dedent(
        """\
        [table]
        key1 = "value"
        key2 = 42
        key3 = { nested = "value" }
        """
    )
    parsed_doc = tomlkit.loads(CONTENT)
    assert pyproject_license.find_value_location(parsed_doc, key) == loc
    assert parsed_doc.as_string() == CONTENT


@pytest.mark.parametrize(
    ["document", "loc"],
    [
        (
            dedent(
                """\
                [project]
                license = "Apache-2.0"
                """
            ),
            (20, 32),
        ),
        (
            dedent(
                """\
                [project]
                license = "BSD"
                """
            ),
            (20, 25),
        ),
        (
            dedent(
                """\
                [project]
                license = "Apache 2.0"
                """
            ),
            None,
        ),
        (
            dedent(
                """\
                [project]
                license = 'Apache 2.0'  # Single quotes are fine
                """
            ),
            None,
        ),
        ("", None),
    ],
)
def test_check_pyproject_license(document, loc):
    linter = Linter("pyproject.toml", document)
    pyproject_license.check_pyproject_license(linter, None)

    expected_linter = Linter("pyproject.toml", document)
    if loc:
        expected_linter.add_warning(
            loc, 'license should be "Apache 2.0"'
        ).add_replacement(loc, '"Apache 2.0"')
    assert linter.warnings == expected_linter.warnings

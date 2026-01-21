# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import tomlkit

from rapids_pre_commit_hooks import pyproject_license
from rapids_pre_commit_hooks.lint import LintWarning, Linter, Replacement
from rapids_pre_commit_hooks_test_utils import parse_named_ranges


@pytest.mark.parametrize(
    ["key", "append"],
    [
        pytest.param(
            ("table", "key1"),
            False,
            id="string-value",
        ),
        pytest.param(
            ("table", "key2"),
            False,
            id="int-value",
        ),
        pytest.param(
            ("table", "key3"),
            False,
            id="subtable",
        ),
        pytest.param(
            ("table", "key3", "nested"),
            False,
            id="subtable-nested",
        ),
        pytest.param(
            ("table",),
            True,
            id="append",
        ),
    ],
)
def test_find_value_location(key, append):
    content, r = parse_named_ranges(
        """\
        + [table]
        + key1 = "value"
        :        ~~~~~~~table.key1._value
        + key2 = 42
        :        ~~table.key2._value
        + key3 = { nested = "value" }
        :        ~~~~~~~~~~~~~~~~~~~~table.key3._value
        :                   ~~~~~~~table.key3.nested._value
        +
        : ^table._append
        + [table2]
        + key = "value"
        """
    )
    parsed_doc = tomlkit.loads(content)
    loc = r
    for component in key:
        loc = loc[component]
    loc = loc["_append" if append else "_value"]
    assert (
        pyproject_license.find_value_location(parsed_doc, key, append) == loc
    )
    assert parsed_doc.as_string() == content


@pytest.mark.parametrize(
    ["content", "message", "replacement_text"],
    [
        # unrecognized license in "= { text = ... }" format
        # that only differs from an acceptable one
        # by internal whitespace should cause a warning
        pytest.param(
            """\
            + [project]
            + license = { text = "Apache-2.0" }
            :                    ~~~~~~~~~~~~warning
            """,
            'license should be "Apache 2.0"',
            None,
            id="license-subtable-with-text-wrong-license-by-whitespace-only",
        ),
        # unrecognized license in "= { text = ... }" format should result
        # in a warning
        pytest.param(
            """\
            + [project]
            + license = { text = "BSD" }
            :                    ~~~~~warning
            """,
            'license should be "Apache 2.0"',
            None,
            id="license-subtable-with-text-wrong-license",
        ),
        # each of the acceptable licenses, expressed in PEP 639 format,
        # should not generate any warnings or replacements
        *(
            pytest.param(
                f"""\
                + [project]
                + license = {{ text = {tomlkit.string(license).as_string()} }}
                """,
                None,
                None,
                id=f"license-correct-{license}",
            )
            for license in pyproject_license.ACCEPTABLE_LICENSES
        ),
        # an acceptable license in single quotes
        # should not generate any warnings or replacements
        pytest.param(
            """\
            + [project]
            + license = { text = 'Apache 2.0' }  # Single quotes are fine
            """,
            None,
            None,
            id="license-correct-single-quotes",
        ),
        # Apache 2.0 license should be added to a file
        # totally missing [project] table
        pytest.param(
            """\
            + [build-system]
            + requires = ["scikit-build-core"]
            :                                  ^warning
            :                                  ^replacement
            """,
            'add project.license with value { text = "Apache 2.0" }',
            '[project]\nlicense = { text = "Apache 2.0" }\n',
            id="no-project-table",
        ),
        # Apache 2.0 license should be added to a file with [project] table
        # but no 'license' key
        pytest.param(
            """\
            + [project]
            + name = "test-project"
            +
            : ^warning
            : ^replacement
            + [build-system]
            + requires = ["scikit-build-core"]
            """,
            'add project.license with value { text = "Apache 2.0" }',
            'license = { text = "Apache 2.0" }\n',
            id="project-table-no-license-key",
        ),
        # Apache 2.0 license should be correctly added to a file with
        # [project] table and other [project.*] tables
        pytest.param(
            """\
            + [project]
            + name = "test-project"
            +
            : ^warning
            : ^replacement
            + [project.optional-dependencies]
            + test = ["pytest"]
            """,
            'add project.license with value { text = "Apache 2.0" }',
            'license = { text = "Apache 2.0" }\n',
            id="project-table-and-project-subtables",
        ),
        # Apache 2.0 licenses should be correctly added to a file with
        # [project.*] tables but no [project] table
        pytest.param(
            """\
            + [project.optional-dependencies]
            + test = ["pytest"]
            :                   ^warning
            :                   ^replacement
            """,
            'add project.license with value { text = "Apache 2.0" }',
            '[project]\nlicense = { text = "Apache 2.0" }\n',
            id="project-subtable-no-project-table",
        ),
    ],
)
def test_check_pyproject_license(
    content,
    message,
    replacement_text,
):
    content, positions = parse_named_ranges(content)
    linter = Linter("pyproject.toml", content, "verify-pyproject-license")
    pyproject_license.check_pyproject_license(linter, Mock())

    assert linter.warnings == (
        []
        if message is None
        else [
            LintWarning(
                positions["warning"],
                message,
                replacements=[]
                if replacement_text is None
                else [Replacement(positions["replacement"], replacement_text)],
            )
        ]
    )

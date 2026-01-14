# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
        # recognized license in "= { text = ... }" form should result
        # in a warning
        (
            dedent(
                """\
                [project]
                license = { text = "Apache-2.0" }
                """
            ),
            (20, 43),
            'license should be "Apache-2.0"',
            None,
            None,
        ),
        # unrecognized license in "= { text = ... }" should result
        # in a warning
        (
            dedent(
                """\
                [project]
                license = { text = "BSD" }
                """
            ),
            (20, 36),
            'license should be "Apache-2.0"',
            None,
            None,
        ),
        # each of the acceptable licenses, expressed in PEP 639 format,
        # should not generate any warnings or replacements
        *(
            (
                dedent(
                    f"""\
                    [project]
                    license = {tomlkit.string(license).as_string()}
                    """
                ),
                None,
                None,
                None,
                None,
            )
            for license in pyproject_license.ACCEPTABLE_LICENSES
        ),
        # an acceptable license in single quotes, expressed in PEP 639 form,
        # should not generate any warnings or replacements
        (
            dedent(
                """\
                [project]
                license = 'Apache-2.0'  # Single quotes are fine
                """
            ),
            None,
            None,
            None,
            None,
        ),
        # a license in PEP 639 form that only differs from an acceptable one
        # by internal whitespace should cause a warning and a replacement
        (
            dedent(
                """\
                [project]
                license = 'Apache 2.0'  # Single quotes are fine
                """
            ),
            (20, 32),
            'license should be "Apache-2.0", got "Apache 2.0"',
            (20, 32),
            'license = "Apache-2.0"\n',
        ),
        # a license in PEP 639 form that only differs from an acceptable one
        # by leading whitespace (including multiple characters) should cause
        # a warning and a replacement
        (
            dedent(
                """\
                [project]
                license = '   Apache-2.0'  # Single quotes are fine
                """
            ),
            (20, 35),
            'license should be "Apache-2.0", got "   Apache-2.0"',
            (20, 35),
            'license = "Apache-2.0"\n',
        ),
        # a license in PEP 639 form that only differs from an acceptable one
        # by trailing whitespace (including multiple characters) should cause
        # a warning and a replacement
        (
            dedent(
                """\
                [project]
                license = 'Apache-2.0   '  # Single quotes are fine
                """
            ),
            (20, 35),
            'license should be "Apache-2.0", got "Apache-2.0   "',
            (20, 35),
            'license = "Apache-2.0"\n',
        ),
        # Apache-2.0 licenses should be added to a file
        # totally missing [project] table
        (
            dedent(
                """\
                [build-system]
                requires = ["scikit-build-core"]
                """
            ),
            (48, 48),
            'add project.license with value "Apache-2.0"',
            (48, 48),
            '[project]\nlicense = "Apache-2.0"\n',
        ),
        # Apache-2.0 licenses should be added to a file with [project] table
        # but no 'license' key
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
            'add project.license with value "Apache-2.0"',
            (32, 32),
            'license = "Apache-2.0"\n',
        ),
        # Apache-2.0 licenses should be correctly added to a file with
        # [project] table and other [project.*] tables
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
            'add project.license with value "Apache-2.0"',
            (32, 32),
            'license = "Apache-2.0"\n',
        ),
        # Apache-2.0 licenses should be correctly added to a file with
        # [project.*] tables but no [project] table
        (
            dedent(
                """\
                [project.optional-dependencies]
                test = ["pytest"]
                """
            ),
            (50, 50),
            'add project.license with value "Apache-2.0"',
            (50, 50),
            '[project]\nlicense = "Apache-2.0"\n',
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
    linter = Linter("pyproject.toml", document, "verify-pyproject-license")
    pyproject_license.check_pyproject_license(linter, Mock())

    expected_linter = Linter(
        "pyproject.toml", document, "verify-pyproject-license"
    )
    if loc and message:
        w = expected_linter.add_warning(loc, message)
        if replacement_loc and replacement_text:
            w.add_replacement(replacement_loc, replacement_text)
    assert linter.warnings == expected_linter.warnings

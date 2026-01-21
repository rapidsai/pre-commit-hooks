# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from unittest.mock import Mock, patch

import pytest

from rapids_pre_commit_hooks import hardcoded_version
from rapids_pre_commit_hooks.lint import LintWarning, Linter
from rapids_pre_commit_hooks_test_utils import parse_named_ranges


@pytest.mark.parametrize(
    ["content", "version"],
    [
        pytest.param(
            """\
            > 26.02
            : ~~~~~0.full
            : ~~0.major
            :    ~~0.minor
            """,
            (26, 2, 0),
            id="no-patch-version",
        ),
        pytest.param(
            """\
            > a26.02
            :  ~~~~~0.full
            :  ~~0.major
            :     ~~0.minor
            """,
            (26, 2, 0),
            id="text-before",
        ),
        pytest.param(
            """\
            > 26.02a
            : ~~~~~0.full
            : ~~0.major
            :    ~~0.minor
            """,
            (26, 2, 0),
            id="text-after",
        ),
        pytest.param(
            """\
            > a26.02a
            :  ~~~~~0.full
            :  ~~0.major
            :     ~~0.minor
            """,
            (26, 2, 0),
            id="text-before-and-after",
        ),
        pytest.param(
            """\
            + 26.02
            : ~~~~~0.full
            : ~~0.major
            :    ~~0.minor
            > 26.02
            : ~~~~~1.full
            : ~~1.major
            :    ~~1.minor
            """,
            (26, 2, 0),
            id="multiple-instances",
        ),
        pytest.param(
            """\
            > 26.02.00
            : ~~~~~~~~0.full
            : ~~0.major
            :    ~~0.minor
            :       ~~0.patch
            """,
            (26, 2, 0),
            id="patch-version",
        ),
        pytest.param(
            """\
            > 26.02.01
            """,
            (26, 2, 0),
            id="wrong-patch-version",
        ),
        pytest.param(
            """\
            > 26.04
            """,
            (26, 2, 0),
            id="wrong-major-minor-version",
        ),
        pytest.param(
            """\
            > 0.48
            : ~~~~0.full
            : ~0.major
            :   ~~0.minor
            """,
            (0, 48, 0),
            id="ucxx-version",
        ),
        pytest.param(
            """\
            > 0.48.00
            : ~~~~~~~0.full
            : ~0.major
            :   ~~0.minor
            :      ~~0.patch
            """,
            (0, 48, 0),
            id="ucxx-patch-version",
        ),
        pytest.param(
            """\
            > 026.02
            """,
            (26, 2, 0),
            id="number-before",
        ),
        pytest.param(
            """\
            > 26.020
            """,
            (26, 2, 0),
            id="number-after",
        ),
        pytest.param(
            """\
            > 26.2.0
            : ~~~~~~0.full
            : ~~0.major
            :    ~0.minor
            :      ~0.patch
            """,
            (26, 2, 0),
            id="no-zero-prefix",
        ),
        pytest.param(
            """\
            > 2026.02.00
            """,
            (26, 2, 0),
            id="4-digit-major",
        ),
        pytest.param(
            """\
            > 26.0002.00
            """,
            (26, 2, 0),
            id="4-digit-minor",
        ),
        pytest.param(
            """\
            > 26.02.0000
            : ~~~~~0.full
            : ~~0.major
            :    ~~0.minor
            """,
            (26, 2, 0),
            id="4-digit-patch",
        ),
    ],
)
def test_find_hardcoded_versions(content, version):
    content, r = parse_named_ranges(content, list)
    assert [
        {group: match.span(group) for group in match.groupdict().keys()}
        for match in hardcoded_version.find_hardcoded_versions(
            content, version
        )
    ] == [{"patch": (-1, -1), **m} for m in r]


@pytest.mark.parametrize(
    ["content", "version", "context"],
    [
        pytest.param(
            "26.02.00\n",
            (26, 2, 0),
            contextlib.nullcontext(),
            id="valid-rapids-version",
        ),
        pytest.param(
            "0.48.00\n",
            (0, 48, 0),
            contextlib.nullcontext(),
            id="valid-ucxx-version",
        ),
        pytest.param(
            "26.02.00",
            None,
            pytest.raises(
                AssertionError,
                match=r'^Expected file ".*/VERSION" to contain ONLY a 3-part '
                r"numeric version, but additional content was found, or no "
                r"trailing newline was found$",
            ),
            id="missing-newline",
        ),
        pytest.param(
            "26.02\n",
            None,
            pytest.raises(
                AssertionError,
                match=r'^Expected file ".*/VERSION" to contain a 3-part '
                r"numeric version, but the patch \(3rd\) part was not found$",
            ),
            id="missing-patch",
        ),
        pytest.param(
            "",
            None,
            pytest.raises(
                AssertionError,
                match=r'Expected file ".*/VERSION" to contain a 3-part '
                r"numeric version, but it was not found$",
            ),
            id="not-version",
        ),
        pytest.param(
            None,
            None,
            pytest.raises(FileNotFoundError),
            id="file-missing",
        ),
    ],
)
def test_read_version_file(tmp_path, content, version, context):
    filename = tmp_path / "VERSION"
    if content is not None:
        with open(filename, "w") as f:
            f.write(content)
    with context:
        assert hardcoded_version.read_version_file(filename) == version


@pytest.mark.parametrize(
    [
        "filename",
        "content",
        "version_file",
        "version",
        "version_file_read",
        "message",
    ],
    [
        pytest.param(
            "file.txt",
            """\
            + RAPIDS 26.02
            :        ~~~~~0
            """,
            "VERSION",
            (26, 2, 0),
            True,
            "do not hard-code version, read from VERSION file instead",
            id="version-file",
        ),
        pytest.param(
            "file.txt",
            """\
            + RAPIDS 26.02.00
            :        ~~~~~~~~0
            """,
            "VERSION",
            (26, 2, 0),
            True,
            "do not hard-code version, read from VERSION file instead",
            id="version-file-patch-version",
        ),
        pytest.param(
            "file.txt",
            """\
            + RAPIDS 26.02
            :        ~~~~~0
            + RAPIDS 26.02.00
            :        ~~~~~~~~1
            """,
            "VERSION",
            (26, 2, 0),
            True,
            "do not hard-code version, read from VERSION file instead",
            id="version-file-multiple",
        ),
        pytest.param(
            "file.txt",
            """\
            + RAPIDS 26.02
            :        ~~~~~0
            """,
            "RAPIDS_VERSION",
            (26, 2, 0),
            True,
            "do not hard-code version, read from RAPIDS_VERSION file instead",
            id="rapids-version-file",
        ),
        pytest.param(
            "file.txt",
            """\
            + RAPIDS 26.04
            """,
            "VERSION",
            (26, 2, 0),
            True,
            "do not hard-code version, read from VERSION file instead",
            id="version-not-found",
        ),
        pytest.param(
            "VERSION",
            """\
            + 26.02.00
            """,
            "VERSION",
            (26, 2, 0),
            False,
            "do not hard-code version, read from VERSION file instead",
            id="skip-version-file",
        ),
    ],
)
def test_check_hardcoded_version(
    filename,
    content,
    version_file,
    version,
    version_file_read,
    message,
):
    content, r = parse_named_ranges(content, list)
    linter = Linter(filename, content, "verify-hardcoded-version")
    with patch(
        "rapids_pre_commit_hooks.hardcoded_version.read_version_file",
        Mock(return_value=version),
    ) as mock_read_version_file:
        hardcoded_version.check_hardcoded_version(
            linter, Mock(version_file=version_file)
        )
    if version_file_read:
        mock_read_version_file.assert_called_once_with(version_file)
    else:
        mock_read_version_file.assert_not_called()
    assert linter.warnings == [LintWarning(m, message) for m in r]

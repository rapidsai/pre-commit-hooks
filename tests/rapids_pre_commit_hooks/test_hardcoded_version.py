# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from unittest.mock import Mock, patch

import pytest

from rapids_pre_commit_hooks import hardcoded_version
from rapids_pre_commit_hooks.lint import LintWarning, Linter


@pytest.mark.parametrize(
    ["content", "version", "matches"],
    [
        pytest.param(
            "26.02",
            (26, 2, 0),
            [
                {
                    "full": (0, 5),
                    "major": (0, 2),
                    "minor": (3, 5),
                    "patch": (-1, -1),
                },
            ],
            id="full-contents",
        ),
        pytest.param(
            "a26.02",
            (26, 2, 0),
            [
                {
                    "full": (1, 6),
                    "major": (1, 3),
                    "minor": (4, 6),
                    "patch": (-1, -1),
                },
            ],
            id="text-before",
        ),
        pytest.param(
            "26.02a",
            (26, 2, 0),
            [
                {
                    "full": (0, 5),
                    "major": (0, 2),
                    "minor": (3, 5),
                    "patch": (-1, -1),
                },
            ],
            id="text-after",
        ),
        pytest.param(
            "a26.02a",
            (26, 2, 0),
            [
                {
                    "full": (1, 6),
                    "major": (1, 3),
                    "minor": (4, 6),
                    "patch": (-1, -1),
                },
            ],
            id="text-before-and-after",
        ),
        pytest.param(
            "26.02\n26.02",
            (26, 2, 0),
            [
                {
                    "full": (0, 5),
                    "major": (0, 2),
                    "minor": (3, 5),
                    "patch": (-1, -1),
                },
                {
                    "full": (6, 11),
                    "major": (6, 8),
                    "minor": (9, 11),
                    "patch": (-1, -1),
                },
            ],
            id="multiple-instances",
        ),
        pytest.param(
            "26.02.00",
            (26, 2, 0),
            [
                {
                    "full": (0, 8),
                    "major": (0, 2),
                    "minor": (3, 5),
                    "patch": (6, 8),
                },
            ],
            id="patch-version",
        ),
        pytest.param(
            "26.02.01",
            (26, 2, 0),
            [],
            id="wrong-patch-version",
        ),
        pytest.param(
            "26.04",
            (26, 2, 0),
            [],
            id="wrong-major-minor-version",
        ),
        pytest.param(
            "0.48",
            (0, 48, 0),
            [
                {
                    "full": (0, 4),
                    "major": (0, 1),
                    "minor": (2, 4),
                    "patch": (-1, -1),
                },
            ],
            id="ucxx-version",
        ),
        pytest.param(
            "0.48.00",
            (0, 48, 0),
            [
                {
                    "full": (0, 7),
                    "major": (0, 1),
                    "minor": (2, 4),
                    "patch": (5, 7),
                },
            ],
            id="ucxx-patch-version",
        ),
        pytest.param(
            "026.02",
            (26, 2, 0),
            [],
            id="number-before",
        ),
        pytest.param(
            "26.020",
            (26, 2, 0),
            [],
            id="number-after",
        ),
        pytest.param(
            "26.2.0",
            (26, 2, 0),
            [
                {
                    "full": (0, 6),
                    "major": (0, 2),
                    "minor": (3, 4),
                    "patch": (5, 6),
                },
            ],
            id="no-zero-prefix",
        ),
    ],
)
def test_find_hardcoded_versions(content, version, matches):
    assert [
        {group: match.span(group) for group in match.groupdict().keys()}
        for match in hardcoded_version.find_hardcoded_versions(
            content, version
        )
    ] == matches


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
            pytest.raises(AssertionError),
            id="missing-newline",
        ),
        pytest.param(
            "26.02\n",
            None,
            pytest.raises(AssertionError),
            id="missing-patch",
        ),
        pytest.param(
            "",
            None,
            pytest.raises(AssertionError),
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
        "expected_warnings",
    ],
    [
        pytest.param(
            "file.txt",
            "RAPIDS 26.02\n",
            "VERSION",
            (26, 2, 0),
            True,
            [
                LintWarning(
                    (7, 12),
                    "do not hard-code version, read from VERSION file instead",
                ),
            ],
            id="version-file",
        ),
        pytest.param(
            "file.txt",
            "RAPIDS 26.02.00\n",
            "VERSION",
            (26, 2, 0),
            True,
            [
                LintWarning(
                    (7, 15),
                    "do not hard-code version, read from VERSION file instead",
                ),
            ],
            id="version-file-patch-version",
        ),
        pytest.param(
            "file.txt",
            "RAPIDS 26.02\n",
            "RAPIDS_VERSION",
            (26, 2, 0),
            True,
            [
                LintWarning(
                    (7, 12),
                    "do not hard-code version, read from RAPIDS_VERSION file "
                    "instead",
                ),
            ],
            id="rapids-version-file",
        ),
        pytest.param(
            "file.txt",
            "RAPIDS 26.04\n",
            "VERSION",
            (26, 2, 0),
            True,
            [],
            id="version-not-found",
        ),
        pytest.param(
            "VERSION",
            "26.02.00\n",
            "VERSION",
            (26, 2, 0),
            False,
            [],
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
    expected_warnings,
):
    linter = Linter(filename, content)
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
    assert linter.warnings == expected_warnings

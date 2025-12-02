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
            "26.02.00",
            [
                {
                    "full_version": (0, 5),
                    "major_minor_version": (0, 5),
                    "patch_version": (-1, -1),
                },
            ],
            id="full-contents",
        ),
        pytest.param(
            "a26.02",
            "26.02.00",
            [
                {
                    "full_version": (1, 6),
                    "major_minor_version": (1, 6),
                    "patch_version": (-1, -1),
                },
            ],
            id="text-before",
        ),
        pytest.param(
            "26.02a",
            "26.02.00",
            [
                {
                    "full_version": (0, 5),
                    "major_minor_version": (0, 5),
                    "patch_version": (-1, -1),
                },
            ],
            id="text-after",
        ),
        pytest.param(
            "a26.02a",
            "26.02.00",
            [
                {
                    "full_version": (1, 6),
                    "major_minor_version": (1, 6),
                    "patch_version": (-1, -1),
                },
            ],
            id="text-before-and-after",
        ),
        pytest.param(
            "26.02\n26.02",
            "26.02.00",
            [
                {
                    "full_version": (0, 5),
                    "major_minor_version": (0, 5),
                    "patch_version": (-1, -1),
                },
                {
                    "full_version": (6, 11),
                    "major_minor_version": (6, 11),
                    "patch_version": (-1, -1),
                },
            ],
            id="multiple-instances",
        ),
        pytest.param(
            "26.02.00",
            "26.02.00",
            [
                {
                    "full_version": (0, 8),
                    "major_minor_version": (0, 5),
                    "patch_version": (6, 8),
                },
            ],
            id="patch-version",
        ),
        pytest.param(
            "26.02.01",
            "26.02.00",
            [],
            id="wrong-patch-version",
        ),
        pytest.param(
            "26.04",
            "26.02.00",
            [],
            id="wrong-major-minor-version",
        ),
        pytest.param(
            "0.48",
            "0.48.00",
            [
                {
                    "full_version": (0, 4),
                    "major_minor_version": (0, 4),
                    "patch_version": (-1, -1),
                },
            ],
            id="ucxx-version",
        ),
        pytest.param(
            "0.48.00",
            "0.48.00",
            [
                {
                    "full_version": (0, 7),
                    "major_minor_version": (0, 4),
                    "patch_version": (5, 7),
                },
            ],
            id="ucxx-patch-version",
        ),
        pytest.param(
            "026.02",
            "26.02.00",
            [],
            id="number-before",
        ),
        pytest.param(
            "26.020",
            "26.02.00",
            [],
            id="number-after",
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
            "26.02.00",
            contextlib.nullcontext(),
            id="valid-rapids-version",
        ),
        pytest.param(
            "0.48.00\n",
            "0.48.00",
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
            "26.02.00",
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
            "26.02.00",
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
            "26.02.00",
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
            "26.02.00",
            True,
            [],
            id="version-not-found",
        ),
        pytest.param(
            "VERSION",
            "26.02.00\n",
            "VERSION",
            "26.02.00",
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

# Copyright (c) 2025, NVIDIA CORPORATION.
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
from unittest.mock import Mock, patch

import pytest

from rapids_pre_commit_hooks import codeowners
from rapids_pre_commit_hooks.lint import Linter, LintWarning, Note, Replacement

MOCK_REQUIRED_CODEOWNERS_LINES = [
    codeowners.RequiredCodeownersLine(
        file="CMakeLists.txt",
        owners=[
            codeowners.project_codeowners("cmake"),
        ],
    ),
    codeowners.RequiredCodeownersLine(
        file="pyproject.toml",
        owners=[
            codeowners.hard_coded_codeowners("@rapidsai/ci-codeowners"),
        ],
        allow_extra=True,
        after=["CMakeLists.txt"],
    ),
]

patch_required_codeowners_lines = patch(
    "rapids_pre_commit_hooks.codeowners.required_codeowners_lines",
    lambda _args: MOCK_REQUIRED_CODEOWNERS_LINES,
)


@pytest.mark.parametrize(
    ["line", "skip", "codeowners_line"],
    [
        (
            "filename @owner1 @owner2",
            0,
            codeowners.CodeownersLine(
                file=codeowners.FilePattern(
                    filename="filename",
                    pos=(0, 8),
                ),
                owners=[
                    codeowners.Owner(
                        owner="@owner1",
                        pos=(9, 16),
                        pos_with_leading_whitespace=(8, 16),
                    ),
                    codeowners.Owner(
                        owner="@owner2",
                        pos=(17, 24),
                        pos_with_leading_whitespace=(16, 24),
                    ),
                ],
            ),
        ),
        (
            "filename @owner1 @owner2",
            1,
            codeowners.CodeownersLine(
                file=codeowners.FilePattern(
                    filename="filename",
                    pos=(1, 9),
                ),
                owners=[
                    codeowners.Owner(
                        owner="@owner1",
                        pos=(10, 17),
                        pos_with_leading_whitespace=(9, 17),
                    ),
                    codeowners.Owner(
                        owner="@owner2",
                        pos=(18, 25),
                        pos_with_leading_whitespace=(17, 25),
                    ),
                ],
            ),
        ),
        (
            "filename\t @owner1  @owner2  # Comment",
            0,
            codeowners.CodeownersLine(
                file=codeowners.FilePattern(
                    filename="filename",
                    pos=(0, 8),
                ),
                owners=[
                    codeowners.Owner(
                        owner="@owner1",
                        pos=(10, 17),
                        pos_with_leading_whitespace=(8, 17),
                    ),
                    codeowners.Owner(
                        owner="@owner2",
                        pos=(19, 26),
                        pos_with_leading_whitespace=(17, 26),
                    ),
                ],
            ),
        ),
        (
            "file\\ name @owner\\ 1 @owner\\ 2",
            0,
            codeowners.CodeownersLine(
                file=codeowners.FilePattern(
                    filename="file\\ name",
                    pos=(0, 10),
                ),
                owners=[
                    codeowners.Owner(
                        owner="@owner\\ 1",
                        pos=(11, 20),
                        pos_with_leading_whitespace=(10, 20),
                    ),
                    codeowners.Owner(
                        owner="@owner\\ 2",
                        pos=(21, 30),
                        pos_with_leading_whitespace=(20, 30),
                    ),
                ],
            ),
        ),
        (
            "",
            0,
            None,
        ),
        (
            " # comment",
            0,
            None,
        ),
    ],
)
def test_parse_codeowners_line(line, skip, codeowners_line):
    assert codeowners.parse_codeowners_line(line, skip) == codeowners_line


@pytest.mark.parametrize(
    ["line", "pos", "warnings"],
    [
        (
            "CMakeLists.txt @rapidsai/cudf-cmake-codeowners",
            (0, 14),
            [],
        ),
        (
            "CMakeLists.txt @someone-else  # comment",
            (0, 14),
            [
                LintWarning(
                    pos=(0, 14),
                    msg="file 'CMakeLists.txt' has incorrect owners",
                    replacements=[
                        Replacement(
                            pos=(14, 28),
                            newtext="",
                        ),
                        Replacement(
                            pos=(28, 28),
                            newtext=" @rapidsai/cudf-cmake-codeowners",
                        ),
                    ],
                ),
            ],
        ),
        (
            "CMakeLists.txt @someone-else @rapidsai/cudf-cmake-codeowners",
            (0, 14),
            [
                LintWarning(
                    pos=(0, 14),
                    msg="file 'CMakeLists.txt' has incorrect owners",
                    replacements=[
                        Replacement(
                            pos=(14, 28),
                            newtext="",
                        ),
                    ],
                ),
            ],
        ),
        (
            "pyproject.toml @someone-else @rapidsai/ci-codeowners",
            (0, 14),
            [],
        ),
    ],
)
@patch_required_codeowners_lines
def test_check_codeowners_line(line, pos, warnings):
    codeowners_line = codeowners.parse_codeowners_line(line, 0)
    linter = Linter(".github/CODEOWNERS", line)
    found_files = []
    codeowners.check_codeowners_line(
        linter, Mock(project_prefix="cudf"), codeowners_line, found_files
    )
    assert linter.warnings == warnings
    assert found_files == [
        (line, pos)
        for line in MOCK_REQUIRED_CODEOWNERS_LINES
        if line.file == codeowners_line.file.filename
    ]


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        (
            dedent(
                """
                CMakeLists.txt @rapidsai/cudf-cmake-codeowners
                pyproject.toml @rapidsai/ci-codeowners
                """
            ),
            [],
        ),
        (
            dedent(
                """
                CMakeLists.txt @someone-else
                pyproject.toml @rapidsai/ci-codeowners
                """
            ),
            [
                LintWarning(
                    pos=(1, 15),
                    msg="file 'CMakeLists.txt' has incorrect owners",
                    replacements=[
                        Replacement(
                            pos=(15, 29),
                            newtext="",
                        ),
                        Replacement(
                            pos=(29, 29),
                            newtext=" @rapidsai/cudf-cmake-codeowners",
                        ),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """
                pyproject.toml @rapidsai/ci-codeowners
                """
            ),
            [
                LintWarning(
                    pos=(0, 0),
                    msg="missing required codeowners",
                    replacements=[
                        Replacement(
                            pos=(40, 40),
                            newtext="CMakeLists.txt "
                            "@rapidsai/cudf-cmake-codeowners\n",
                        ),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """
                pyproject.toml @rapidsai/ci-codeowners"""
            ),
            [
                LintWarning(
                    pos=(0, 0),
                    msg="missing required codeowners",
                    replacements=[
                        Replacement(
                            pos=(39, 39),
                            newtext="\nCMakeLists.txt "
                            "@rapidsai/cudf-cmake-codeowners\n",
                        ),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """
                pyproject.toml @rapidsai/ci-codeowners
                CMakeLists.txt @rapidsai/cudf-cmake-codeowners
                """
            ),
            [
                LintWarning(
                    pos=(1, 15),
                    msg="file 'pyproject.toml' should come after "
                    "'CMakeLists.txt'",
                    notes=[
                        Note(
                            pos=(40, 54),
                            msg="file 'CMakeLists.txt' is here",
                        ),
                    ],
                ),
            ],
        ),
    ],
)
@patch_required_codeowners_lines
def test_check_codeowners(content, warnings):
    linter = Linter(".github/CODEOWNERS", content)
    codeowners.check_codeowners(linter, Mock(project_prefix="cudf"))
    assert linter.warnings == warnings

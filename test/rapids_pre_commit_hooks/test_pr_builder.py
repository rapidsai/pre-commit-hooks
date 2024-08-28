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
from unittest.mock import Mock, patch

import pytest

from rapids_pre_commit_hooks.lint import Linter, LintWarning, Note, Replacement
from rapids_pre_commit_hooks.pr_builder import PRBuilderChecker


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        (
            dedent(
                """\
                jobs:
                  pr-builder: {}
                """
            ),
            [],
        ),
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    if: !cancelled()
                    with:
                      needs: ${{ toJSON(needs) }}
                """
            ),
            [
                LintWarning(
                    (24, 26),
                    "if specified, 'if' condition of 'pr-builder' should be 'always()'",
                    replacements=[
                        Replacement((28, 40), "always()"),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    if: always()
                """
            ),
            [
                LintWarning(
                    (24, 26),
                    "if 'if' condition is specified, pass 'needs: "
                    "${{ toJSON(needs) }}' as an input",
                    replacements=[
                        Replacement(
                            (37, 37), "    with:\n      needs: ${{ toJSON(needs) }}\n"
                        ),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    if: always()
                    with:
                      key: value
                """
            ),
            [
                LintWarning(
                    (24, 26),
                    "if 'if' condition is specified, pass 'needs: "
                    "${{ toJSON(needs) }}' as an input",
                    replacements=[
                        Replacement((64, 64), "      needs: ${{ toJSON(needs) }}\n"),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    if: always()
                    with:
                      needs: '{}'
                """
            ),
            [
                LintWarning(
                    (53, 58),
                    "'needs' input should be '${{ toJSON(needs) }}'",
                    replacements=[
                        Replacement((60, 64), "${{ toJSON(needs) }}"),
                    ],
                ),
            ],
        ),
    ],
)
def test_check_pr_builder_job(content, warnings):
    linter = Linter(".github/workflows/pr.yaml", content)
    checker = PRBuilderChecker(linter, Mock(ignore_dependencies=[]))
    checker.check_pr_builder_job(checker.root.value[0][1].value[0][1])
    assert linter.warnings == warnings


@pytest.mark.parametrize(
    ["content", "pr_builder_index", "warnings"],
    [
        (
            dedent(
                """\
                jobs:
                  pr-builder: {}
                  other-job-1: {}
                  other-job-2: {}
                """
            ),
            0,
            [],
        ),
        (
            dedent(
                """\
                jobs:
                  other-job-1: {}
                  pr-builder: {}
                  other-job-2: {}
                """
            ),
            1,
            [
                LintWarning(
                    (26, 36),
                    "place pr-builder job before all other jobs",
                    replacements=[
                        Replacement((26, 41), ""),
                        Replacement((8, 8), "pr-builder: {}\n"),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """\
                jobs:
                  other-job-1: {}
                  other-job-2: {}
                  pr-builder: {}
                """
            ),
            2,
            [
                LintWarning(
                    (44, 54),
                    "place pr-builder job before all other jobs",
                    replacements=[
                        Replacement((44, 59), ""),
                        Replacement((8, 8), "pr-builder: {}\n"),
                    ],
                ),
            ],
        ),
    ],
)
def test_check_jobs_first_job(content, pr_builder_index, warnings):
    linter = Linter(".github/workflows/pr.yaml", content)
    checker = PRBuilderChecker(linter, Mock(ignore_dependencies=[]))
    with patch.object(checker, "check_pr_builder_job") as mock_check_pr_builder_job:
        checker.check_jobs(checker.root.value[0][1])
    mock_check_pr_builder_job.assert_called_once_with(
        checker.root.value[0][1].value[pr_builder_index][1]
    )
    assert linter.warnings == warnings


@pytest.mark.parametrize(
    ["content", "ignore_dependencies", "warnings"],
    [
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    needs:
                      - other-job-1
                      - other-job-2
                      - other-job-3
                  other-job-1: {}
                  other-job-2: {}
                  other-job-3: {}
                """
            ),
            [],
            [],
        ),
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    needs:
                      - other-job-1
                      - other-job-3
                      - other-job-2
                  other-job-1: {}
                  other-job-2: {}
                  other-job-3: {}
                """
            ),
            [],
            [
                LintWarning(
                    (24, 29),
                    "'pr-builder' job should depend on all other jobs in the order "
                    "they appear",
                    notes=[
                        Note(
                            (24, 29),
                            "to ignore a job dependency, pass it as "
                            "--ignore-dependency",
                        ),
                    ],
                    replacements=[
                        Replacement(
                            (37, 90),
                            "- other-job-1\n      - other-job-2\n      - other-job-3",
                        ),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    with:
                      needs: ${{ toJSON(needs) }}
                  other-job-1: {}
                  other-job-2: {}
                  other-job-3: {}
                """
            ),
            [],
            [
                LintWarning(
                    (8, 18),
                    "'pr-builder' job should depend on all other jobs in the order "
                    "they appear",
                    notes=[
                        Note(
                            (8, 18),
                            "to ignore a job dependency, pass it as "
                            "--ignore-dependency",
                        ),
                    ],
                    replacements=[
                        Replacement(
                            (64, 64),
                            "    needs:\n"
                            "      - other-job-1\n"
                            "      - other-job-2\n"
                            "      - other-job-3\n",
                        ),
                    ],
                ),
            ],
        ),
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    needs:
                      - other-job-1
                      - other-job-2
                  other-job-1: {}
                  other-job-2: {}
                  other-job-3: {}
                """
            ),
            ["other-job-3"],
            [],
        ),
        (
            dedent(
                """\
                jobs:
                  pr-builder:
                    needs:
                      - other-job-1
                      - other-job-2
                      - other-job-3
                  other-job-1: {}
                  other-job-2: {}
                  other-job-3: {}
                """
            ),
            ["other-job-3"],
            [
                LintWarning(
                    (24, 29),
                    "'pr-builder' job should depend on all other jobs in the order "
                    "they appear",
                    notes=[
                        Note(
                            (24, 29),
                            "to ignore a job dependency, pass it as "
                            "--ignore-dependency",
                        ),
                    ],
                    replacements=[
                        Replacement(
                            (37, 90),
                            "- other-job-1\n      - other-job-2",
                        ),
                    ],
                ),
            ],
        ),
    ],
    ids=[
        "correct",
        "unsorted",
        "missing",
        "ignore-dependencies-correct",
        "ignore-dependencies-extra",
    ],
)
def test_check_other_jobs(content, ignore_dependencies, warnings):
    linter = Linter(".github/workflows/pr.yaml", content)
    checker = PRBuilderChecker(linter, Mock(ignore_dependencies=ignore_dependencies))
    checker.check()
    assert linter.warnings == warnings

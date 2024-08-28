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

import argparse
from typing import Optional

import yaml

from .lint import Linter, LintMain
from .utils.yaml import AnchorPreservingLoader, get_indent_before_node, node_has_type


class PRBuilderChecker:
    def __init__(self, linter: Linter, args: argparse.Namespace) -> None:
        self.linter: Linter = linter
        self.args: argparse.Namespace = args
        self.ignore_dependencies: set[str] = set(self.args.ignore_dependencies)

        loader = AnchorPreservingLoader(self.linter.content)
        try:
            root = loader.get_single_node()
            assert root is not None
        finally:
            loader.dispose()
        self.root: "yaml.Node" = root
        self.anchors: dict[str, "yaml.Node"] = loader.document_anchors[0]
        self.used_anchors: set[str] = set()
        self.other_jobs: list[tuple["yaml.Node", "yaml.Node"]] = []
        self.pr_builder_key: Optional["yaml.Node"] = None
        self.pr_builder_value: Optional["yaml.Node"] = None
        self.pr_builder_needs_key: Optional["yaml.Node"] = None
        self.pr_builder_needs_value: Optional["yaml.Node"] = None
        self.pr_builder_last_key: Optional["yaml.Node"] = None
        self.pr_builder_last_value: Optional["yaml.Node"] = None

    def append_needs(
        self,
        if_key: "yaml.Node",
        last_key: "yaml.Node",
        last_value: "yaml.Node",
        content: str,
    ) -> None:
        assert self.linter.content[last_value.end_mark.index] == "\n"

        indent = get_indent_before_node(self.linter.content, last_key)
        content = content.replace("\n", f"\n{indent}")
        content = f"{indent}{content}\n"
        self.linter.add_warning(
            (if_key.start_mark.index, if_key.end_mark.index),
            "if 'if' condition is specified, pass 'needs: ${{ toJSON(needs) }}' as an "
            "input",
        ).add_replacement(
            (last_value.end_mark.index + 1, last_value.end_mark.index + 1), content
        )

    def check_pr_builder_job(self, node: "yaml.Node") -> None:
        if node_has_type(node, "map"):
            if_key: Optional["yaml.Node"] = None
            if_value: Optional["yaml.Node"] = None
            with_key: Optional["yaml.Node"] = None
            with_value: Optional["yaml.Node"] = None
            for job_key, job_value in node.value:
                if node_has_type(job_key, "str"):
                    if job_key.value == "needs":
                        self.pr_builder_needs_key = job_key
                        self.pr_builder_needs_value = job_value
                    elif job_key.value == "if":
                        if_key = job_key
                        if_value = job_value
                    elif job_key.value == "with":
                        if node_has_type(job_value, "map"):
                            with_key = job_key
                            with_value = job_value
                    self.pr_builder_last_key = job_key
                    self.pr_builder_last_value = job_value
            if if_key:
                assert if_value
                if if_value.value != "always()":
                    self.linter.add_warning(
                        (if_key.start_mark.index, if_key.end_mark.index),
                        "if specified, 'if' condition of 'pr-builder' should be "
                        "'always()'",
                    ).add_replacement(
                        (if_value.start_mark.index, if_value.end_mark.index), "always()"
                    )
                if with_key:
                    assert with_value
                    for input_key, input_value in with_value.value:
                        if (
                            node_has_type(input_key, "str")
                            and input_key.value == "needs"
                        ):
                            if input_value.value != "${{ toJSON(needs) }}":
                                self.linter.add_warning(
                                    (
                                        input_key.start_mark.index,
                                        input_key.end_mark.index,
                                    ),
                                    "'needs' input should be '${{ toJSON(needs) }}'",
                                ).add_replacement(
                                    (
                                        input_value.start_mark.index,
                                        input_value.end_mark.index,
                                    ),
                                    "${{ toJSON(needs) }}",
                                )
                            break
                    else:
                        self.append_needs(
                            if_key,
                            input_key,
                            input_value,
                            "needs: ${{ toJSON(needs) }}",
                        )
                else:
                    assert self.pr_builder_last_key
                    assert self.pr_builder_last_value
                    self.append_needs(
                        if_key,
                        self.pr_builder_last_key,
                        self.pr_builder_last_value,
                        "with:\n  needs: ${{ toJSON(needs) }}",
                    )

    def check_jobs(
        self,
        node: "yaml.Node",
    ) -> None:
        if node_has_type(node, "map"):
            first = True
            first_key: Optional["yaml.Node"] = None
            pr_builder_key: Optional["yaml.Node"] = None
            pr_builder_value: Optional["yaml.Node"] = None
            for jobs_key, jobs_value in node.value:
                if node_has_type(jobs_key, "str"):
                    if jobs_key.value == "pr-builder":
                        self.pr_builder_key = jobs_key
                        self.pr_builder_value = jobs_value
                        self.check_pr_builder_job(jobs_value)
                        if not first:
                            pr_builder_key = jobs_key
                            pr_builder_value = jobs_value
                    else:
                        if first:
                            first_key = jobs_key
                        if jobs_key.value not in self.ignore_dependencies:
                            self.other_jobs.append(jobs_key.value)
                first = False
            if first_key and pr_builder_key:
                assert pr_builder_value
                w = self.linter.add_warning(
                    (pr_builder_key.start_mark.index, pr_builder_key.end_mark.index),
                    "place pr-builder job before all other jobs",
                )
                end = pr_builder_value.end_mark.index
                previous_newline = self.linter.content[:end].rfind("\n")
                if all(
                    c == " " for c in self.linter.content[previous_newline + 1 : end]
                ):
                    end = previous_newline
                previous_newline_pr_builder = self.linter.content[
                    : pr_builder_key.start_mark.index
                ].rfind("\n")
                assert all(
                    c == " "
                    for c in self.linter.content[
                        previous_newline + 1 : pr_builder_key.start_mark.index
                    ]
                )
                w.add_replacement(
                    (
                        previous_newline_pr_builder + 1,
                        end + 1,
                    ),
                    "",
                )
                previous_newline_first = self.linter.content[
                    : first_key.start_mark.index
                ].rfind("\n")
                assert all(
                    c == " "
                    for c in self.linter.content[
                        previous_newline + 1 : first_key.start_mark.index
                    ]
                )
                w.add_replacement(
                    (previous_newline_first + 1, previous_newline_first + 1),
                    self.linter.content[previous_newline_pr_builder + 1 : end + 1],
                )

    def check_root(self) -> None:
        if node_has_type(self.root, "map"):
            for root_key, root_value in self.root.value:
                if node_has_type(root_key, "str") and root_key.value == "jobs":
                    self.check_jobs(root_value)

    def check(self) -> None:
        self.check_root()

        # Final checks
        if self.pr_builder_key:
            assert self.pr_builder_value
            if self.pr_builder_needs_key:
                assert self.pr_builder_needs_value
                if (
                    not node_has_type(self.pr_builder_needs_value, "seq")
                    or [node.value for node in self.pr_builder_needs_value.value]
                    != self.other_jobs
                ):
                    w = self.linter.add_warning(
                        (
                            self.pr_builder_needs_key.start_mark.index,
                            self.pr_builder_needs_key.end_mark.index,
                        ),
                        "'pr-builder' job should depend on all other jobs in the "
                        "order they appear",
                    )
                    w.add_note(
                        (
                            self.pr_builder_needs_key.start_mark.index,
                            self.pr_builder_needs_key.end_mark.index,
                        ),
                        "to ignore a job dependency, pass it as --ignore-dependency",
                    )

                    end = self.pr_builder_needs_value.end_mark.index
                    previous_newline = self.linter.content[:end].rfind("\n")
                    if all(
                        c == " "
                        for c in self.linter.content[previous_newline + 1 : end]
                    ):
                        end = previous_newline
                    indent = get_indent_before_node(
                        self.linter.content, self.pr_builder_needs_value
                    )

                    w.add_replacement(
                        (self.pr_builder_needs_value.start_mark.index, end),
                        f"\n{indent}".join(f"- {job}" for job in self.other_jobs),
                    )
            else:
                assert self.pr_builder_last_key
                assert self.pr_builder_last_value

                w = self.linter.add_warning(
                    (
                        self.pr_builder_key.start_mark.index,
                        self.pr_builder_key.end_mark.index,
                    ),
                    "'pr-builder' job should depend on all other jobs in the order "
                    "they appear",
                )
                w.add_note(
                    (
                        self.pr_builder_key.start_mark.index,
                        self.pr_builder_key.end_mark.index,
                    ),
                    "to ignore a job dependency, pass it as --ignore-dependency",
                )

                end = self.pr_builder_value.end_mark.index
                previous_newline = self.linter.content[:end].rfind("\n")
                if all(
                    c == " " for c in self.linter.content[previous_newline + 1 : end]
                ):
                    end = previous_newline + 1
                indent = get_indent_before_node(
                    self.linter.content, self.pr_builder_last_key
                )

                w.add_replacement(
                    (end, end),
                    f"{indent}needs:\n"
                    + "".join(f"{indent}  - {job}\n" for job in self.other_jobs),
                )


def check_pr_builder(linter: Linter, args: argparse.Namespace) -> None:
    PRBuilderChecker(linter, args).check()


def main() -> None:
    m = LintMain()
    m.argparser.description = "Verify that the pr-builder job is set up correctly."
    m.argparser.add_argument(
        "--ignore-dependency",
        action="append",
        metavar="dependency",
        dest="ignore_dependencies",
        default=[],
        help="Do not require another job as a dependency",
    )
    with m.execute() as ctx:
        ctx.add_check(check_pr_builder)


if __name__ == "__main__":
    main()

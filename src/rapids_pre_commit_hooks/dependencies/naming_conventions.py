# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import re
from typing import TYPE_CHECKING

from ..utils.dependencies_yaml import Handler

if TYPE_CHECKING:
    import argparse
    from collections.abc import Generator
    from typing import Optional

    import yaml

    from ..lint import Linter


class NamingConventionsHandler(Handler):
    @dataclasses.dataclass(frozen=True)
    class NamingConventionKey:
        table: str = dataclasses.field(kw_only=True)
        key: "Optional[str]" = dataclasses.field(kw_only=True, default=None)

    PROJECT_NAMING_CONVENTIONS: "dict[NamingConventionKey, str]" = {
        NamingConventionKey(table="build-system"): "py_build_{project_name}",
        NamingConventionKey(
            table="tool.rapids-build-backend", key="requires"
        ): "py_rapids_build_{project_name}",
        NamingConventionKey(table="project"): "py_run_{project_name}",
        NamingConventionKey(
            table="project.optional-dependencies", key="test"
        ): "py_test_{project_name}",
    }

    @dataclasses.dataclass
    class FileContext:
        pyproject_output_node: "Optional[yaml.Node]" = None
        project_name: "Optional[str]" = None
        pyproject_dir_node: "Optional[yaml.Node]" = None
        table_node: "Optional[yaml.Node]" = None
        key_node: "Optional[yaml.Node]" = None

    def __init__(self, linter: "Linter", args: "argparse.Namespace") -> None:
        self.linter = linter
        self.args = args

    @contextlib.contextmanager
    def handle_files_item(
        self,
        files_context: "None",  # noqa: ARG002
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[NamingConventionsHandler.FileContext]":
        context = NamingConventionsHandler.FileContext()
        yield context

        if (
            context.pyproject_output_node
            and context.pyproject_dir_node
            and context.table_node
        ):
            naming_conventions_key = (
                NamingConventionsHandler.NamingConventionKey(
                    table=context.table_node.value,
                    key=None
                    if context.key_node is None
                    else context.key_node.value,
                )
            )
            if (
                naming_convention
                := NamingConventionsHandler.PROJECT_NAMING_CONVENTIONS.get(
                    naming_conventions_key
                )
            ):
                expected_file_key_name = naming_convention.format(
                    project_name=context.project_name
                )
                if key.value != expected_file_key_name:
                    w = self.linter.add_warning(
                        (key.start_mark.index, key.end_mark.index),
                        "expected file key name is "
                        f'"{expected_file_key_name}"',
                    )
                    w.add_replacement(
                        (key.start_mark.index, key.end_mark.index),
                        expected_file_key_name,
                    )
                    w.add_note(
                        (
                            context.pyproject_output_node.start_mark.index,
                            context.pyproject_output_node.end_mark.index,
                        ),
                        "file key has pyproject output type",
                    )
                    w.add_note(
                        (
                            context.pyproject_dir_node.start_mark.index,
                            context.pyproject_dir_node.end_mark.index,
                        ),
                        f'and project name "{context.project_name}"',
                    )
                    w.add_note(
                        (
                            context.table_node.start_mark.index,
                            context.table_node.end_mark.index,
                        ),
                        f'and table extra "{context.table_node.value}"',
                    )
                    if context.key_node:
                        w.add_note(
                            (
                                context.key_node.start_mark.index,
                                context.key_node.end_mark.index,
                            ),
                            f'and key extra "{context.key_node.value}"',
                        )

    def handle_file_output_item(
        self,
        file_output_context: "NamingConventionsHandler.FileContext",
        item: "yaml.Node",
    ) -> None:
        if item.value == "pyproject":
            file_output_context.pyproject_output_node = item

    def handle_pyproject_dir(
        self,
        files_item_context: "NamingConventionsHandler.FileContext",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",
    ) -> None:
        if match := re.search(
            r"^python/(?P<project_dirname>[^/]+)$", value.value
        ):
            files_item_context.project_name = match.group(
                "project_dirname"
            ).replace("-", "_")
            files_item_context.pyproject_dir_node = value

    def handle_extras_table(
        self,
        extras_context: "NamingConventionsHandler.FileContext",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",
    ) -> None:
        extras_context.table_node = value

    def handle_extras_key(
        self,
        extras_context: "NamingConventionsHandler.FileContext",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",
    ) -> None:
        extras_context.key_node = value

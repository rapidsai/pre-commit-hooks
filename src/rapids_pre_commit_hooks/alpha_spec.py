# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import dataclasses
import os
import re
from functools import cache
from typing import Any, Optional

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import Specifier, SpecifierSet
from rapids_metadata.metadata import RAPIDSMetadata, RAPIDSVersion
from rapids_metadata.remote import fetch_latest

from .lint import Linter, LintMain
from .utils.yaml import (
    Anchor,
    is_reference_anchor,
    preserve_anchor,
    quote_string,
)
from .utils.dependencies_yaml import (
    Handler,
    is_python_output_type,
    traverse_dependencies_yaml,
)

ALPHA_SPECIFIER: str = ">=0.0.0a0"

CUDA_SUFFIX_REGEX: re.Pattern = re.compile(r"^(?P<package>.*)-cu[0-9]{2}$")


@cache
def all_metadata() -> "RAPIDSMetadata":
    return fetch_latest()


def get_rapids_version(args: argparse.Namespace) -> "RAPIDSVersion":
    md = all_metadata()
    return (
        md.versions[args.rapids_version]
        if args.rapids_version
        else md.get_current_version(os.getcwd(), args.rapids_version_file)
    )


def strip_cuda_suffix(args: argparse.Namespace, name: str) -> str:
    if (match := CUDA_SUFFIX_REGEX.search(name)) and match.group(
        "package"
    ) in get_rapids_version(args).cuda_suffixed_packages:
        return match.group("package")
    return name


class OrderPreservingSpecifierSet(SpecifierSet):
    def __str__(self):
        return ",".join(str(s) for s in self)


class AlphaSpecHandler(Handler):
    @dataclasses.dataclass
    class ItemContext:
        has_python_output_type: bool = False
        packages: "list[tuple[Optional[Anchor], yaml.ScalarNode]]" = (
            dataclasses.field(default_factory=list)
        )

    @dataclasses.dataclass
    class PackagesContext:
        parent_context: "AlphaSpecHandler.ItemContext"
        packages_is_reference_anchor: bool

    def __init__(self, linter: Linter, args: argparse.Namespace):
        self.linter = linter
        self.args = args

    def handle_output_type(
        self,
        output_types_context: "AlphaSpecHandler.ItemContext",
        item: "yaml.Node",
    ) -> None:
        if is_python_output_type(item.value):
            output_types_context.has_python_output_type = True

    @contextlib.contextmanager
    def _handle_item(self) -> "Any":
        context = AlphaSpecHandler.ItemContext()
        yield context

        if context.has_python_output_type:
            for anchor, node in context.packages:
                self._check_package(anchor, node)

    def handle_common_item(
        self,
        common_context: "Any",  # noqa: ARG002
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Any":
        return self._handle_item()

    def handle_specific_item(
        self,
        specific_context: "Any",  # noqa: ARG002
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Any":
        return self._handle_item()

    def handle_packages(
        self,
        common_or_matrices_item_context: "AlphaSpecHandler.ItemContext",
        anchor: "Optional[Anchor]",  # noqa: ARG002
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.nullcontext[AlphaSpecHandler.PackagesContext]":
        return contextlib.nullcontext(
            AlphaSpecHandler.PackagesContext(
                common_or_matrices_item_context, is_reference_anchor(anchor)
            )
        )

    def handle_package(
        self,
        packages_context: "AlphaSpecHandler.PackagesContext",  # noqa: ARG002
        anchor: "Optional[Anchor]",
        node: "yaml.ScalarNode",
    ) -> None:
        if (
            packages_context.packages_is_reference_anchor
            or is_reference_anchor(anchor)
        ):
            return

        packages_context.parent_context.packages.append((anchor, node))

    def _check_package(
        self,
        anchor: "Optional[Anchor]",
        node: "yaml.ScalarNode",
    ) -> None:
        try:
            req = Requirement(node.value)
        except InvalidRequirement:
            return

        if (
            strip_cuda_suffix(self.args, req.name)
            not in get_rapids_version(self.args).prerelease_packages
        ):
            return

        has_alpha_spec = any(str(s) == ALPHA_SPECIFIER for s in req.specifier)
        if self.args.mode == "development" and not has_alpha_spec:
            req.specifier = OrderPreservingSpecifierSet(
                [*req.specifier, Specifier(ALPHA_SPECIFIER)]
            )
            self.linter.add_warning(
                (node.start_mark.index, node.end_mark.index),
                f"add alpha spec for RAPIDS package {req.name}",
            ).add_replacement(
                (node.start_mark.index, node.end_mark.index),
                preserve_anchor(
                    quote_string(
                        str(req),
                        node.style,
                    ),
                    anchor,
                ),
            )
        elif self.args.mode == "release" and has_alpha_spec:
            req.specifier = OrderPreservingSpecifierSet(
                s for s in req.specifier if str(s) != ALPHA_SPECIFIER
            )
            self.linter.add_warning(
                (node.start_mark.index, node.end_mark.index),
                f"remove alpha spec for RAPIDS package {req.name}",
            ).add_replacement(
                (node.start_mark.index, node.end_mark.index),
                preserve_anchor(
                    quote_string(
                        str(req),
                        node.style,
                    ),
                    anchor,
                ),
            )


def check_alpha_spec(linter: Linter, args: argparse.Namespace) -> None:
    handler = AlphaSpecHandler(linter, args)
    traverse_dependencies_yaml(handler, linter.content)


def main() -> None:
    m = LintMain("verify-alpha-spec")
    m.argparser.description = (
        "Verify that RAPIDS packages in dependencies.yaml do (or do not) have "
        "the alpha spec."
    )
    m.argparser.add_argument(
        "--mode",
        help="mode to use (development has alpha spec, release does not)",
        choices=["development", "release"],
        default="development",
    )
    m.argparser.add_argument(
        "--rapids-version",
        help="Specify a RAPIDS version to use instead of reading from the "
        "VERSION file",
    )
    m.argparser.add_argument(
        "--rapids-version-file",
        help="Specify a file to read the RAPIDS version from instead of "
        "VERSION",
        default="VERSION",
    )
    with m.execute() as ctx:
        ctx.add_check(check_alpha_spec)


if __name__ == "__main__":
    main()

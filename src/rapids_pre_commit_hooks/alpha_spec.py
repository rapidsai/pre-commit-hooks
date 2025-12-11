# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
from functools import cache, total_ordering

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from rapids_metadata.metadata import RAPIDSMetadata, RAPIDSVersion
from rapids_metadata.remote import fetch_latest

from .lint import Linter, LintMain

ALPHA_SPECIFIER: str = ">=0.0.0a0"

ALPHA_SPEC_OUTPUT_TYPES: set[str] = {
    "pyproject",
    "requirements",
}

CUDA_SUFFIX_REGEX: re.Pattern = re.compile(r"^(?P<package>.*)-cu[0-9]{2}$")


@cache
def all_metadata() -> "RAPIDSMetadata":
    return fetch_latest()


def node_has_type(node: "yaml.Node", tag_type: str) -> bool:
    return node.tag == f"tag:yaml.org,2002:{tag_type}"


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


def check_and_mark_anchor(
    anchors: dict[str, "yaml.Node"], used_anchors: set[str], node: "yaml.Node"
) -> tuple[bool, str | None]:
    for key, value in anchors.items():
        if value == node:
            anchor = key
            break
    else:
        anchor = None
    if anchor in used_anchors:
        return False, anchor
    if anchor is not None:
        used_anchors.add(anchor)
    return True, anchor


def check_package_spec(
    linter: Linter,
    args: argparse.Namespace,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    @total_ordering
    class SpecPriority:
        def __init__(self, spec: str):
            self.spec: str = spec

        def __eq__(self, other: object) -> bool:
            assert isinstance(other, SpecPriority)
            return self.spec == other.spec

        def __lt__(self, other: object) -> bool:
            assert isinstance(other, SpecPriority)
            if self.spec == other.spec:
                return False
            if self.spec == ALPHA_SPECIFIER:
                return False
            if other.spec == ALPHA_SPECIFIER:
                return True
            return self.sort_str() < other.sort_str()

        def sort_str(self) -> str:
            return "".join(c for c in self.spec if c not in "<>=")

    def create_specifier_string(specifiers: set[str]) -> str:
        return ",".join(sorted(specifiers, key=SpecPriority))

    if node_has_type(node, "str"):
        try:
            req = Requirement(node.value)
        except InvalidRequirement:
            return
        if (
            strip_cuda_suffix(args, req.name)
            in get_rapids_version(args).prerelease_packages
        ):
            descend, anchor = check_and_mark_anchor(
                anchors, used_anchors, node
            )
            if descend:
                has_alpha_spec = any(
                    str(s) == ALPHA_SPECIFIER for s in req.specifier
                )
                if args.mode == "development" and not has_alpha_spec:
                    linter.add_warning(
                        (node.start_mark.index, node.end_mark.index),
                        f"add alpha spec for RAPIDS package {req.name}",
                    ).add_replacement(
                        (node.start_mark.index, node.end_mark.index),
                        str(
                            (f"&{anchor} " if anchor else "")
                            + req.name
                            + create_specifier_string(
                                {str(s) for s in req.specifier}
                                | {ALPHA_SPECIFIER},
                            )
                        ),
                    )
                elif args.mode == "release" and has_alpha_spec:
                    linter.add_warning(
                        (node.start_mark.index, node.end_mark.index),
                        f"remove alpha spec for RAPIDS package {req.name}",
                    ).add_replacement(
                        (node.start_mark.index, node.end_mark.index),
                        str(
                            (f"&{anchor} " if anchor else "")
                            + req.name
                            + create_specifier_string(
                                {str(s) for s in req.specifier}
                                - {ALPHA_SPECIFIER},
                            )
                        ),
                    )


def check_packages(
    linter: Linter,
    args: argparse.Namespace,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        descend, _ = check_and_mark_anchor(anchors, used_anchors, node)
        if descend:
            for package_spec in node.value:
                check_package_spec(
                    linter, args, anchors, used_anchors, package_spec
                )


def check_common(
    linter: Linter,
    args: argparse.Namespace,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        for dependency_set in node.value:
            if node_has_type(dependency_set, "map"):
                for (
                    dependency_set_key,
                    dependency_set_value,
                ) in dependency_set.value:
                    if (
                        node_has_type(dependency_set_key, "str")
                        and dependency_set_key.value == "packages"
                    ):
                        check_packages(
                            linter,
                            args,
                            anchors,
                            used_anchors,
                            dependency_set_value,
                        )


def check_matrices(
    linter: Linter,
    args: argparse.Namespace,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        for item in node.value:
            if node_has_type(item, "map"):
                for matrix_key, matrix_value in item.value:
                    if (
                        node_has_type(matrix_key, "str")
                        and matrix_key.value == "packages"
                    ):
                        check_packages(
                            linter, args, anchors, used_anchors, matrix_value
                        )


def check_specific(
    linter: Linter,
    args: argparse.Namespace,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        for matrix_matcher in node.value:
            if node_has_type(matrix_matcher, "map"):
                for (
                    matrix_matcher_key,
                    matrix_matcher_value,
                ) in matrix_matcher.value:
                    if (
                        node_has_type(matrix_matcher_key, "str")
                        and matrix_matcher_key.value == "matrices"
                    ):
                        check_matrices(
                            linter,
                            args,
                            anchors,
                            used_anchors,
                            matrix_matcher_value,
                        )


def check_dependencies(
    linter: Linter,
    args: argparse.Namespace,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        for _, dependencies_value in node.value:
            if node_has_type(dependencies_value, "map"):
                for (
                    dependency_key,
                    dependency_value,
                ) in dependencies_value.value:
                    if node_has_type(dependency_key, "str"):
                        if dependency_key.value == "common":
                            check_common(
                                linter,
                                args,
                                anchors,
                                used_anchors,
                                dependency_value,
                            )
                        elif dependency_key.value == "specific":
                            check_specific(
                                linter,
                                args,
                                anchors,
                                used_anchors,
                                dependency_value,
                            )


def check_root(
    linter: Linter,
    args: argparse.Namespace,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        for root_key, root_value in node.value:
            if (
                node_has_type(root_key, "str")
                and root_key.value == "dependencies"
            ):
                check_dependencies(
                    linter, args, anchors, used_anchors, root_value
                )


class AnchorPreservingLoader(yaml.SafeLoader):
    """A SafeLoader that preserves the anchors for later reference. The anchors
    can be found in the document_anchors member, which is a list of
    dictionaries, one dictionary for each parsed document.
    """

    def __init__(self, stream) -> None:
        super().__init__(stream)
        self.document_anchors: list[dict[str, yaml.Node]] = []

    def compose_document(self) -> "yaml.Node":
        # Drop the DOCUMENT-START event.
        self.get_event()

        # Compose the root node.
        node = self.compose_node(None, None)  # type: ignore[arg-type]

        # Drop the DOCUMENT-END event.
        self.get_event()

        self.document_anchors.append(self.anchors)
        self.anchors = {}
        assert node is not None
        return node


def check_alpha_spec(linter: Linter, args: argparse.Namespace) -> None:
    loader = AnchorPreservingLoader(linter.content)
    try:
        root = loader.get_single_node()
        assert root is not None
    finally:
        loader.dispose()
    check_root(linter, args, loader.document_anchors[0], set(), root)


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

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

from functools import reduce

import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from .lint import LintMain

RAPIDS_VERSIONED_PACKAGES = {
    "rmm",
    "pylibcugraphops",
    "pylibcugraph",
    "nx-cugraph",
    "dask-cudf",
    "cuspatial",
    "cuproj",
    "cuml",
    "cugraph",
    "cudf",
    "ptxcompiler",
    "cubinlinker",
    "cugraph-dgl",
    "cugraph-pyg",
    "cugraph-equivariant",
    "raft-dask",
    "pylibwholegraph",
    "pylibraft",
    "cuxfilter",
    "cucim",
    "ucx-py",
    "ucxx",
    "pynvjitlink",
    "distributed-ucxx",
}

ALPHA_SPECIFIER = ">=0.0.0a0"

ALPHA_SPEC_OUTPUT_TYPES = {
    "pyproject",
    "requirements",
}


def check_package_spec(linter, args, node):
    if node.tag == "tag:yaml.org,2002:str":
        req = Requirement(node.value)
        if req.name in RAPIDS_VERSIONED_PACKAGES:
            has_alpha_spec = any(
                filter(lambda s: str(s) == ALPHA_SPECIFIER, req.specifier)
            )
            if args.mode == "development" and not has_alpha_spec:
                req.specifier &= ALPHA_SPECIFIER
                linter.add_warning(
                    (node.start_mark.index, node.end_mark.index),
                    f"add alpha spec for RAPIDS package {req.name}",
                ).add_replacement(
                    (node.start_mark.index, node.end_mark.index), str(req)
                )
            elif args.mode == "release" and has_alpha_spec:
                req.specifier = reduce(
                    lambda ss, s: ss & str(s),
                    filter(lambda s: str(s) != ALPHA_SPECIFIER, req.specifier),
                    SpecifierSet(),
                )
                linter.add_warning(
                    (node.start_mark.index, node.end_mark.index),
                    f"remove alpha spec for RAPIDS package {req.name}",
                ).add_replacement(
                    (node.start_mark.index, node.end_mark.index), str(req)
                )


def check_packages(linter, args, node):
    if node.tag == "tag:yaml.org,2002:seq":
        for package_spec in node.value:
            check_package_spec(linter, args, package_spec)


def check_common(linter, args, node):
    if node.tag == "tag:yaml.org,2002:seq":
        for dependency_set in node.value:
            if dependency_set.tag == "tag:yaml.org,2002:map":
                for dependency_set_key, dependency_set_value in dependency_set.value:
                    if (
                        dependency_set_key.tag == "tag:yaml.org,2002:str"
                        and dependency_set_key.value == "packages"
                    ):
                        packages = dependency_set_value
                check_packages(linter, args, packages)


def check_matrices(linter, args, node):
    if node.tag == "tag:yaml.org,2002:seq":
        for item in node.value:
            if item.tag == "tag:yaml.org,2002:map":
                for matrix_key, matrix_value in item.value:
                    if (
                        matrix_key.tag == "tag:yaml.org,2002:str"
                        and matrix_key.value == "packages"
                    ):
                        check_packages(linter, args, matrix_value)


def check_specific(linter, args, node):
    if node.tag == "tag:yaml.org,2002:seq":
        for matrix_matcher in node.value:
            if matrix_matcher.tag == "tag:yaml.org,2002:map":
                for matrix_matcher_key, matrix_matcher_value in matrix_matcher.value:
                    if (
                        matrix_matcher_key.tag == "tag:yaml.org,2002:str"
                        and matrix_matcher_key.value == "matrices"
                    ):
                        matrices = matrix_matcher_value
                check_matrices(linter, args, matrices)


def check_dependencies(linter, args, node):
    if node.tag == "tag:yaml.org,2002:map":
        for _, dependencies_value in node.value:
            if dependencies_value.tag == "tag:yaml.org,2002:map":
                for dependency_key, dependency_value in dependencies_value.value:
                    if dependency_key.tag == "tag:yaml.org,2002:str":
                        if dependency_key.value == "common":
                            check_common(linter, args, dependency_value)
                        elif dependency_key.value == "specific":
                            check_specific(linter, args, dependency_value)


def check_root(linter, args, node):
    if node.tag == "tag:yaml.org,2002:map":
        for root_key, root_value in node.value:
            if (
                root_key.tag == "tag:yaml.org,2002:str"
                and root_key.value == "dependencies"
            ):
                check_dependencies(linter, args, root_value)


def check_alpha_spec(linter, args):
    check_root(linter, args, yaml.compose(linter.content))


def main():
    m = LintMain()
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
    with m.execute() as ctx:
        ctx.add_check(check_alpha_spec)


if __name__ == "__main__":
    main()

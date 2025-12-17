# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import TYPE_CHECKING

from .lint import LintMain, Linter

if TYPE_CHECKING:
    import argparse
    import os
    from collections.abc import Iterator

HARDCODED_VERSION_RE: re.Pattern = re.compile(
    r"(?:^|\D)(?P<full>(?P<major>\d{1,2})\.(?P<minor>\d{1,2})(?:\.(?P<patch>\d{1,2}))?)(?=\D|$)"
)


def find_hardcoded_versions(
    content: str, full_version: tuple[int, int, int]
) -> "Iterator[re.Match[str]]":
    major, minor, patch = full_version
    return (
        match
        for match in HARDCODED_VERSION_RE.finditer(content)
        if int(match.group("major")) == major
        and int(match.group("minor")) == minor
        and (not match.group("patch") or int(match.group("patch")) == patch)
    )


def read_version_file(
    filename: "str | os.PathLike[str]",
) -> tuple[int, int, int]:
    with open(filename) as f:
        contents = f.read()
    match = HARDCODED_VERSION_RE.search(contents)
    assert match
    assert contents == f"{match.group('full')}\n"
    assert match.group("patch")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def check_hardcoded_version(
    linter: Linter, args: "argparse.Namespace"
) -> None:
    if linter.filename == args.version_file:
        return
    full_version = read_version_file(args.version_file)
    for match in find_hardcoded_versions(linter.content, full_version):
        linter.add_warning(
            match.span("full"),
            f"do not hard-code version, read from {args.version_file} "
            "file instead",
        )


def main() -> None:
    m = LintMain("verify-hardcoded-version")
    m.argparser.description = (
        "Verify that files do not contain hard-coded software versions."
    )
    m.argparser.add_argument(
        "--version-file",
        help="Specify a file to read the version from instead of VERSION",
        default="VERSION",
    )
    with m.execute() as ctx:
        ctx.add_check(check_hardcoded_version)


if __name__ == "__main__":
    main()

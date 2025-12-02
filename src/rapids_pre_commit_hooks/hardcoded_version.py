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
    r"(?:^|\D)(?P<full_version>(?P<major_minor_version>\d{1,2}\.\d{2})(?:\.(?P<patch_version>\d{2}))?)(?=\D|$)"
)


def find_hardcoded_versions(
    content: str, full_version: str
) -> "Iterator[re.Match[str]]":
    search_match = HARDCODED_VERSION_RE.search(full_version)
    assert search_match
    assert search_match.span() == (0, len(full_version))
    major_minor_version = search_match.group("major_minor_version")
    return (
        match
        for match in HARDCODED_VERSION_RE.finditer(content)
        if match.group("full_version") == full_version
        or (
            not match.group("patch_version")
            and match.group("major_minor_version") == major_minor_version
        )
    )


def read_version_file(filename: "str | os.PathLike[str]") -> str:
    with open(filename) as f:
        contents = f.read()
    match = HARDCODED_VERSION_RE.search(contents)
    assert match
    assert contents == f"{match.group('full_version')}\n"
    assert match.group("patch_version")
    return match.group("full_version")


def check_hardcoded_version(
    linter: Linter, args: "argparse.Namespace"
) -> None:
    if linter.filename == args.version_file:
        return
    full_version = read_version_file(args.version_file)
    for match in find_hardcoded_versions(linter.content, full_version):
        linter.add_warning(
            match.span("full_version"),
            f"do not hard-code version, read from {args.version_file} "
            "file instead",
        )


def main() -> None:
    m = LintMain()
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

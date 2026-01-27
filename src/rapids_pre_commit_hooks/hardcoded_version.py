# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import bisect
import re
from typing import TYPE_CHECKING

import tomlkit
import tomlkit.exceptions

from .lint import LintMain, Linter
from .utils.toml import find_value_location

if TYPE_CHECKING:
    import argparse
    import os
    from collections.abc import Generator

    from .lint import Lines

# Matches any 2-part or 3-part numeric version strings, and stores the
# components in named capture groups:
#
# * "full" = entire version
# * "major" = first component (required)
# * "minor" = second component  (required)
# * "patch" = third component (optional)
#
# Common cases it intentionally does not match:
#
# * full-year calendar versions (e.g. "2026.2.0")
HARDCODED_VERSION_RE: re.Pattern = re.compile(
    r"(?:^|\D)(?P<full>(?P<major>\d{1,2})\.(?P<minor>\d{1,2})(?:\.(?P<patch>\d{1,2}))?)(?=\D|$)"
)

PYPROJECT_TOML_RE: re.Pattern = re.compile(r"(?:^|/)pyproject\.toml$")
DEPRECATED_RE: re.Pattern = re.compile(r"\.\. deprecated::|@deprecated")
NUMBER_ARRAY_RE: re.Pattern = re.compile(r"^[ .,0-9-]*$")


def get_excluded_section_pyproject_toml(
    document: tomlkit.TOMLDocument, path: tuple[str, ...]
) -> "Generator[tuple[int, int]]":
    try:
        yield find_value_location(document, path, append=False)
    except tomlkit.exceptions.NonExistentKey:
        pass


def get_excluded_sections_pyproject_toml(
    linter: Linter,
) -> "Generator[tuple[int, int]]":
    document = tomlkit.loads(linter.content)

    yield from get_excluded_section_pyproject_toml(
        document, ("project", "dependencies")
    )
    yield from get_excluded_section_pyproject_toml(
        document, ("project", "optional-dependencies")
    )
    yield from get_excluded_section_pyproject_toml(
        document, ("build-system", "requires")
    )
    yield from get_excluded_section_pyproject_toml(
        document, ("tool", "rapids-build-backend", "requires")
    )


def get_excluded_sections(linter: Linter) -> "Generator[tuple[int, int]]":
    if PYPROJECT_TOML_RE.search(linter.filename):
        yield from get_excluded_sections_pyproject_toml(linter)


def is_deprecation_notice(lines: "Lines", match: "re.Match[str]") -> bool:
    this_line = lines.line_for_pos(match.start())
    first_line = max(0, this_line - 3)
    start = lines.pos[first_line][0]
    end = lines.pos[this_line][1]
    return bool(DEPRECATED_RE.search(lines.content[start:end]))


def is_number_array(lines: "Lines", match: "re.Match[str]") -> bool:
    this_line = lines.line_for_pos(match.start())
    start, end = lines.pos[this_line]
    return bool(NUMBER_ARRAY_RE.search(lines.content[start:end]))


def skip_heuristics(lines: "Lines", match: "re.Match[str]") -> bool:
    if is_deprecation_notice(lines, match):
        return True
    if is_number_array(lines, match):
        return True
    return False


def find_hardcoded_versions(
    content: str, full_version: tuple[int, int, int]
) -> "Generator[re.Match[str]]":
    """Detect all instances of a specific 2- or 3-part version in text
    content."""

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
    assert match, (
        f'Expected file "{filename}" to contain a 3-part numeric version, but '
        "it was not found"
    )
    assert contents == f"{match.group('full')}\n", (
        f'Expected file "{filename}" to contain ONLY a 3-part numeric '
        "version, but additional content was found, or no trailing "
        "newline was found"
    )
    assert match.group("patch"), (
        f'Expected file "{filename}" to contain a 3-part numeric version, but '
        "the patch (3rd) part was not found"
    )
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def check_hardcoded_version(
    linter: Linter, args: "argparse.Namespace"
) -> None:
    # If the linter is currently checking the VERSION file, don't issue any
    # warnings
    if linter.filename == args.version_file:
        return

    full_version = read_version_file(args.version_file)
    excluded_sections = sorted(get_excluded_sections(linter))
    for match in find_hardcoded_versions(linter.content, full_version):
        section_index = bisect.bisect_right(
            excluded_sections, match.span("full")
        )
        if section_index > 0:
            section_start, section_end = excluded_sections[section_index - 1]
            if (
                match.start("full") >= section_start
                and match.end("full") <= section_end
            ):
                continue

        if skip_heuristics(linter.lines, match):
            continue

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
        help="File to read the version from (default: VERSION)",
        default="VERSION",
    )
    with m.execute() as ctx:
        ctx.add_check(check_hardcoded_version)


if __name__ == "__main__":
    main()

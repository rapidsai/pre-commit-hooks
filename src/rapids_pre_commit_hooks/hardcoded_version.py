# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bisect
import re
from enum import Enum
from textwrap import dedent
from typing import TYPE_CHECKING

import tomlkit
import tomlkit.exceptions

from .lint import LintMain, Linter
from .utils.toml import find_value_span

if TYPE_CHECKING:
    import argparse
    import os
    from collections.abc import Generator
    from typing import Optional

    from .lint import Lines, Span

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
DEPRECATED_VERSIONADD_RE: re.Pattern = re.compile(
    r"\.\. (?:deprecated|versionchanged|versionadded)::|@deprecated"
)
NUMBER_ARRAY_RE: re.Pattern = re.compile(r"^[ .,0-9-]*$")
VERSION_DOC_RE: re.Pattern = re.compile(
    r"\b(?:in|since|after) (?:(?:version|release) )?$", re.IGNORECASE
)
TODO_DOC_RE: re.Pattern = re.compile(
    r"TODO\((?P<version>\d{1,2}+(?:\.\d{1,2})+)\)"
)


class VersionScheme(Enum):
    CALVER = "calver"
    SEMVER = "semver"


def get_previous_minor_version(
    version: tuple[int, int, int], version_scheme: VersionScheme
) -> "Optional[tuple[int, int, int]]":
    if version_scheme == VersionScheme.CALVER:
        year, month, _ = version
        month_index = year * 12 + month - 1
        prev_month_index = month_index - 2
        prev_year = prev_month_index // 12
        prev_month = (prev_month_index % 12) + 1
        return (prev_year, prev_month, 0)
    elif version_scheme == VersionScheme.SEMVER:
        major, minor, _ = version
        if minor == 0:
            return None
        return (major, minor - 1, 0)

    raise ValueError


def get_next_minor_version(
    version: tuple[int, int, int], version_scheme: VersionScheme
) -> tuple[int, int, int]:
    if version_scheme == VersionScheme.CALVER:
        year, month, _ = version
        month_index = year * 12 + month - 1
        next_month_index = month_index + 2
        next_year = next_month_index // 12
        next_month = (next_month_index % 12) + 1
        return (next_year, next_month, 0)
    elif version_scheme == VersionScheme.SEMVER:
        major, minor, _ = version
        return (major, minor + 1, 0)

    raise ValueError


def get_excluded_span_pyproject_toml(
    document: tomlkit.TOMLDocument, path: tuple[str, ...]
) -> "Generator[Span]":
    try:
        yield find_value_span(document, path, append=False)
    except tomlkit.exceptions.NonExistentKey:
        pass


def get_excluded_spans_pyproject_toml(
    linter: Linter,
) -> "Generator[Span]":
    document = tomlkit.loads(linter.content)

    yield from get_excluded_span_pyproject_toml(
        document, ("project", "dependencies")
    )
    yield from get_excluded_span_pyproject_toml(
        document, ("project", "optional-dependencies")
    )
    yield from get_excluded_span_pyproject_toml(
        document, ("build-system", "requires")
    )
    yield from get_excluded_span_pyproject_toml(
        document, ("tool", "rapids-build-backend", "requires")
    )


def get_excluded_spans(linter: Linter) -> "Generator[Span]":
    if PYPROJECT_TOML_RE.search(linter.filename):
        yield from get_excluded_spans_pyproject_toml(linter)


def is_deprecation_versionadd(lines: "Lines", match: "re.Match[str]") -> bool:
    this_line = lines.line_for_pos(match.start("full"))
    first_line = max(0, this_line - 3)
    start = lines.spans[first_line][0]
    end = lines.spans[this_line][1]
    return bool(DEPRECATED_VERSIONADD_RE.search(lines.content[start:end]))


def is_number_array(lines: "Lines", match: "re.Match[str]") -> bool:
    this_line = lines.line_for_pos(match.start("full"))
    start, end = lines.spans[this_line]
    return bool(NUMBER_ARRAY_RE.search(lines.content[start:end]))


def is_version_doc(lines: "Lines", match: "re.Match[str]") -> bool:
    return bool(VERSION_DOC_RE.search(lines.content[: match.start("full")]))


def is_todo_doc(lines: "Lines", match: "re.Match[str]") -> bool:
    this_line = lines.line_for_pos(match.start("full"))
    start, end = lines.spans[this_line]
    todo_match = TODO_DOC_RE.search(lines.content[start:end])
    return (
        todo_match is not None
        and start + todo_match.start("version") == match.start("full")
        and start + todo_match.end("version") == match.end("full")
    )


def skip_heuristics(lines: "Lines", match: "re.Match[str]") -> bool:
    if is_deprecation_versionadd(lines, match):
        return True
    if is_number_array(lines, match):
        return True
    if is_version_doc(lines, match):
        return True
    if is_todo_doc(lines, match):
        return True
    return False


def find_hardcoded_versions(
    content: str,
    full_version: tuple[int, int, int],
    version_scheme: VersionScheme,
) -> "Generator[re.Match[str]]":
    """Detect all instances of a specific 2- or 3-part version in text
    content."""

    to_check = [
        full_version,
        get_next_minor_version(full_version, version_scheme),
    ]
    if (
        prev := get_previous_minor_version(full_version, version_scheme)
    ) is not None:
        to_check.append(prev)
    return (
        match
        for match in HARDCODED_VERSION_RE.finditer(content)
        if any(
            int(match.group("major")) == major
            and int(match.group("minor")) == minor
            and (
                not match.group("patch") or int(match.group("patch")) == patch
            )
            for major, minor, patch in to_check
        )
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


def agent_instructions_hardcoded_version(*, version_file: str) -> str:
    return dedent(
        f'''\
        This check exists to ensure that RAPIDS packages do not hard-code the
        RAPIDS version in a script when programmatically reading from the
        {version_file} file would suffice. Doing so reduces maintenance
        burden and reduces the possibility of hard-coded versions in scripts
        neglecting to be updated in a version bump.

        This check is very prone to false positives. The following conditions
        can trigger a false positive:

        1. Floating-point numbers in code that incidentally happen to equal the
        version number even with no relation.

        2. Code documentation that states when a function was added,
        deprecated, etc.

        3. Files that genuinely use the version but are incapable of
        programmatically reading from the {version_file} file.

        This check has some heuristics to catch most of these false positives,
        but due to the nature of the problem, we can't catch everything. Choose
        one of the following options to resolve this problem (in order of
        preference):

        1. If you are writing a script that already reads from the
        {version_file} file, use the stored result from that read.

        2. If you are writing a script that is capable of doing so, but does
        not yet do so, programmatically read from the {version_file} file.

        3. If you are writing documentation, comments, etc., you have likely
        encountered a false positive. You may optionally file an issue or PR
        at https://github.com/rapidsai/pre-commit-hooks to add a heuristic to
        suppress the false positive.

        4. Suppress the false positive with a next-line directive. For example:

          // rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
          // This function was added to the library as of 26.10
          void new_function();

        Or use a block directive:

          def new_function():
              # rapids-pre-commit-hooks: disable[verify-hardcoded-version]
              """New function that does stuff.

              Added to the library as of 26.10
              """
              # rapids-pre-commit-hooks: enable[verify-hardcoded-version]
              pass

        5. If you are writing a file in a format that has no mechanism for
        writing comments, or if false positives are extremely common in the
        file format, add an exclusion to .pre-commit-config.yaml.
        '''  # noqa: E501
    )


def check_hardcoded_version(
    linter: Linter, args: "argparse.Namespace"
) -> None:
    # If the linter is currently checking the VERSION file, don't issue any
    # warnings
    if linter.filename == args.version_file:
        return

    full_version = read_version_file(args.version_file)
    excluded_spans = sorted(get_excluded_spans(linter))
    for match in find_hardcoded_versions(
        linter.content, full_version, VersionScheme(args.version_scheme)
    ):
        span_index = bisect.bisect_right(excluded_spans, match.span("full"))
        if span_index > 0:
            span_start, span_end = excluded_spans[span_index - 1]
            if (
                match.start("full") >= span_start
                and match.end("full") <= span_end
            ):
                continue

        if skip_heuristics(linter.lines, match):
            continue

        w = linter.add_warning(
            match.span("full"),
            f"do not hard-code version, read from {args.version_file} "
            "file instead",
        )
        w.add_note(
            match.span("full"),
            "if this is intentional (as part of a docstring or deprecation "
            "notice), suppress it with rapids-pre-commit-hooks: "
            "disable-next-line[verify-hardcoded-version] - see "
            "https://github.com/rapidsai/pre-commit-hooks/blob/main/README.md"
            "#suppressing-false-positives for details",
        )
        w.add_agent_instructions(
            agent_instructions_hardcoded_version(
                version_file=args.version_file
            )
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
    m.argparser.add_argument(
        "--version-scheme",
        help="Version scheme to use (default: calver)",
        default=VersionScheme.CALVER.value,
        choices=[s.value for s in VersionScheme],
    )
    with m.execute() as ctx:
        ctx.add_check(check_hardcoded_version)


if __name__ == "__main__":
    main()

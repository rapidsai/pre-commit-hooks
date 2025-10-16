# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import dataclasses
import datetime
import functools
import os
import re
import warnings
from collections.abc import Callable, Generator, Iterable
from typing import Optional

import git

from .lint import Lines, Linter, LintMain, LintWarning

COPYRIGHT_PATTERN: str = (
    r"(?P<full_copyright_text>"
    r"(?P<nvidia_copyright_text>Copyright *(?:\(c\))? *"
    r"(?P<years>(?P<first_year>\d{4})(-(?P<last_year>\d{4}))?),?"
    r" *NVIDIA C(?:ORPORATION|orporation))[^\r\n]*)"
)
COPYRIGHT_RE: re.Pattern = re.compile(COPYRIGHT_PATTERN)
SPDX_COPYRIGHT_RE: re.Pattern = re.compile(
    r"(?P<spdx_filecopyrighttext_tag>SPDX-FileCopyrightText: )"
    rf"{COPYRIGHT_PATTERN}"
    r"(?:(?:\n|\r\n|\r)[^\r\n]*"
    r"(?P<spdx_license_identifier_tag>SPDX-License-Identifier: )"
    r"(?P<spdx_license_identifier_text>[^\r\n]+))?"
)
BRANCH_RE: re.Pattern = re.compile(r"^branch-(?P<major>\d+)\.(?P<minor>\d+)$")
COPYRIGHT_REPLACEMENT: str = (
    "Copyright (c) {first_year}-{last_year}, NVIDIA CORPORATION"
)
C_STYLE_COMMENTS_RE: re.Pattern = re.compile(
    r"\.(?:c|cpp|cxx|cu|h|hpp|hxx|cuh|js|java|rs)$"
)

LONG_FORM_LICENSE_TEXT: dict[str, list[list[str]]] = {
    "Apache-2.0": [
        [
            "",
            'Licensed under the Apache License, Version 2.0 (the "License");',
            "you may not use this file except in compliance with the License.",
            "You may obtain a copy of the License at",
            "",
            "    http://www.apache.org/licenses/LICENSE-2.0",
            "",
            "Unless required by applicable law or agreed to in writing, software",  # noqa: E501
            'distributed under the License is distributed on an "AS IS" BASIS,',  # noqa: E501
            "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",  # noqa: E501
            "See the License for the specific language governing permissions and",  # noqa: E501
            "limitations under the License.",
        ],
        [
            "",
            'Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except',  # noqa: E501
            "in compliance with the License. You may obtain a copy of the License at",  # noqa: E501
            "",
            "http://www.apache.org/licenses/LICENSE-2.0",
            "",
            "Unless required by applicable law or agreed to in writing, software distributed under the License",  # noqa: E501
            'is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express',  # noqa: E501
            "or implied. See the License for the specific language governing permissions and limitations under",  # noqa: E501
            "the License.",
        ],
        [
            "",
            'Licensed under the Apache License, Version 2.0 (the "License");',
            "you may not use this file except in compliance with the License.",
            "You may obtain a copy of the License at",
            "",
            "    http://www.apache.org/licenses/LICENSE-2.0",
            "",
            "Unless required by applicable law or agreed to in writing, software distributed under the License",  # noqa: E501
            'is distributed on an "AS IS" BASIS,  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express',  # noqa: E501
            "or implied. See the License for the specific language governing permissions and limitations under",  # noqa: E501
            "the License.",
        ],
        [
            'Licensed under the Apache License, Version 2.0 (the "License");',
            "you may not use this file except in compliance with the License.",
            "You may obtain a copy of the License at",
            "",
            "    http://www.apache.org/licenses/LICENSE-2.0",
            "",
            "Unless required by applicable law or agreed to in writing, software",  # noqa: E501
            'distributed under the License is distributed on an "AS IS" BASIS,',  # noqa: E501
            "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",  # noqa: E501
            "See the License for the specific language governing permissions and",  # noqa: E501
            "limitations under the License.",
        ],
    ],
}


_PosType = tuple[int, int]


@dataclasses.dataclass
class CopyrightMatch:
    span: _PosType
    spdx_filecopyrighttext_tag_span: _PosType | None
    full_copyright_text_span: _PosType
    nvidia_copyright_text_span: _PosType
    years_span: _PosType
    first_year_span: _PosType
    last_year_span: _PosType | None
    spdx_license_identifier_tag_span: _PosType | None
    spdx_license_identifier_text_span: _PosType | None
    long_form_text_span: _PosType | None = None


class NoTargetBranchWarning(RuntimeWarning):
    pass


class ConflictingFilesWarning(RuntimeWarning):
    pass


def match_copyright(lines: Lines, start: int = 0) -> CopyrightMatch | None:
    def min_start(m: re.Match | None) -> int:
        assert m
        return m.start()

    if re_match := min(
        filter(
            bool,
            [
                SPDX_COPYRIGHT_RE.search(lines.content, start),
                COPYRIGHT_RE.search(lines.content, start),
            ],
        ),
        default=None,
        key=min_start,
    ):

        def optional_match(name: str) -> _PosType | None:
            try:
                return (
                    span if (span := re_match.span(name)) != (-1, -1) else None
                )
            except IndexError:
                return None

        match = CopyrightMatch(
            span=re_match.span(),
            spdx_filecopyrighttext_tag_span=optional_match(
                "spdx_filecopyrighttext_tag"
            ),
            full_copyright_text_span=re_match.span("full_copyright_text"),
            nvidia_copyright_text_span=re_match.span("nvidia_copyright_text"),
            years_span=re_match.span("years"),
            first_year_span=re_match.span("first_year"),
            last_year_span=optional_match("last_year"),
            spdx_license_identifier_tag_span=optional_match(
                "spdx_license_identifier_tag"
            ),
            spdx_license_identifier_text_span=optional_match(
                "spdx_license_identifier_text"
            ),
        )

        try:
            license_identifier = re_match.group("spdx_license_identifier_text")
        except IndexError:
            license_identifier = None

        if pos := find_long_form_text(
            lines, license_identifier, re_match.end()
        ):
            match.long_form_text_span = pos
            match.span = (match.span[0], pos[1])

        return match

    return None


def match_all_copyright(lines: Lines) -> Generator[CopyrightMatch]:
    start = 0

    while match := match_copyright(lines, start):
        yield match
        start = match.span[1]


def find_long_form_text(
    lines: Lines, identifier: str | None, index: int
) -> _PosType | None:
    line = lines.line_for_pos(index)
    rest_of_lines = lines.pos[line + 1 :]

    licenses: Iterable[list[list[str]]] = LONG_FORM_LICENSE_TEXT.values()
    if identifier:
        try:
            licenses = [LONG_FORM_LICENSE_TEXT[identifier]]
        except KeyError:
            return None

    for license in licenses:
        for text_lines in license:
            if len(rest_of_lines) < len(text_lines):
                continue

            prefix: str | None = None
            first_line: str | None = None
            for file_pos, text_line in zip(rest_of_lines, text_lines):
                file_line = lines.content[file_pos[0] : file_pos[1]]
                if first_line is None:
                    first_line = file_line
                if text_line == "":
                    if prefix is None or prefix.startswith(file_line):
                        continue
                else:
                    if prefix is None:
                        if file_line.endswith(text_line):
                            prefix = file_line[: -len(text_line)]
                            continue
                    elif file_line == f"{prefix}{text_line}":
                        continue

                break
            else:
                assert prefix is not None
                assert first_line is not None
                return (
                    lines.pos[line + 1][0] + min(len(prefix), len(first_line)),
                    lines.pos[line + len(text_lines)][1],
                )

    return None


def strip_copyright(
    lines: Lines, copyright_matches: list[CopyrightMatch]
) -> list[str]:
    segments = []

    def append_segment(start: int, item: CopyrightMatch) -> int:
        segments.append(lines.content[start : item.span[0]])
        return item.span[1]

    start = functools.reduce(append_segment, copyright_matches, 0)
    segments.append(lines.content[start:])
    return segments


def add_copy_rename_note(
    linter: Linter,
    warning: LintWarning,
    change_type: str,
    old_filename: str | os.PathLike[str] | None,
) -> None:
    CHANGE_VERBS = {
        "C": "copied",
        "R": "renamed",
    }
    try:
        change_verb = CHANGE_VERBS[change_type]
    except KeyError:
        pass
    else:
        warning.add_note(
            (0, len(linter.content)),
            f"file was {change_verb} from '{old_filename}' and is assumed to "
            "share history with it",
        )
        warning.add_note(
            (0, len(linter.content)),
            "change file contents if you want its copyright dates to only be "
            "determined by its own edit history",
        )


def apply_copyright_revert(
    linter: Linter,
    change_type: str,
    old_filename: str | os.PathLike[str] | None,
    old_content: str,
    old_match: CopyrightMatch,
    new_match: CopyrightMatch,
) -> None:
    if (
        old_content[slice(*old_match.years_span)]
        == linter.content[slice(*new_match.years_span)]
    ):
        warning_pos = new_match.full_copyright_text_span
    else:
        warning_pos = new_match.years_span
    w = linter.add_warning(
        warning_pos,
        "copyright is not out of date and should not be updated",
    )
    w.add_replacement(
        new_match.full_copyright_text_span,
        old_content[slice(*old_match.full_copyright_text_span)],
    )
    add_copy_rename_note(linter, w, change_type, old_filename)


def apply_copyright_update(
    linter: Linter,
    match: CopyrightMatch,
    year: int,
) -> None:
    w = linter.add_warning(match.years_span, "copyright is out of date")
    w.add_replacement(
        match.nvidia_copyright_text_span,
        COPYRIGHT_REPLACEMENT.format(
            first_year=linter.content[slice(*match.first_year_span)],
            last_year=year,
        ),
    )


def apply_spdx_filecopyrighttext_tag_insert(
    linter: Linter, match: CopyrightMatch
) -> None:
    span = (
        match.full_copyright_text_span[0],
        match.full_copyright_text_span[0],
    )
    w = linter.add_warning(
        match.full_copyright_text_span, "include SPDX-FileCopyrightText header"
    )
    w.add_replacement(span, "SPDX-FileCopyrightText: ")


def apply_spdx_license_update(
    linter: Linter, match: CopyrightMatch, identifier: str
) -> None:
    assert match.spdx_license_identifier_tag_span
    assert match.spdx_license_identifier_text_span
    w = linter.add_warning(
        (
            match.spdx_license_identifier_tag_span[0],
            match.spdx_license_identifier_text_span[1],
        ),
        "SPDX-License-Identifier is incorrect",
    )
    w.add_replacement(match.spdx_license_identifier_text_span, identifier)


def apply_spdx_license_insert(
    linter: Linter, match: CopyrightMatch, identifier: str
) -> None:
    match_start_pos = (
        match.spdx_filecopyrighttext_tag_span or match.full_copyright_text_span
    )[0]
    line = linter.lines.line_for_pos(match_start_pos)

    line_start_pos = linter.lines.pos[line][0]
    line_start = linter.content[line_start_pos:match_start_pos]

    if C_STYLE_COMMENTS_RE.search(linter.filename):
        line_start = line_start.replace("/*", " *")

    next_line_start_pos = linter.lines.pos[line][1]
    w = linter.add_warning(
        (match_start_pos, match.full_copyright_text_span[1]),
        "no SPDX-License-Identifier header found",
    )
    w.add_replacement(
        (next_line_start_pos, next_line_start_pos),
        f"\n{line_start}SPDX-License-Identifier: {identifier}",
    )


def apply_spdx_long_form_text_removal(
    linter: Linter, match: CopyrightMatch
) -> None:
    assert match.long_form_text_span
    span = (
        (
            match.spdx_license_identifier_text_span
            or match.full_copyright_text_span
        )[1],
        match.long_form_text_span[1],
    )
    w = linter.add_warning(
        match.long_form_text_span, "remove long-form copyright text"
    )
    w.add_replacement(span, "")


def apply_spdx_updates(
    linter: Linter, args: argparse.Namespace, match: CopyrightMatch
) -> None:
    if not match.spdx_filecopyrighttext_tag_span:
        apply_spdx_filecopyrighttext_tag_insert(linter, match)
    if not match.spdx_license_identifier_text_span:
        apply_spdx_license_insert(linter, match, args.spdx_license_identifier)
    elif (
        linter.content[slice(*match.spdx_license_identifier_text_span)]
        != args.spdx_license_identifier
    ):
        apply_spdx_license_update(linter, match, args.spdx_license_identifier)
    if match.long_form_text_span:
        apply_spdx_long_form_text_removal(linter, match)


def apply_copyright_check(
    linter: Linter,
    args: argparse.Namespace,
    change_type: str,
    old_filename: str | os.PathLike[str] | None,
    old_content: str | None,
) -> None:
    content_changed = linter.content != old_content

    if content_changed or args.force_spdx:
        year_env = os.getenv("RAPIDS_TEST_YEAR")
        if year_env:
            try:
                current_year = int(year_env)
            except ValueError:
                current_year = datetime.datetime.now().year
        else:
            current_year = datetime.datetime.now().year
        new_copyright_matches = list(match_all_copyright(linter.lines))

        if old_content is not None:
            old_lines = Lines(old_content)
            old_copyright_matches = list(match_all_copyright(old_lines))

        def match_year_sort(match: CopyrightMatch) -> tuple[int, int]:
            return (
                int(
                    linter.content[
                        slice(*(match.last_year_span or match.first_year_span))
                    ]
                ),
                int(linter.content[slice(*match.first_year_span)]),
            )

        if old_content is not None and strip_copyright(
            old_lines, old_copyright_matches
        ) == strip_copyright(linter.lines, new_copyright_matches):
            if content_changed:
                for old_match, new_match in zip(
                    old_copyright_matches, new_copyright_matches
                ):
                    if (
                        old_content[slice(*old_match.full_copyright_text_span)]
                        != linter.content[
                            slice(*new_match.full_copyright_text_span)
                        ]
                    ):
                        apply_copyright_revert(
                            linter,
                            change_type,
                            old_filename,
                            old_content,
                            old_match,
                            new_match,
                        )

            if args.force_spdx:
                newest_match = max(new_copyright_matches, key=match_year_sort)
                apply_spdx_updates(linter, args, newest_match)
        elif new_copyright_matches:
            newest_match = max(new_copyright_matches, key=match_year_sort)
            if (
                int(
                    linter.content[
                        slice(
                            *(
                                newest_match.last_year_span
                                or newest_match.first_year_span
                            )
                        )
                    ]
                )
                < current_year
            ):
                apply_copyright_update(linter, newest_match, current_year)
            if args.spdx or args.force_spdx:
                apply_spdx_updates(linter, args, newest_match)
        elif content_changed:
            linter.add_warning((0, 0), "no copyright notice found")


def get_target_branch(
    repo: "git.Repo", args: argparse.Namespace
) -> str | None:
    """Determine which branch is the "target" branch.

    The target branch is determined in the following order:

    * If the ``--target-branch`` argument is passed, that branch is used. This
      allows users to set a base branch on the command line.
    * If the ``$TARGET_BRANCH`` environment variable is defined, that branch is
      used. This allows users to locally set a base branch on a one-time basis.
    * If the ``$GITHUB_BASE_REF`` environment variable is defined, that branch
      is used. This allows GitHub Actions to easily use this tool.
    * If the ``$RAPIDS_BASE_BRANCH`` environment variable is defined, that
      branch is used. This allows GitHub Actions inside ``copy-pr-bot`` to
      easily use this tool.
    * If the Git configuration option ``rapidsai.baseBranch`` is defined, that
      branch is used. This allows users to locally set a base branch on a
      long-term basis.
    * If the ``--main-branch`` argument is passed, that branch is used. This
      allows projects to use a branching strategy other than
      ``branch-<major>.<minor>``.
    * If a ``branch-<major>.<minor>`` branch exists, that branch is used. If
      more than one such branch exists, the one with the latest version is
      used. This supports the expected default.
    * Otherwise, None is returned and a warning is issued.
    """
    # Try --target-branch
    if args.target_branch:
        return args.target_branch

    # Try environment
    if target_branch_name := os.getenv("TARGET_BRANCH"):
        return target_branch_name
    if target_branch_name := os.getenv("GITHUB_BASE_REF"):
        return target_branch_name
    if target_branch_name := os.getenv("RAPIDS_BASE_BRANCH"):
        return target_branch_name

    # Try config
    with repo.config_reader() as r:
        target_branch_name = r.get("rapidsai", "baseBranch", fallback=None)
    if target_branch_name:
        return target_branch_name

    # Try --main-branch
    if args.main_branch:
        return args.main_branch

    # Try newest branch-xx.yy
    try:
        return max(
            (
                (branch, (match.group("major"), match.group("minor")))
                for branch in repo.heads
                if (match := BRANCH_RE.search(branch.name))
            ),
            key=lambda i: i[1],
        )[0].name
    except ValueError:
        pass

    # Appropriate branch not found
    warnings.warn(
        "Could not determine target branch. Try setting the TARGET_BRANCH "
        "environment variable, or setting the rapidsai.baseBranch "
        "configuration option.",
        NoTargetBranchWarning,
    )
    return None


def get_target_branch_upstream_commit(
    repo: "git.Repo", args: argparse.Namespace
) -> git.Commit | None:
    # If no target branch can be determined, use HEAD if it exists
    target_branch_name = get_target_branch(repo, args)
    if target_branch_name is None:
        try:
            return repo.head.commit
        except ValueError:
            return None

    commits_to_try = []

    try:
        target_branch = repo.heads[target_branch_name]
    except IndexError:
        pass
    else:
        # Try the branch specified by the branch name
        commits_to_try.append(target_branch.commit)

        # If the branch has an upstream, try it and exit
        if target_branch_upstream := target_branch.tracking_branch():
            return max(
                [target_branch.commit, target_branch_upstream.commit],
                key=lambda commit: commit.committed_datetime,
            )

    def try_get_ref(remote: "git.Remote") -> Optional["git.Reference"]:
        try:
            return remote.refs[target_branch_name]
        except IndexError:
            return None

    try:
        # Try branches in all remotes that have the branch name
        upstream_commit = max(
            (
                upstream
                for remote in repo.remotes
                if (upstream := try_get_ref(remote))
            ),
            key=lambda upstream: upstream.commit.committed_datetime,
        ).commit
    except ValueError:
        pass
    else:
        commits_to_try.append(upstream_commit)

    if commits_to_try:
        return max(
            commits_to_try, key=lambda commit: commit.committed_datetime
        )

    # No branch with the specified name, local or remote, can be found, so
    # return HEAD if it exists
    try:
        return repo.head.commit
    except ValueError:
        return None


def get_changed_files(
    args: argparse.Namespace,
) -> dict[str | os.PathLike[str], tuple[str, Optional["git.Blob"]]]:
    try:
        repo = git.Repo()
    except git.InvalidGitRepositoryError:
        return {
            os.path.relpath(os.path.join(dirpath, filename), "."): ("A", None)
            for dirpath, dirnames, filenames in os.walk(".")
            for filename in filenames
        }

    changed_files: dict[
        str | os.PathLike[str], tuple[str, Optional["git.Blob"]]
    ] = {f: ("A", None) for f in repo.untracked_files}
    target_branch_upstream_commit = get_target_branch_upstream_commit(
        repo, args
    )
    if target_branch_upstream_commit is None:
        changed_files.update(
            {blob.path: ("A", None) for _, blob in repo.index.iter_blobs()}
        )
        return changed_files

    for merge_base in repo.merge_base(
        repo.head.commit, target_branch_upstream_commit, all=True
    ):
        diffs = merge_base.diff(
            other=None,
            find_copies=True,
            find_copies_harder=True,
            find_renames=True,
        )
        for diff in diffs:
            if diff.change_type == "A":
                assert diff.b_path is not None
                changed_files[diff.b_path] = (diff.change_type, None)
            elif diff.change_type != "D":
                assert diff.b_path is not None
                assert diff.change_type is not None
                changed_files[diff.b_path] = (diff.change_type, diff.a_blob)

    return changed_files


def normalize_git_filename(filename: str | os.PathLike[str]) -> str | None:
    relpath = os.path.relpath(filename)
    if re.search(r"^\.\.(/|$)", relpath):
        return None
    return relpath


def find_blob(
    tree: "git.Tree", filename: str | os.PathLike[str]
) -> Optional["git.Blob"]:
    d1, d2 = os.path.split(filename)
    split = [d2]
    while d1:
        d1, d2 = os.path.split(d1)
        split.insert(0, d2)

    while len(split) > 1:
        component = split.pop(0)
        try:
            tree = next(t for t in tree.trees if t.name == component)
        except StopIteration:
            return None

    try:
        return next(blob for blob in tree.blobs if blob.name == split[0])
    except StopIteration:
        return None


def check_copyright(
    args: argparse.Namespace,
) -> Callable[[Linter, argparse.Namespace], None]:
    changed_files = get_changed_files(args)

    def the_check(linter: Linter, args: argparse.Namespace) -> None:
        if not (git_filename := normalize_git_filename(linter.filename)):
            warnings.warn(
                f'File "{linter.filename}" is outside of current directory. '
                "Not running linter on it.",
                ConflictingFilesWarning,
            )
            return

        old_filename: str | os.PathLike[str] | None
        old_content: str | None
        try:
            change_type, changed_file = changed_files[git_filename]
        except KeyError:
            if args.force_spdx:
                change_type = "M"
                old_filename = linter.filename
                old_content = linter.content
            else:
                return
        else:
            if changed_file is None:
                old_filename = None
                old_content = None
            else:
                old_filename = changed_file.path
                old_content = changed_file.data_stream.read().decode()
        apply_copyright_check(
            linter, args, change_type, old_filename, old_content
        )

    return the_check


def main() -> None:
    m = LintMain()
    m.argparser.description = (
        "Verify that all files have had their copyright notices updated. Each "
        "file will be compared against the target branch (determined "
        "automatically or with the --target-branch argument) to decide "
        "whether or not they need a copyright update.\n\n"
        "--main-branch and --target-branch effectively control the same "
        "thing, but --target-branch has higher precedence and is meant only "
        "for a user-local override, while --main-branch is a project-wide "
        "setting. Both --main-branch and --target-branch may be specified."
    )
    m.argparser.add_argument(
        "--main-branch",
        metavar="<main branch>",
        help="main branch to use instead of branch-<major>.<minor>",
    )
    m.argparser.add_argument(
        "--target-branch",
        metavar="<target branch>",
        help="target branch to check modified files against",
    )
    m.argparser.add_argument(
        "--spdx",
        action="store_true",
        help="require SPDX headers",
    )
    m.argparser.add_argument(
        "--force-spdx",
        action="store_true",
        help="enforce SPDX headers even if the file hasn't changed "
        "(implies --spdx)",
    )
    m.argparser.add_argument(
        "--spdx-license-identifier",
        metavar="<license identifier>",
        help="content of SPDX-License-Identifier header",
        default="Apache-2.0",
    )
    with m.execute() as ctx:
        ctx.add_check(check_copyright(ctx.args))


if __name__ == "__main__":
    main()

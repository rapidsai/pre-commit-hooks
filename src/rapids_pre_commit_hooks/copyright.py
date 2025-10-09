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
import datetime
import functools
import os
import re
import warnings
from collections.abc import Callable
from typing import Optional

import git

from .lint import Lines, Linter, LintMain, LintWarning

SPDX_COPYRIGHT_RE: re.Pattern = re.compile(
    r"(?P<spdx>SPDX-FileCopyrightText: )?"
    r"(?P<text>Copyright *(?:\(c\))? *"
    r"(?P<years>(?P<first_year>\d{4})(-(?P<last_year>\d{4}))?),?"
    r" *NVIDIA C(?:ORPORATION|orporation))[^\r\n]*"
)
SPDX_LICENSE_PATTERN: str = (
    r"SPDX-License-Identifier: (?P<identifier>[a-zA-Z0-9.-]+)"
)
SPDX_LICENSE_RE: re.Pattern = re.compile(SPDX_LICENSE_PATTERN)
SPDX_LICENSE_LINE_RE: re.Pattern = re.compile(
    rf"(?:\n|\r\n|\r)[^\n\r]*{SPDX_LICENSE_PATTERN}[^\n\r]*"
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
    ],
}


class NoTargetBranchWarning(RuntimeWarning):
    pass


class ConflictingFilesWarning(RuntimeWarning):
    pass


def match_copyright(content: str) -> list[re.Match]:
    return list(SPDX_COPYRIGHT_RE.finditer(content))


def find_long_form_text(
    lines: Lines, identifier: str, index: int
) -> tuple[int, int] | None:
    line = lines.line_for_pos(index)
    rest_of_lines = lines.pos[line + 1 :]

    try:
        long_form_texts = LONG_FORM_LICENSE_TEXT[identifier]
    except KeyError:
        return None

    for text_lines in long_form_texts:
        if len(rest_of_lines) < len(text_lines):
            continue

        prefix: str | None = None
        for file_pos, text_line in zip(rest_of_lines, text_lines):
            file_line = lines.content[file_pos[0] : file_pos[1]]
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
            return (lines.pos[line][1], lines.pos[line + len(text_lines)][1])

    return None


def strip_copyright(
    args: argparse.Namespace, lines: Lines, copyright_matches: list[re.Match]
) -> list[str]:
    segments = []

    def calculate_start(first: bool, start: int) -> int:
        if (args.spdx or args.force_spdx) and not first:
            if match := SPDX_LICENSE_LINE_RE.search(lines.content, start):
                start = match.end()
            if pos := find_long_form_text(
                lines, args.spdx_license_identifier, start
            ):
                start = pos[1]
        return start

    def append_segment(
        first_and_start: tuple[bool, int], item: re.Match
    ) -> tuple[bool, int]:
        first, start = first_and_start
        segments.append(
            lines.content[calculate_start(first, start) : item.start()]
        )
        return False, item.end()

    first, start = functools.reduce(
        append_segment, copyright_matches, (True, 0)
    )
    segments.append(lines.content[calculate_start(first, start) :])
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
    old_match: re.Match,
    new_match: re.Match,
) -> None:
    if old_match.group("years") == new_match.group("years"):
        warning_pos = new_match.span()
    else:
        warning_pos = new_match.span("years")
    w = linter.add_warning(
        warning_pos,
        "copyright is not out of date and should not be updated",
    )
    w.add_replacement(new_match.span(), old_match.group())
    add_copy_rename_note(linter, w, change_type, old_filename)


def apply_copyright_update(
    linter: Linter,
    match: re.Match,
    year: int,
) -> None:
    w = linter.add_warning(match.span("years"), "copyright is out of date")
    w.add_replacement(
        match.span("text"),
        COPYRIGHT_REPLACEMENT.format(
            first_year=match.group("first_year"),
            last_year=year,
        ),
    )


def apply_spdx_text_insert(linter: Linter, match: re.Match) -> None:
    span = (match.span("text")[0], match.span("text")[0])
    w = linter.add_warning(
        match.span("text"), "include SPDX-FileCopyrightText header"
    )
    w.add_replacement(span, "SPDX-FileCopyrightText: ")


def apply_spdx_license_update(
    linter: Linter, match: re.Match, identifier: str
) -> None:
    w = linter.add_warning(
        match.span(), "SPDX-License-Identifier is incorrect"
    )
    w.add_replacement(match.span("identifier"), identifier)


def apply_spdx_license_insert(
    linter: Linter, matches: list[re.Match], identifier: str
) -> None:
    w = linter.add_warning((0, 0), "no SPDX-License-Identifier header found")
    for match in matches:
        match_start_pos = match.span()[0]
        line = linter.lines.line_for_pos(match_start_pos)
        line_start_pos = linter.lines.pos[line][0]
        line_start = linter.content[line_start_pos:match_start_pos]
        if C_STYLE_COMMENTS_RE.search(linter.filename):
            line_start = line_start.replace("/*", " *")
        next_line_start_pos = linter.lines.pos[line][1]
        w.add_replacement(
            (next_line_start_pos, next_line_start_pos),
            f"\n{line_start}SPDX-License-Identifier: {identifier}",
        )


def apply_spdx_long_form_text_removal(
    linter: Linter, args: argparse.Namespace, match: re.Match
) -> None:
    if (
        pos := find_long_form_text(
            linter.lines, args.spdx_license_identifier, match.span()[0]
        )
    ) is not None:
        w = linter.add_warning(pos, "remove long-form copyright text")
        w.add_replacement(pos, "")


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
        new_copyright_matches = match_copyright(linter.content)

        if old_content is not None:
            old_copyright_matches = match_copyright(old_content)

        if old_content is not None and strip_copyright(
            args, Lines(old_content), old_copyright_matches
        ) == strip_copyright(args, linter.lines, new_copyright_matches):
            if content_changed or args.force_spdx:
                for old_match, new_match in zip(
                    old_copyright_matches, new_copyright_matches
                ):
                    if (
                        old_match.group("text") != new_match.group("text")
                        and content_changed
                    ):
                        apply_copyright_revert(
                            linter,
                            change_type,
                            old_filename,
                            old_match,
                            new_match,
                        )
                    if args.force_spdx:
                        if not new_match.group("spdx"):
                            apply_spdx_text_insert(linter, new_match)
                        apply_spdx_long_form_text_removal(
                            linter, args, new_match
                        )
        elif new_copyright_matches:
            for match in new_copyright_matches:
                if (
                    int(match.group("last_year") or match.group("first_year"))
                    < current_year
                ) and linter.content != old_content:
                    apply_copyright_update(linter, match, current_year)
                if args.spdx or args.force_spdx:
                    if not match.group("spdx"):
                        apply_spdx_text_insert(linter, match)
                    apply_spdx_long_form_text_removal(linter, args, match)
        elif linter.content != old_content:
            linter.add_warning((0, 0), "no copyright notice found")

    if (args.spdx and content_changed) or args.force_spdx:
        found = False
        for match in SPDX_LICENSE_RE.finditer(linter.content):
            found = True
            if match.group("identifier") != args.spdx_license_identifier:
                apply_spdx_license_update(
                    linter, match, args.spdx_license_identifier
                )
            apply_spdx_long_form_text_removal(linter, args, match)
        if not found:
            apply_spdx_license_insert(
                linter,
                new_copyright_matches,
                args.spdx_license_identifier,
            )


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

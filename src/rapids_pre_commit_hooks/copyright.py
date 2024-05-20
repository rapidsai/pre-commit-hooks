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

import datetime
import functools
import os
import re
import warnings

import git

from .lint import LintMain

COPYRIGHT_RE = re.compile(
    r"Copyright *(?:\(c\))? *(?P<years>(?P<first_year>\d{4})(-(?P<last_year>\d{4}))?),?"
    r" *NVIDIA C(?:ORPORATION|orporation)"
)
BRANCH_RE = re.compile(r"^branch-(?P<major>[0-9]+)\.(?P<minor>[0-9]+)$")
COPYRIGHT_REPLACEMENT = "Copyright (c) {first_year}-{last_year}, NVIDIA CORPORATION"


class NoTargetBranchWarning(RuntimeWarning):
    pass


class ConflictingFilesWarning(RuntimeWarning):
    pass


def match_copyright(content):
    return list(COPYRIGHT_RE.finditer(content))


def strip_copyright(content, copyright_matches):
    lines = []

    def append_stripped(start, item):
        lines.append(content[start : item.start()])
        return item.end()

    start = functools.reduce(append_stripped, copyright_matches, 0)
    lines.append(content[start:])
    return lines


def apply_copyright_revert(linter, old_match, new_match):
    if old_match.group("years") == new_match.group("years"):
        warning_pos = new_match.span()
    else:
        warning_pos = new_match.span("years")
    linter.add_warning(
        warning_pos,
        "copyright is not out of date and should not be updated",
    ).add_replacement(new_match.span(), old_match.group())


def apply_copyright_update(linter, match, year):
    linter.add_warning(match.span("years"), "copyright is out of date").add_replacement(
        match.span(),
        COPYRIGHT_REPLACEMENT.format(
            first_year=match.group("first_year"),
            last_year=year,
        ),
    )


def apply_copyright_check(linter, old_content):
    if linter.content != old_content:
        current_year = datetime.datetime.now().year
        new_copyright_matches = match_copyright(linter.content)

        if old_content is not None:
            old_copyright_matches = match_copyright(old_content)

        if old_content is not None and strip_copyright(
            old_content, old_copyright_matches
        ) == strip_copyright(linter.content, new_copyright_matches):
            for old_match, new_match in zip(
                old_copyright_matches, new_copyright_matches
            ):
                if old_match.group() != new_match.group():
                    apply_copyright_revert(linter, old_match, new_match)
        elif new_copyright_matches:
            for match in new_copyright_matches:
                if (
                    int(match.group("last_year") or match.group("first_year"))
                    < current_year
                ):
                    apply_copyright_update(linter, match, current_year)
        else:
            linter.add_warning((0, 0), "no copyright notice found")


def get_target_branch(repo, args):
    """Determine which branch is the "target" branch.

    The target branch is determined in the following order:

    * If the ``--target-branch`` argument is passed, that branch is used. This allows
      users to set a base branch on the command line.
    * If the ``$TARGET_BRANCH`` environment variable is defined, that branch is
      used. This allows users to locally set a base branch on a one-time basis.
    * If the ``$GITHUB_BASE_REF`` environment variable is defined, that branch is used.
      This allows GitHub Actions to easily use this tool.
    * If the ``$RAPIDS_BASE_BRANCH`` environment variable is defined, that branch is
      used. This allows GitHub Actions inside ``copy-pr-bot`` to easily use this tool.
    * If the Git configuration option ``rapidsai.baseBranch`` is defined, that branch is
      used. This allows users to locally set a base branch on a long-term basis.
    * If the ``--main-branch`` argument is passed, that branch is used. This allows
      projects to use a branching strategy other than ``branch-<major>.<minor>``.
    * If a ``branch-<major>.<minor>`` branch exists, that branch is used. If more than
      one such branch exists, the one with the latest version is used. This supports the
      expected default.
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
        "Could not determine target branch. Try setting the TARGET_BRANCH environment "
        "variable, or setting the rapidsai.baseBranch configuration option.",
        NoTargetBranchWarning,
    )
    return None


def get_target_branch_upstream_commit(repo, args):
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

    def try_get_ref(remote):
        try:
            return remote.refs[target_branch_name]
        except IndexError:
            return None

    try:
        # Try branches in all remotes that have the branch name
        upstream_commit = max(
            (upstream for remote in repo.remotes if (upstream := try_get_ref(remote))),
            key=lambda upstream: upstream.commit.committed_datetime,
        ).commit
    except ValueError:
        pass
    else:
        commits_to_try.append(upstream_commit)

    if commits_to_try:
        return max(commits_to_try, key=lambda commit: commit.committed_datetime)

    # No branch with the specified name, local or remote, can be found, so return HEAD
    # if it exists
    try:
        return repo.head.commit
    except ValueError:
        return None


def get_changed_files(args):
    try:
        repo = git.Repo()
    except git.InvalidGitRepositoryError:
        return {
            os.path.relpath(os.path.join(dirpath, filename), "."): None
            for dirpath, dirnames, filenames in os.walk(".")
            for filename in filenames
        }

    changed_files = {f: None for f in repo.untracked_files}
    target_branch_upstream_commit = get_target_branch_upstream_commit(repo, args)
    if target_branch_upstream_commit is None:
        changed_files.update({blob.path: None for _, blob in repo.index.iter_blobs()})
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
                changed_files[diff.b_path] = None
            elif diff.change_type != "D":
                changed_files[diff.b_path] = diff.a_blob

    return changed_files


def normalize_git_filename(filename):
    relpath = os.path.relpath(filename)
    if re.search(r"^\.\.(/|$)", relpath):
        return None
    return relpath


def find_blob(tree, filename):
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


def check_copyright(args):
    changed_files = get_changed_files(args)

    def the_check(linter, args):
        if not (git_filename := normalize_git_filename(linter.filename)):
            warnings.warn(
                f'File "{linter.filename}" is outside of current directory. Not '
                "running linter on it.",
                ConflictingFilesWarning,
            )
            return

        try:
            changed_file = changed_files[git_filename]
        except KeyError:
            return

        old_content = (
            changed_file.data_stream.read().decode()
            if changed_file is not None
            else None
        )
        apply_copyright_check(linter, old_content)

    return the_check


def main():
    m = LintMain()
    m.argparser.description = (
        "Verify that all files have had their copyright notices updated. Each file "
        "will be compared against the target branch (determined automatically or with "
        "the --target-branch argument) to decide whether or not they need a copyright "
        "update.\n\n"
        "--main-branch and --target-branch effectively control the same thing, but "
        "--target-branch has higher precedence and is meant only for a user-local "
        "override, while --main-branch is a project-wide setting. Both --main-branch "
        "and --target-branch may be specified."
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
    with m.execute() as ctx:
        ctx.add_check(check_copyright(ctx.args))


if __name__ == "__main__":
    main()

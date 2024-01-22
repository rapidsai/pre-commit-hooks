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


class NoTargetBranchWarning(Warning):
    pass


class ConflictingFilesError(RuntimeError):
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
                    if old_match.group("years") == new_match.group("years"):
                        warning_pos = new_match.span()
                    else:
                        warning_pos = new_match.span("years")
                    linter.add_warning(
                        warning_pos,
                        "copyright is not out of date and should not be updated",
                    ).add_replacement(new_match.span(), old_match.group())
        else:
            if new_copyright_matches:
                for match in new_copyright_matches:
                    if (
                        int(match.group("last_year") or match.group("first_year"))
                        < current_year
                    ):
                        linter.add_warning(
                            match.span("years"), "copyright is out of date"
                        ).add_replacement(
                            match.span(),
                            COPYRIGHT_REPLACEMENT.format(
                                first_year=match.group("first_year"),
                                last_year=current_year,
                            ),
                        )
            else:
                linter.add_warning((0, 0), "no copyright notice found")


def get_target_branch(repo):
    """Determine which branch is the "target" branch.

    The target branch is determined in the following order:

    * If the ``$GITHUB_BASE_REF`` environment variable is defined and points to a valid
      branch, that branch is used. This allows GitHub Actions to easily use this tool.
    * If either of the ``$TARGET_BRANCH`` or ``$RAPIDS_BASE_BRANCH`` environment
      variables, in that order, are defined and point to a valid branch, that branch is
      used. This allows users to locally set a base branch on a one-time basis.
    * If the configuration option ``rapidsai.baseBranch`` points to a valid branch, that
      branch is used. This allows users to locally set a base branch on a long-term
      basis.
    * If a ``branch-<major>.<minor>`` branch exists, that branch is used. If more than
      one such branch exists, the one with the latest version is used. This supports the
      expected default.
    * Otherwise, None is returned and a warning is issued.
    """
    # Try environment
    if target_branch_name := os.getenv("GITHUB_BASE_REF"):
        try:
            return repo.heads[target_branch_name]
        except IndexError:
            pass
    if target_branch_name := os.getenv("TARGET_BRANCH"):
        try:
            return repo.heads[target_branch_name]
        except IndexError:
            pass
    if target_branch_name := os.getenv("RAPIDS_BASE_BRANCH"):
        try:
            return repo.heads[target_branch_name]
        except IndexError:
            pass

    # Try config
    with repo.config_reader() as r:
        target_branch_name = r.get("rapidsai", "baseBranch", fallback=None)
    if target_branch_name:
        try:
            return repo.heads[target_branch_name]
        except IndexError:
            pass

    # Try newest branch-xx.yy
    branches = sorted(
        (
            (branch, (match.group("major"), match.group("minor")))
            for branch in repo.heads
            if (match := BRANCH_RE.search(branch.name))
        ),
        key=lambda i: i[1],
        reverse=True,
    )
    try:
        return branches[0][0]
    except IndexError:
        pass

    # Appropriate branch not found
    warnings.warn(
        "Could not determine target branch. Try setting the TARGET_BRANCH or "
        "RAPIDS_BASE_BRANCH environment variable.",
        NoTargetBranchWarning,
    )
    return None


def get_target_branch_upstream_commit(repo):
    target_branch = get_target_branch(repo)
    if target_branch is None:
        try:
            return repo.head.commit
        except ValueError:
            return None

    target_branch_upstream = target_branch.tracking_branch()
    if target_branch_upstream:
        return target_branch_upstream.commit

    def try_get_ref(remote):
        try:
            return remote.refs[target_branch.name]
        except IndexError:
            return None

    candidate_upstreams = sorted(
        (upstream for remote in repo.remotes if (upstream := try_get_ref(remote))),
        key=lambda upstream: upstream.commit.committed_datetime,
        reverse=True,
    )
    try:
        return candidate_upstreams[0].commit
    except IndexError:
        pass

    return target_branch.commit


def get_changed_files():
    try:
        repo = git.Repo()
    except git.InvalidGitRepositoryError:
        return {
            os.path.relpath(os.path.join(dirpath, filename), "."): None
            for dirpath, dirnames, filenames in os.walk(".")
            for filename in filenames
        }

    changed_files = {f: None for f in repo.untracked_files}
    target_branch_upstream_commit = get_target_branch_upstream_commit(repo)
    if target_branch_upstream_commit is None:
        changed_files.update({blob.path: None for _, blob in repo.index.iter_blobs()})
        return changed_files

    diffs = target_branch_upstream_commit.diff(
        other=None,
        merge_base=True,
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


def check_copyright():
    changed_files = get_changed_files()

    def the_check(linter, args):
        try:
            changed_file = changed_files[linter.filename]
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
    with m.execute() as ctx:
        ctx.add_check(check_copyright())


if __name__ == "__main__":
    main()

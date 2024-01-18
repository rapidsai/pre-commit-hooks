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

import git

from .lint import LintMain

COPYRIGHT_RE = re.compile(
    r"Copyright *(?:\(c\))? *(?P<years>(?P<first_year>\d{4})(-(?P<last_year>\d{4}))?),?"
    r" *NVIDIA C(?:ORPORATION|orporation)"
)
BRANCH_RE = re.compile(r"^branch-(?P<major>[0-9]+)\.(?P<minor>[0-9]+)$")


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
                            f"Copyright (c) {match.group('first_year')}-{current_year}"
                            ", NVIDIA CORPORATION",
                        )
            else:
                linter.add_warning((0, 0), "no copyright notice found")


def get_target_branch(repo):
    # Try environment
    target_branch_name = os.getenv("GITHUB_BASE_REF")
    if target_branch_name:
        try:
            return repo.heads[target_branch_name]
        except IndexError:
            pass
    target_branch_name = os.getenv("TARGET_BRANCH")
    if target_branch_name:
        try:
            return repo.heads[target_branch_name]
        except IndexError:
            pass
    target_branch_name = os.getenv("RAPIDS_BASE_BRANCH")
    if target_branch_name:
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
    return None


def get_target_branch_upstream_commit(repo):
    target_branch = get_target_branch(repo)
    if target_branch is None:
        return repo.head.commit

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


def get_changed_files(repo, target_branch_upstream_commit):
    changed_files = {}

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

    changed_files.update({f: None for f in repo.untracked_files})
    return changed_files


def check_copyright():
    repo = git.Repo()
    target_branch_upstream_commit = get_target_branch_upstream_commit(repo)
    changed_files = get_changed_files(repo, target_branch_upstream_commit)

    def the_check(linter, args):
        try:
            changed_file = changed_files[linter.filename]
        except KeyError:
            return

        old_content = changed_file.data_stream.read().decode("utf-8")
        apply_copyright_check(linter, old_content)

    return the_check


def main():
    m = LintMain()
    with m.execute() as ctx:
        ctx.add_check(check_copyright())


if __name__ == "__main__":
    main()

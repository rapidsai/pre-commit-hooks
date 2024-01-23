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

import contextlib
import datetime
import os.path
import tempfile
from unittest.mock import Mock, patch

import git
import pytest
from freezegun import freeze_time

from rapids_pre_commit_hooks import copyright
from rapids_pre_commit_hooks.lint import Linter


def test_match_copyright():
    CONTENT = r"""
Copyright (c) 2024 NVIDIA CORPORATION
Copyright (c) 2021-2024 NVIDIA CORPORATION
# Copyright 2021,  NVIDIA Corporation and affiliates
"""

    re_matches = copyright.match_copyright(CONTENT)
    matches = [
        {
            "span": match.span(),
            "years": match.span("years"),
            "first_year": match.span("first_year"),
            "last_year": match.span("last_year"),
        }
        for match in re_matches
    ]
    assert matches == [
        {
            "span": (1, 38),
            "years": (15, 19),
            "first_year": (15, 19),
            "last_year": (-1, -1),
        },
        {
            "span": (39, 81),
            "years": (53, 62),
            "first_year": (53, 57),
            "last_year": (58, 62),
        },
        {
            "span": (84, 119),
            "years": (94, 98),
            "first_year": (94, 98),
            "last_year": (-1, -1),
        },
    ]


def test_strip_copyright():
    CONTENT = r"""
This is a line before the first copyright statement
Copyright (c) 2024 NVIDIA CORPORATION
This is a line between the first two copyright statements
Copyright (c) 2021-2024 NVIDIA CORPORATION
This is a line between the next two copyright statements
# Copyright 2021,  NVIDIA Corporation and affiliates
This is a line after the last copyright statement
"""
    matches = copyright.match_copyright(CONTENT)
    stripped = copyright.strip_copyright(CONTENT, matches)
    assert stripped == [
        "\nThis is a line before the first copyright statement\n",
        "\nThis is a line between the first two copyright statements\n",
        "\nThis is a line between the next two copyright statements\n# ",
        " and affiliates\nThis is a line after the last copyright statement\n",
    ]

    stripped = copyright.strip_copyright("No copyright here", [])
    assert stripped == ["No copyright here"]


@freeze_time("2024-01-18")
def test_apply_copyright_check():
    def run_apply_copyright_check(old_content, new_content):
        linter = Linter("file.txt", new_content)
        copyright.apply_copyright_check(linter, old_content)
        return linter

    expected_linter = Linter("file.txt", "No copyright notice")
    expected_linter.add_warning((0, 0), "no copyright notice found")

    linter = run_apply_copyright_check(None, "No copyright notice")
    assert linter.warnings == expected_linter.warnings

    linter = run_apply_copyright_check("No copyright notice", "No copyright notice")
    assert linter.warnings == []

    OLD_CONTENT = r"""
Copyright (c) 2021-2023 NVIDIA CORPORATION
Copyright (c) 2023 NVIDIA CORPORATION
Copyright (c) 2024 NVIDIA CORPORATION
Copyright (c) 2025 NVIDIA CORPORATION
This file has not been changed
"""
    linter = run_apply_copyright_check(OLD_CONTENT, OLD_CONTENT)
    assert linter.warnings == []

    NEW_CONTENT = r"""
Copyright (c) 2021-2023 NVIDIA CORPORATION
Copyright (c) 2023 NVIDIA CORPORATION
Copyright (c) 2024 NVIDIA CORPORATION
Copyright (c) 2025 NVIDIA CORPORATION
This file has been changed
"""
    expected_linter = Linter("file.txt", NEW_CONTENT)
    expected_linter.add_warning((15, 24), "copyright is out of date").add_replacement(
        (1, 43), "Copyright (c) 2021-2024, NVIDIA CORPORATION"
    )
    expected_linter.add_warning((58, 62), "copyright is out of date").add_replacement(
        (44, 81), "Copyright (c) 2023-2024, NVIDIA CORPORATION"
    )

    linter = run_apply_copyright_check(OLD_CONTENT, NEW_CONTENT)
    assert linter.warnings == expected_linter.warnings

    expected_linter = Linter("file.txt", NEW_CONTENT)
    expected_linter.add_warning((15, 24), "copyright is out of date").add_replacement(
        (1, 43), "Copyright (c) 2021-2024, NVIDIA CORPORATION"
    )
    expected_linter.add_warning((58, 62), "copyright is out of date").add_replacement(
        (44, 81), "Copyright (c) 2023-2024, NVIDIA CORPORATION"
    )

    linter = run_apply_copyright_check(None, NEW_CONTENT)
    assert linter.warnings == expected_linter.warnings

    NEW_CONTENT = r"""
Copyright (c) 2021-2024 NVIDIA CORPORATION
Copyright (c) 2023 NVIDIA CORPORATION
Copyright (c) 2024 NVIDIA CORPORATION
Copyright (c) 2025 NVIDIA Corporation
This file has not been changed
"""
    expected_linter = Linter("file.txt", NEW_CONTENT)
    expected_linter.add_warning(
        (15, 24), "copyright is not out of date and should not be updated"
    ).add_replacement((1, 43), "Copyright (c) 2021-2023 NVIDIA CORPORATION")
    expected_linter.add_warning(
        (120, 157), "copyright is not out of date and should not be updated"
    ).add_replacement((120, 157), "Copyright (c) 2025 NVIDIA CORPORATION")

    linter = run_apply_copyright_check(OLD_CONTENT, NEW_CONTENT)
    assert linter.warnings == expected_linter.warnings


@pytest.fixture
def git_repo():
    with tempfile.TemporaryDirectory() as d:
        repo = git.Repo.init(d)
        with repo.config_writer() as w:
            w.set_value("user", "name", "RAPIDS Test Fixtures")
            w.set_value("user", "email", "testfixtures@rapids.ai")
        yield repo


def test_get_target_branch(git_repo):
    master = git_repo.head.reference

    with open(os.path.join(git_repo.working_tree_dir, "file.txt"), "w") as f:
        f.write("File\n")
    git_repo.index.add(["file.txt"])
    git_repo.index.commit("Initial commit")
    with pytest.warns(
        copyright.NoTargetBranchWarning,
        match=r"^Could not determine target branch[.] Try setting the TARGET_BRANCH or "
        r"RAPIDS_BASE_BRANCH environment variable, or setting the rapidsai.baseBranch "
        r"configuration option[.]$",
    ):
        assert copyright.get_target_branch(git_repo) is None

    branch_24_02 = git_repo.create_head("branch-24.02")
    assert copyright.get_target_branch(git_repo) == branch_24_02

    branch_24_04 = git_repo.create_head("branch-24.04")
    branch_24_03 = git_repo.create_head("branch-24.03")
    assert copyright.get_target_branch(git_repo) == branch_24_04

    branch_25_01 = git_repo.create_head("branch-25.01")
    assert copyright.get_target_branch(git_repo) == branch_25_01

    with git_repo.config_writer() as w:
        w.set_value("rapidsai", "baseBranch", "nonexistent")
    assert copyright.get_target_branch(git_repo) == branch_25_01

    with git_repo.config_writer() as w:
        w.set_value("rapidsai", "baseBranch", "branch-24.03")
    assert copyright.get_target_branch(git_repo) == branch_24_03

    with patch.dict("os.environ", {"GITHUB_BASE_REF": "nonexistent"}):
        assert copyright.get_target_branch(git_repo) == branch_24_03

    with patch.dict("os.environ", {"GITHUB_BASE_REF": "master"}):
        assert copyright.get_target_branch(git_repo) == master

    with patch.dict(
        "os.environ", {"GITHUB_BASE_REF": "master", "RAPIDS_BASE_BRANCH": "nonexistent"}
    ):
        assert copyright.get_target_branch(git_repo) == master

    with patch.dict(
        "os.environ",
        {"GITHUB_BASE_REF": "master", "RAPIDS_BASE_BRANCH": "branch-24.02"},
    ):
        assert copyright.get_target_branch(git_repo) == branch_24_02

    with patch.dict(
        "os.environ",
        {
            "GITHUB_BASE_REF": "master",
            "RAPIDS_BASE_BRANCH": "branch-24.02",
            "TARGET_BRANCH": "nonexistent",
        },
    ):
        assert copyright.get_target_branch(git_repo) == branch_24_02

    with patch.dict(
        "os.environ",
        {
            "GITHUB_BASE_REF": "master",
            "RAPIDS_BASE_BRANCH": "branch-24.02",
            "TARGET_BRANCH": "branch-24.04",
        },
    ):
        assert copyright.get_target_branch(git_repo) == branch_24_04
        with pytest.warns(
            copyright.NoSuchBranchWarning,
            match=r'^--target-branch: branch name "nonexistent" does not exist\.$',
        ):
            assert copyright.get_target_branch(git_repo, "nonexistent") == branch_24_04
        assert copyright.get_target_branch(git_repo, "master") == master


def test_get_target_branch_upstream_commit(git_repo):
    def fn(repo, filename):
        return os.path.join(repo.working_tree_dir, filename)

    def write_file(repo, filename, contents):
        with open(fn(repo, filename), "w") as f:
            f.write(contents)

    # fmt: off
    with tempfile.TemporaryDirectory() as remote_dir_1, \
         tempfile.TemporaryDirectory() as remote_dir_2:
        # fmt: on
        remote_repo_1 = git.Repo.init(remote_dir_1)
        remote_repo_2 = git.Repo.init(remote_dir_2)

        remote_1_master = remote_repo_1.head.reference

        write_file(remote_repo_1, "file1.txt", "File 1")
        write_file(remote_repo_1, "file2.txt", "File 2")
        write_file(remote_repo_1, "file3.txt", "File 3")
        write_file(remote_repo_1, "file4.txt", "File 4")
        write_file(remote_repo_1, "file5.txt", "File 5")
        write_file(remote_repo_1, "file6.txt", "File 6")
        remote_repo_1.index.add(
            [
                "file1.txt",
                "file2.txt",
                "file3.txt",
                "file4.txt",
                "file5.txt",
                "file6.txt",
            ]
        )
        remote_repo_1.index.commit("Initial commit")

        remote_1_branch_1 = remote_repo_1.create_head(
            "branch-1", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_1
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file1.txt", "File 1 modified")
        remote_repo_1.index.add(["file1.txt"])
        remote_repo_1.index.commit("Update file1.txt")

        remote_1_branch_2 = remote_repo_1.create_head(
            "branch-2", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_2
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file2.txt", "File 2 modified")
        remote_repo_1.index.add(["file2.txt"])
        remote_repo_1.index.commit("Update file2.txt")

        remote_1_branch_3 = remote_repo_1.create_head(
            "branch-3", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_3
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file3.txt", "File 3 modified")
        remote_repo_1.index.add(["file3.txt"])
        remote_repo_1.index.commit(
            "Update file3.txt",
            commit_date=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        )

        remote_1_branch_4 = remote_repo_1.create_head(
            "branch-4", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_4
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file4.txt", "File 4 modified")
        remote_repo_1.index.add(["file4.txt"])
        remote_repo_1.index.commit(
            "Update file4.txt",
            commit_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        )

        remote_2_1 = remote_repo_2.create_remote("remote-1", remote_dir_1)
        remote_2_1.fetch(["master"])
        remote_2_master = remote_repo_2.create_head("master", remote_2_1.refs["master"])

        remote_2_branch_3 = remote_repo_2.create_head(
            "branch-3", remote_2_master.commit
        )
        remote_repo_2.head.reference = remote_2_branch_3
        remote_repo_2.head.reset(index=True, working_tree=True)
        write_file(remote_repo_2, "file3.txt", "File 3 modified")
        remote_repo_2.index.add(["file3.txt"])
        remote_repo_2.index.commit(
            "Update file3.txt",
            commit_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        )

        remote_2_branch_4 = remote_repo_2.create_head(
            "branch-4", remote_2_master.commit
        )
        remote_repo_2.head.reference = remote_2_branch_4
        remote_repo_2.head.reset(index=True, working_tree=True)
        write_file(remote_repo_2, "file4.txt", "File 4 modified")
        remote_repo_2.index.add(["file4.txt"])
        remote_repo_2.index.commit(
            "Update file4.txt",
            commit_date=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        )

        remote_2_branch_5 = remote_repo_2.create_head(
            "branch-5", remote_2_master.commit
        )
        remote_repo_2.head.reference = remote_2_branch_5
        remote_repo_2.head.reset(index=True, working_tree=True)
        write_file(remote_repo_2, "file5.txt", "File 5 modified")
        remote_repo_2.index.add(["file5.txt"])
        remote_repo_2.index.commit("Update file5.txt")

        remote_1 = git_repo.create_remote("unconventional/remote/name/1", remote_dir_1)
        remote_1.fetch(["master", "branch-1", "branch-2", "branch-3", "branch-4"])
        remote_2 = git_repo.create_remote("unconventional/remote/name/2", remote_dir_2)
        remote_2.fetch(["branch-3", "branch-4", "branch-5"])

        main = git_repo.create_head("main", remote_1.refs["master"])

        branch_1 = git_repo.create_head("branch-1-renamed", remote_1.refs["master"])
        with branch_1.config_writer() as w:
            w.set_value("remote", "unconventional/remote/name/1")
            w.set_value("merge", "branch-1")
        git_repo.head.reference = branch_1
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove("file1.txt", working_tree=True)
        git_repo.index.commit("Remove file1.txt")

        branch_2 = git_repo.create_head("branch-2", remote_1.refs["master"])
        git_repo.head.reference = branch_2
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove("file2.txt", working_tree=True)
        git_repo.index.commit("Remove file2.txt")

        branch_3 = git_repo.create_head("branch-3", remote_1.refs["master"])
        git_repo.head.reference = branch_3
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove("file3.txt", working_tree=True)
        git_repo.index.commit("Remove file3.txt")

        branch_4 = git_repo.create_head("branch-4", remote_1.refs["master"])
        git_repo.head.reference = branch_4
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove(["file4.txt"], working_tree=True)
        git_repo.index.commit("Remove file4.txt")

        branch_5 = git_repo.create_head("branch-5", remote_1.refs["master"])
        git_repo.head.reference = branch_5
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove(["file5.txt"], working_tree=True)
        git_repo.index.commit("Remove file5.txt")

        branch_6 = git_repo.create_head("branch-6", remote_1.refs["master"])
        git_repo.head.reference = branch_6
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove(["file6.txt"], working_tree=True)
        git_repo.index.commit("Remove file6.txt")

        git_repo.head.reference = main
        git_repo.head.reset(index=True, working_tree=True)

        def mock_target_branch(branch):
            return patch(
                "rapids_pre_commit_hooks.copyright.get_target_branch",
                Mock(return_value=branch),
            )

        with mock_target_branch(branch_1):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo)
                == remote_1.refs["branch-1"].commit
            )

        with mock_target_branch(branch_2):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo)
                == remote_1.refs["branch-2"].commit
            )

        with mock_target_branch(branch_3):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo)
                == remote_1.refs["branch-3"].commit
            )

        with mock_target_branch(branch_4):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo)
                == remote_2.refs["branch-4"].commit
            )

        with mock_target_branch(branch_5):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo)
                == remote_2.refs["branch-5"].commit
            )

        with mock_target_branch(branch_6):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo) == branch_6.commit
            )

        with mock_target_branch(None):
            assert copyright.get_target_branch_upstream_commit(git_repo) == main.commit


def test_get_changed_files(git_repo):
    def mock_os_walk(top):
        return patch(
            "os.walk",
            Mock(
                return_value=(
                    (
                        "."
                        if (rel := os.path.relpath(dirpath, top)) == "."
                        else os.path.join(".", rel),
                        dirnames,
                        filenames,
                    )
                    for dirpath, dirnames, filenames in os.walk(top)
                )
            ),
        )

    with tempfile.TemporaryDirectory() as non_git_dir, patch(
        "os.getcwd", Mock(return_value=non_git_dir)
    ), mock_os_walk(non_git_dir):
        with open(os.path.join(non_git_dir, "top.txt"), "w") as f:
            f.write("Top file\n")
        os.mkdir(os.path.join(non_git_dir, "subdir1"))
        os.mkdir(os.path.join(non_git_dir, "subdir1/subdir2"))
        with open(os.path.join(non_git_dir, "subdir1", "subdir2", "sub.txt"), "w") as f:
            f.write("Subdir file\n")
        assert copyright.get_changed_files(Mock(target_branch=None)) == {
            "top.txt": None,
            "subdir1/subdir2/sub.txt": None,
        }

    def fn(filename):
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename, contents):
        with open(fn(filename), "w") as f:
            f.write(contents)

    def file_contents(verbed):
        return f"This file will be {verbed}\n" * 100

    write_file("untouched.txt", file_contents("untouched"))
    write_file("copied.txt", file_contents("copied"))
    write_file("modified_and_copied.txt", file_contents("modified and copied"))
    write_file("copied_and_modified.txt", file_contents("copied and modified"))
    write_file("deleted.txt", file_contents("deleted"))
    write_file("renamed.txt", file_contents("renamed"))
    write_file("modified_and_renamed.txt", file_contents("modified and renamed"))
    write_file("modified.txt", file_contents("modified"))
    write_file("chmodded.txt", file_contents("chmodded"))
    write_file("untracked.txt", file_contents("untracked"))
    git_repo.index.add(
        [
            "untouched.txt",
            "copied.txt",
            "modified_and_copied.txt",
            "copied_and_modified.txt",
            "deleted.txt",
            "renamed.txt",
            "modified_and_renamed.txt",
            "modified.txt",
            "chmodded.txt",
        ]
    )

    with patch("os.getcwd", Mock(return_value=git_repo.working_tree_dir)), mock_os_walk(
        git_repo.working_tree_dir
    ), patch(
        "rapids_pre_commit_hooks.copyright.get_target_branch_upstream_commit",
        Mock(return_value=None),
    ):
        assert copyright.get_changed_files(Mock(target_branch=None)) == {
            "untouched.txt": None,
            "copied.txt": None,
            "modified_and_copied.txt": None,
            "copied_and_modified.txt": None,
            "deleted.txt": None,
            "renamed.txt": None,
            "modified_and_renamed.txt": None,
            "modified.txt": None,
            "chmodded.txt": None,
            "untracked.txt": None,
        }

    git_repo.index.commit("Initial commit")

    # Ensure that diff is done against merge base, not branch tip
    git_repo.index.remove(["modified.txt"], working_tree=True)
    git_repo.index.commit("Remove modified.txt")

    pr_branch = git_repo.create_head("pr", "HEAD~")
    git_repo.head.reference = pr_branch
    git_repo.head.reset(index=True, working_tree=True)

    write_file("copied_2.txt", file_contents("copied"))
    git_repo.index.remove(
        ["deleted.txt", "modified_and_renamed.txt"], working_tree=True
    )
    git_repo.index.move(["renamed.txt", "renamed_2.txt"])
    write_file(
        "modified.txt", file_contents("modified") + "This file has been modified\n"
    )
    os.chmod(fn("chmodded.txt"), 0o755)
    write_file("untouched.txt", file_contents("untouched") + "Oops\n")
    write_file("added.txt", file_contents("added"))
    write_file("added_and_deleted.txt", file_contents("added and deleted"))
    write_file(
        "modified_and_copied.txt",
        file_contents("modified and copied") + "This file has been modified\n",
    )
    write_file("modified_and_copied_2.txt", file_contents("modified and copied"))
    write_file(
        "copied_and_modified_2.txt",
        file_contents("copied and modified") + "This file has been modified\n",
    )
    write_file(
        "modified_and_renamed_2.txt",
        file_contents("modified and renamed") + "This file has been modified\n",
    )
    git_repo.index.add(
        [
            "untouched.txt",
            "added.txt",
            "added_and_deleted.txt",
            "modified_and_copied.txt",
            "modified_and_copied_2.txt",
            "copied_and_modified_2.txt",
            "copied_2.txt",
            "modified_and_renamed_2.txt",
            "modified.txt",
            "chmodded.txt",
        ]
    )
    write_file("untouched.txt", file_contents("untouched"))
    os.unlink(fn("added_and_deleted.txt"))

    target_branch = git_repo.heads["master"]
    merge_base = git_repo.merge_base(target_branch, "HEAD")[0]
    old_files = {
        blob.path: blob
        for blob in merge_base.tree.traverse(lambda b, _: isinstance(b, git.Blob))
    }

    # Truly need to be checked
    changed = {
        "added.txt": None,
        "untracked.txt": None,
        "modified_and_renamed_2.txt": "modified_and_renamed.txt",
        "modified.txt": "modified.txt",
        "copied_and_modified_2.txt": "copied_and_modified.txt",
        "modified_and_copied.txt": "modified_and_copied.txt",
    }

    # Superfluous, but harmless because the content is identical
    superfluous = {
        "chmodded.txt": "chmodded.txt",
        "modified_and_copied_2.txt": "modified_and_copied.txt",
        "copied_2.txt": "copied.txt",
        "renamed_2.txt": "renamed.txt",
    }

    with patch("os.getcwd", Mock(return_value=git_repo.working_tree_dir)), patch(
        "rapids_pre_commit_hooks.copyright.get_target_branch_upstream_commit",
        Mock(return_value=target_branch.commit),
    ):
        changed_files = copyright.get_changed_files(Mock(target_branch=None))
    assert {
        path: old_blob.path if old_blob else None
        for path, old_blob in changed_files.items()
    } == changed | superfluous

    for new, old in changed.items():
        if old:
            with open(fn(new), "rb") as f:
                new_contents = f.read()
            old_contents = old_files[old].data_stream.read()
            assert new_contents != old_contents
            assert changed_files[new].data_stream.read() == old_contents

    for new, old in superfluous.items():
        if old:
            with open(fn(new), "rb") as f:
                new_contents = f.read()
            old_contents = old_files[old].data_stream.read()
            assert new_contents == old_contents
            assert changed_files[new].data_stream.read() == old_contents


def test_find_blob(git_repo):
    with open(os.path.join(git_repo.working_tree_dir, "top.txt"), "w"):
        pass
    os.mkdir(os.path.join(git_repo.working_tree_dir, "sub1"))
    os.mkdir(os.path.join(git_repo.working_tree_dir, "sub1", "sub2"))
    with open(os.path.join(git_repo.working_tree_dir, "sub1", "sub2", "sub.txt"), "w"):
        pass
    git_repo.index.add(["top.txt", "sub1/sub2/sub.txt"])
    git_repo.index.commit("Initial commit")

    assert copyright.find_blob(git_repo.head.commit.tree, "top.txt").path == "top.txt"
    assert (
        copyright.find_blob(git_repo.head.commit.tree, "sub1/sub2/sub.txt").path
        == "sub1/sub2/sub.txt"
    )
    assert copyright.find_blob(git_repo.head.commit.tree, "nonexistent.txt") is None


def test_get_file_last_modified(git_repo):
    def fn(filename):
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename, contents):
        with open(fn(filename), "w") as f:
            f.write(contents)

    def expected_return_value(commit, filename):
        return (commit, copyright.find_blob(commit.tree, filename))

    @contextlib.contextmanager
    def no_match_copyright():
        with patch(
            "rapids_pre_commit_hooks.copyright.match_copyright", Mock()
        ) as match_copyright, patch(
            "rapids_pre_commit_hooks.copyright.strip_copyright", Mock()
        ) as strip_copyright:
            yield
            match_copyright.assert_not_called()
            strip_copyright.assert_not_called()

    write_file("file1.txt", "File 1")
    git_repo.index.add("file1.txt")
    git_repo.index.commit("Initial commit")
    with no_match_copyright():
        assert copyright.get_file_last_modified(
            git_repo.head.commit, "file1.txt"
        ) == expected_return_value(git_repo.head.commit, "file1.txt")

    write_file("file2.txt", "File 2")
    git_repo.index.add("file2.txt")
    git_repo.index.commit("Add file2.txt")
    with no_match_copyright():
        assert copyright.get_file_last_modified(
            git_repo.head.commit, "file1.txt"
        ) == expected_return_value(git_repo.head.commit.parents[0], "file1.txt")
        assert copyright.get_file_last_modified(
            git_repo.head.commit.parents[0], "file1.txt"
        ) == expected_return_value(git_repo.head.commit.parents[0], "file1.txt")
        assert copyright.get_file_last_modified(
            git_repo.head.commit, "file2.txt"
        ) == expected_return_value(git_repo.head.commit, "file2.txt")
        assert copyright.get_file_last_modified(
            git_repo.head.commit.parents[0], "file2.txt"
        ) == (None, None)

    git_repo.index.remove("file1.txt", working_tree=True)
    write_file("file1_2.txt", "File 1")
    write_file("file2_2.txt", "File 2")
    git_repo.index.add(["file1_2.txt", "file2_2.txt"])
    git_repo.index.commit("Rename and copy")
    with no_match_copyright():
        assert copyright.get_file_last_modified(
            git_repo.head.commit, "file1_2.txt"
        ) == expected_return_value(
            git_repo.head.commit.parents[0].parents[0], "file1.txt"
        )
        assert copyright.get_file_last_modified(
            git_repo.head.commit, "file2_2.txt"
        ) == expected_return_value(git_repo.head.commit.parents[0], "file2.txt")

    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023 NVIDIA CORPORATION
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    git_repo.index.commit("Add copyrighted file")
    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    git_repo.index.commit("Update copyright")
    assert copyright.get_file_last_modified(
        git_repo.head.commit, "copyright.txt"
    ) == expected_return_value(git_repo.head.commit.parents[0], "copyright.txt")

    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
New content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    git_repo.index.commit("New contents")
    assert copyright.get_file_last_modified(
        git_repo.head.commit, "copyright.txt"
    ) == expected_return_value(git_repo.head.commit, "copyright.txt")

    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
Updated content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    commit_1 = git_repo.index.commit(
        "Update contents",
        commit_date=datetime.datetime(2024, 1, 23, tzinfo=datetime.timezone.utc),
    )
    commit_2 = git_repo.index.commit(
        "Update contents",
        commit_date=datetime.datetime(2024, 1, 24, tzinfo=datetime.timezone.utc),
        parent_commits=commit_1.parents,
    )
    git_repo.index.commit("Merge", parent_commits=[commit_1, commit_2])
    assert copyright.get_file_last_modified(
        git_repo.head.commit, "copyright.txt"
    ) == expected_return_value(commit_2, "copyright.txt")

    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
New updated content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    commit_1 = git_repo.index.commit(
        "Update contents again",
        commit_date=datetime.datetime(2024, 1, 24, tzinfo=datetime.timezone.utc),
    )
    commit_2 = git_repo.index.commit(
        "Update contents again",
        commit_date=datetime.datetime(2024, 1, 23, tzinfo=datetime.timezone.utc),
        parent_commits=commit_1.parents,
    )
    git_repo.index.commit("Merge", parent_commits=[commit_1, commit_2])
    assert copyright.get_file_last_modified(
        git_repo.head.commit, "copyright.txt"
    ) == expected_return_value(commit_1, "copyright.txt")

    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
Old content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    git_repo.index.commit(
        "Old content",
        commit_date=datetime.datetime(2024, 1, 23, tzinfo=datetime.timezone.utc),
    )
    old_commit = git_repo.head.commit
    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
New content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    new_commit = git_repo.index.commit(
        "New content",
        commit_date=datetime.datetime(2024, 1, 24, tzinfo=datetime.timezone.utc),
    )
    git_repo.index.commit(
        "Merge",
        commit_date=datetime.datetime(2024, 1, 25, tzinfo=datetime.timezone.utc),
        parent_commits=[git_repo.head.commit, old_commit],
    )
    assert copyright.get_file_last_modified(
        git_repo.head.commit, "copyright.txt"
    ) == expected_return_value(new_commit, "copyright.txt")

    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
Old content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    git_repo.index.commit(
        "Old content",
        commit_date=datetime.datetime(2024, 1, 23, tzinfo=datetime.timezone.utc),
    )
    old_commit = git_repo.head.commit
    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023 NVIDIA CORPORATION
New content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    new_commit = git_repo.index.commit(
        "New content",
        commit_date=datetime.datetime(2024, 1, 25, tzinfo=datetime.timezone.utc),
    )
    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
New content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    git_repo.index.commit(
        "Merge",
        commit_date=datetime.datetime(2024, 1, 24, tzinfo=datetime.timezone.utc),
        parent_commits=[git_repo.head.commit, old_commit],
    )
    assert copyright.get_file_last_modified(
        git_repo.head.commit, "copyright.txt"
    ) == expected_return_value(new_commit, "copyright.txt")

    write_file(
        "copyright.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023 NVIDIA CORPORATION
Copyrighted content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    git_repo.index.commit("Add copyrighted content")
    write_file(
        "copyright2.txt",
        """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
Copyrighted content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright2.txt")
    git_repo.index.commit("Copy copyrighted file")
    assert copyright.get_file_last_modified(
        git_repo.head.commit, "copyright2.txt"
    ) == expected_return_value(git_repo.head.commit.parents[0], "copyright.txt")

    git_repo.index.remove("copyright2.txt", working_tree=True)
    write_file(
        "copyright.txt",
        f"""
Beginning of copyrighted file
Copyright (c) 2023 NVIDIA CORPORATION
{'''Lots of content
'''} * 100
End of copyrighted file
""",
    )
    git_repo.index.add("copyright.txt")
    git_repo.index.commit("Add copyrighted content")
    write_file(
        "copyright2.txt",
        f"""
Beginning of copyrighted file
Copyright (c) 2023 NVIDIA CORPORATION
{'''Lots of content
'''} * 100
More content
End of copyrighted file
""",
    )
    git_repo.index.add("copyright2.txt")
    git_repo.index.commit("Copy and modify copyrighted file")
    assert copyright.get_file_last_modified(
        git_repo.head.commit, "copyright2.txt"
    ) == expected_return_value(git_repo.head.commit, "copyright2.txt")


def test_apply_batch_copyright_check(git_repo):
    def fn(filename):
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename, content):
        with open(fn(filename), "w") as f:
            f.write(content)

    CONTENT = """
Beginning of copyrighted file
Copyright (c) 2023 NVIDIA CORPORATION
End of copyrighted file
"""
    write_file("file.txt", CONTENT)
    git_repo.index.add("file.txt")
    git_repo.index.commit(
        "Initial commit",
        commit_date=datetime.datetime(2023, 2, 1, tzinfo=datetime.timezone.utc),
    )

    linter = Linter("file.txt", CONTENT)
    copyright.apply_batch_copyright_check(git_repo, linter)
    assert linter.warnings == []

    linter = Linter("file.txt", CONTENT + "Oops")
    with pytest.warns(
        copyright.ConflictingFilesWarning,
        match=r'^File "file[.]txt" differs from Git history[.] Not running batch '
        r"copyright update[.]$",
    ):
        copyright.apply_batch_copyright_check(git_repo, linter)
    assert linter.warnings == []

    linter = Linter("file2.txt", CONTENT + "Oops")
    with pytest.warns(
        copyright.ConflictingFilesWarning,
        match=r'^File "file2[.]txt" not in Git history[.] Not running batch copyright '
        r"update[.]$",
    ):
        copyright.apply_batch_copyright_check(git_repo, linter)
    assert linter.warnings == []

    CONTENT = """
Beginning of copyrighted file
Copyright (c) 2023 NVIDIA CORPORATION
New content
End of copyrighted file
"""
    write_file("file.txt", CONTENT)
    git_repo.index.add("file.txt")
    git_repo.index.commit(
        "Add content",
        commit_date=datetime.datetime(2024, 2, 1, tzinfo=datetime.timezone.utc),
    )

    expected_linter = Linter("file.txt", CONTENT)
    expected_linter.add_warning((45, 49), "copyright is out of date").add_replacement(
        (31, 68), "Copyright (c) 2023-2024, NVIDIA CORPORATION"
    )

    linter = Linter("file.txt", CONTENT)
    copyright.apply_batch_copyright_check(git_repo, linter)
    assert linter.warnings == expected_linter.warnings

    CONTENT = """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
New content
End of copyrighted file
"""
    write_file("file.txt", CONTENT)
    git_repo.index.add("file.txt")
    git_repo.index.commit(
        "Add content",
        commit_date=datetime.datetime(2024, 2, 2, tzinfo=datetime.timezone.utc),
    )

    linter = Linter("file.txt", CONTENT)
    copyright.apply_batch_copyright_check(git_repo, linter)
    assert linter.warnings == []

    CONTENT = """
Beginning of copyrighted file
Copyright (c) 2023-2024 NVIDIA CORPORATION
Newer content
End of copyrighted file
"""
    write_file("file.txt", CONTENT)
    git_repo.index.add("file.txt")
    git_repo.index.commit(
        "Update copyright and content",
        commit_date=datetime.datetime(2024, 2, 3, tzinfo=datetime.timezone.utc),
    )

    linter = Linter("file.txt", CONTENT)
    copyright.apply_batch_copyright_check(git_repo, linter)
    assert linter.warnings == []

    CONTENT = """
Beginning of copyrighted file
Copyright (c) 2023-2025 NVIDIA CORPORATION
Newer content
End of copyrighted file
"""
    write_file("file.txt", CONTENT)
    git_repo.index.add("file.txt")
    git_repo.index.commit(
        "Update copyright again",
        commit_date=datetime.datetime(2025, 2, 1, tzinfo=datetime.timezone.utc),
    )

    expected_linter = Linter("file.txt", CONTENT)
    expected_linter.add_warning(
        (45, 54), "copyright is not out of date and should not be updated"
    ).add_replacement((31, 73), "Copyright (c) 2023-2024 NVIDIA CORPORATION")

    linter = Linter("file.txt", CONTENT)
    copyright.apply_batch_copyright_check(git_repo, linter)
    assert linter.warnings == expected_linter.warnings

    CONTENT = """
Beginning of copyrighted file
Copyright (c) 2023-2025 NVIDIA CORPORATION
Even newer content
End of copyrighted file
"""
    write_file("file.txt", CONTENT)
    git_repo.index.add("file.txt")
    git_repo.index.commit(
        "Update copyright again",
        commit_date=datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc),
    )

    expected_linter = Linter("file.txt", CONTENT)
    expected_linter.add_warning((45, 54), "copyright is out of date").add_replacement(
        (31, 73), "Copyright (c) 2023-2026, NVIDIA CORPORATION"
    )

    linter = Linter("file.txt", CONTENT)
    copyright.apply_batch_copyright_check(git_repo, linter)
    assert linter.warnings == expected_linter.warnings


@freeze_time("2024-01-18")
def test_check_copyright(git_repo):
    def fn(filename):
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename, contents):
        with open(fn(filename), "w") as f:
            f.write(contents)

    def file_contents(num):
        return rf"""
Copyright (c) 2021-2023 NVIDIA CORPORATION
File {num}
"""

    def file_contents_modified(num):
        return rf"""
Copyright (c) 2021-2023 NVIDIA CORPORATION
File {num} modified
"""

    write_file("file1.txt", file_contents(1))
    write_file("file2.txt", file_contents(2))
    write_file("file3.txt", file_contents(3))
    write_file("file4.txt", file_contents(4))
    git_repo.index.add(["file1.txt", "file2.txt", "file3.txt", "file4.txt"])
    git_repo.index.commit("Initial commit")

    branch_1 = git_repo.create_head("branch-1", "master")
    git_repo.head.reference = branch_1
    git_repo.head.reset(index=True, working_tree=True)
    write_file("file1.txt", file_contents_modified(1))
    git_repo.index.add(["file1.txt"])
    git_repo.index.commit("Update file1.txt")

    branch_2 = git_repo.create_head("branch-2", "master")
    git_repo.head.reference = branch_2
    git_repo.head.reset(index=True, working_tree=True)
    write_file("file2.txt", file_contents_modified(2))
    git_repo.index.add(["file2.txt"])
    git_repo.index.commit("Update file2.txt")

    pr = git_repo.create_head("pr", "branch-1")
    git_repo.head.reference = pr
    git_repo.head.reset(index=True, working_tree=True)
    write_file("file3.txt", file_contents_modified(3))
    git_repo.index.add(["file3.txt"])
    git_repo.index.commit("Update file3.txt")
    write_file("file4.txt", file_contents_modified(4))
    git_repo.index.add(["file4.txt"])
    git_repo.index.commit("Update file4.txt")
    git_repo.index.move(["file2.txt", "file5.txt"])
    git_repo.index.commit("Rename file2.txt to file5.txt")

    write_file("file6.txt", file_contents(6))

    def mock_repo_cwd():
        return patch("os.getcwd", Mock(return_value=git_repo.working_tree_dir))

    def mock_target_branch_upstream_commit(target_branch):
        def func(repo, target_branch_arg):
            assert target_branch == target_branch_arg
            return repo.heads[target_branch].commit

        return patch(
            "rapids_pre_commit_hooks.copyright.get_target_branch_upstream_commit", func
        )

    def mock_apply_copyright_check():
        return patch("rapids_pre_commit_hooks.copyright.apply_copyright_check", Mock())

    @contextlib.contextmanager
    def no_apply_batch_copyright_check():
        with patch(
            "rapids_pre_commit_hooks.copyright.apply_batch_copyright_check", Mock()
        ) as apply_batch_copyright_check:
            yield
            apply_batch_copyright_check.assert_not_called()

    #############################
    # branch-1 is target branch
    #############################

    mock_args = Mock(target_branch="branch-1", batch=False)

    with mock_repo_cwd(), mock_target_branch_upstream_commit("branch-1"):
        copyright_checker = copyright.check_copyright(mock_args)

    linter = Linter("file1.txt", file_contents_modified(1))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_not_called()

    linter = Linter("file5.txt", file_contents(2))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(2))

    linter = Linter("file3.txt", file_contents_modified(3))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(3))

    linter = Linter("file4.txt", file_contents_modified(4))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(4))

    linter = Linter("file6.txt", file_contents(6))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, None)

    #############################
    # branch-2 is target branch
    #############################

    mock_args = Mock(target_branch="branch-2", batch=False)

    with mock_repo_cwd(), mock_target_branch_upstream_commit("branch-2"):
        copyright_checker = copyright.check_copyright(mock_args)

    linter = Linter("file1.txt", file_contents_modified(1))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(1))

    linter = Linter("file5.txt", file_contents(2))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(2))

    linter = Linter("file3.txt", file_contents_modified(3))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(3))

    linter = Linter("file4.txt", file_contents_modified(4))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(4))

    linter = Linter("file6.txt", file_contents(6))
    # fmt: off
    with mock_apply_copyright_check() as apply_copyright_check, \
            no_apply_batch_copyright_check():
        # fmt: on
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, None)


def test_check_copyright_batch():
    git_repo = Mock()
    with patch("git.Repo", Mock(return_value=git_repo)), patch(
        "rapids_pre_commit_hooks.copyright.apply_copyright_check", Mock()
    ) as apply_copyright_check, patch(
        "rapids_pre_commit_hooks.copyright.apply_batch_copyright_check", Mock()
    ) as apply_batch_copyright_check:
        mock_args = Mock(batch=True)
        copyright_checker = copyright.check_copyright(mock_args)
        linter = Mock()
        copyright_checker(linter, mock_args)
        apply_batch_copyright_check.assert_called_once_with(git_repo, linter)
        apply_copyright_check.assert_not_called()

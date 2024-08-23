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

import argparse
import datetime
import os.path
import tempfile
from io import BufferedReader
from textwrap import dedent
from typing import Any, TextIO, Union
from unittest.mock import Mock, patch

import git
import pytest
from freezegun import freeze_time

from rapids_pre_commit_hooks import copyright
from rapids_pre_commit_hooks.lint import Linter


def test_match_copyright():
    CONTENT = dedent(
        r"""
        Copyright (c) 2024 NVIDIA CORPORATION
        Copyright (c) 2021-2024 NVIDIA CORPORATION
        # Copyright 2021,  NVIDIA Corporation and affiliates
        """
    )

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
    CONTENT = dedent(
        r"""
        This is a line before the first copyright statement
        Copyright (c) 2024 NVIDIA CORPORATION
        This is a line between the first two copyright statements
        Copyright (c) 2021-2024 NVIDIA CORPORATION
        This is a line between the next two copyright statements
        # Copyright 2021,  NVIDIA Corporation and affiliates
        This is a line after the last copyright statement
        """
    )
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
    def run_apply_copyright_check(old_content: str, new_content: str):
        linter = Linter("file.txt", new_content)
        copyright.apply_copyright_check(linter, old_content)
        return linter

    expected_linter = Linter("file.txt", "No copyright notice")
    expected_linter.add_warning((0, 0), "no copyright notice found")

    linter = run_apply_copyright_check(None, "No copyright notice")
    assert linter.warnings == expected_linter.warnings

    linter = run_apply_copyright_check("No copyright notice", "No copyright notice")
    assert linter.warnings == []

    OLD_CONTENT = dedent(
        r"""
        Copyright (c) 2021-2023 NVIDIA CORPORATION
        Copyright (c) 2023 NVIDIA CORPORATION
        Copyright (c) 2024 NVIDIA CORPORATION
        Copyright (c) 2025 NVIDIA CORPORATION
        This file has not been changed
        """
    )
    linter = run_apply_copyright_check(OLD_CONTENT, OLD_CONTENT)
    assert linter.warnings == []

    NEW_CONTENT = dedent(
        r"""
        Copyright (c) 2021-2023 NVIDIA CORPORATION
        Copyright (c) 2023 NVIDIA CORPORATION
        Copyright (c) 2024 NVIDIA CORPORATION
        Copyright (c) 2025 NVIDIA CORPORATION
        This file has been changed
        """
    )
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

    NEW_CONTENT = dedent(
        r"""
        Copyright (c) 2021-2024 NVIDIA CORPORATION
        Copyright (c) 2023 NVIDIA CORPORATION
        Copyright (c) 2024 NVIDIA CORPORATION
        Copyright (c) 2025 NVIDIA Corporation
        This file has not been changed
        """
    )
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
def git_repo(tmp_path: os.PathLike[str]) -> git.Repo:
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as w:
        w.set_value("user", "name", "RAPIDS Test Fixtures")
        w.set_value("user", "email", "testfixtures@rapids.ai")
    return repo


def test_get_target_branch(git_repo: git.Repo):
    assert git_repo.working_tree_dir is not None

    with patch.dict("os.environ", {}, clear=True):
        args = Mock(main_branch=None, target_branch=None)

        with open(os.path.join(git_repo.working_tree_dir, "file.txt"), "w") as f:
            f.write("File\n")
        git_repo.index.add(["file.txt"])
        git_repo.index.commit("Initial commit")
        with pytest.warns(
            copyright.NoTargetBranchWarning,
            match=r"^Could not determine target branch[.] Try setting the "
            r"TARGET_BRANCH environment variable, or setting the rapidsai.baseBranch "
            r"configuration option[.]$",
        ):
            assert copyright.get_target_branch(git_repo, args) is None

        git_repo.create_head("branch-24.02")
        assert copyright.get_target_branch(git_repo, args) == "branch-24.02"

        args.main_branch = ""
        args.target_branch = ""

        git_repo.create_head("branch-24.04")
        git_repo.create_head("branch-24.03")
        assert copyright.get_target_branch(git_repo, args) == "branch-24.04"

        git_repo.create_head("branch-25.01")
        assert copyright.get_target_branch(git_repo, args) == "branch-25.01"

        args.main_branch = "main"
        assert copyright.get_target_branch(git_repo, args) == "main"

        with git_repo.config_writer() as w:
            w.set_value("rapidsai", "baseBranch", "nonexistent")
        assert copyright.get_target_branch(git_repo, args) == "nonexistent"

        with git_repo.config_writer() as w:
            w.set_value("rapidsai", "baseBranch", "branch-24.03")
        assert copyright.get_target_branch(git_repo, args) == "branch-24.03"

        with patch.dict("os.environ", {"RAPIDS_BASE_BRANCH": "nonexistent"}):
            assert copyright.get_target_branch(git_repo, args) == "nonexistent"

        with patch.dict("os.environ", {"RAPIDS_BASE_BRANCH": "master"}):
            assert copyright.get_target_branch(git_repo, args) == "master"

        with patch.dict(
            "os.environ",
            {"GITHUB_BASE_REF": "nonexistent", "RAPIDS_BASE_BRANCH": "master"},
        ):
            assert copyright.get_target_branch(git_repo, args) == "nonexistent"

        with patch.dict(
            "os.environ",
            {"GITHUB_BASE_REF": "branch-24.02", "RAPIDS_BASE_BRANCH": "master"},
        ):
            assert copyright.get_target_branch(git_repo, args) == "branch-24.02"

        with patch.dict(
            "os.environ",
            {
                "GITHUB_BASE_REF": "branch-24.02",
                "RAPIDS_BASE_BRANCH": "master",
                "TARGET_BRANCH": "nonexistent",
            },
        ):
            assert copyright.get_target_branch(git_repo, args) == "nonexistent"

        with patch.dict(
            "os.environ",
            {
                "GITHUB_BASE_REF": "branch-24.02",
                "RAPIDS_BASE_BRANCH": "master",
                "TARGET_BRANCH": "branch-24.04",
            },
        ):
            assert copyright.get_target_branch(git_repo, args) == "branch-24.04"
            args.target_branch = "nonexistent"
            assert copyright.get_target_branch(git_repo, args) == "nonexistent"
            args.target_branch = "master"
            assert copyright.get_target_branch(git_repo, args) == "master"


def test_get_target_branch_upstream_commit(git_repo: git.Repo):
    def fn(repo: git.Repo, filename: str) -> str:
        assert repo.working_tree_dir is not None
        return os.path.join(repo.working_tree_dir, filename)

    def write_file(repo: git.Repo, filename: str, contents: str):
        with open(fn(repo, filename), "w") as f:
            f.write(contents)

    def mock_target_branch(branch: Any):
        return patch(
            "rapids_pre_commit_hooks.copyright.get_target_branch",
            Mock(return_value=branch),
        )

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
        write_file(remote_repo_1, "file7.txt", "File 7")
        remote_repo_1.index.add(
            [
                "file1.txt",
                "file2.txt",
                "file3.txt",
                "file4.txt",
                "file5.txt",
                "file6.txt",
                "file7.txt",
            ]
        )
        remote_repo_1.index.commit("Initial commit")

        remote_1_branch_1 = remote_repo_1.create_head(
            "branch-1-renamed", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_1  # type: ignore
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file1.txt", "File 1 modified")
        remote_repo_1.index.add(["file1.txt"])
        remote_repo_1.index.commit(
            "Update file1.txt",
            commit_date=datetime.datetime(2024, 2, 1, tzinfo=datetime.timezone.utc),
        )

        remote_1_branch_2 = remote_repo_1.create_head(
            "branch-2", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_2  # type: ignore
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file2.txt", "File 2 modified")
        remote_repo_1.index.add(["file2.txt"])
        remote_repo_1.index.commit("Update file2.txt")

        remote_1_branch_3 = remote_repo_1.create_head(
            "branch-3", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_3  # type: ignore
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
        remote_repo_1.head.reference = remote_1_branch_4  # type: ignore
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file4.txt", "File 4 modified")
        remote_repo_1.index.add(["file4.txt"])
        remote_repo_1.index.commit(
            "Update file4.txt",
            commit_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        )

        remote_1_branch_7 = remote_repo_1.create_head(
            "branch-7", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_7  # type: ignore
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file7.txt", "File 7 modified")
        remote_repo_1.index.add(["file7.txt"])
        remote_repo_1.index.commit(
            "Update file7.txt",
            commit_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        )

        remote_2_1 = remote_repo_2.create_remote("remote-1", remote_dir_1)
        remote_2_1.fetch(["master"])
        remote_2_master = remote_repo_2.create_head("master", remote_2_1.refs["master"])

        remote_2_branch_3 = remote_repo_2.create_head(
            "branch-3", remote_2_master.commit
        )
        remote_repo_2.head.reference = remote_2_branch_3  # type: ignore
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
        remote_repo_2.head.reference = remote_2_branch_4  # type: ignore
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
        remote_repo_2.head.reference = remote_2_branch_5  # type: ignore
        remote_repo_2.head.reset(index=True, working_tree=True)
        write_file(remote_repo_2, "file5.txt", "File 5 modified")
        remote_repo_2.index.add(["file5.txt"])
        remote_repo_2.index.commit("Update file5.txt")

        with mock_target_branch(None):
            assert copyright.get_target_branch_upstream_commit(git_repo, Mock()) is None

        with mock_target_branch("branch-1"):
            assert copyright.get_target_branch_upstream_commit(git_repo, Mock()) is None

        remote_1 = git_repo.create_remote("unconventional/remote/name/1", remote_dir_1)
        remote_1.fetch([
            "master",
            "branch-1-renamed",
            "branch-2",
            "branch-3",
            "branch-4",
            "branch-7",
        ])
        remote_2 = git_repo.create_remote("unconventional/remote/name/2", remote_dir_2)
        remote_2.fetch(["branch-3", "branch-4", "branch-5"])

        main = git_repo.create_head("main", remote_1.refs["master"])

        branch_1 = git_repo.create_head("branch-1", remote_1.refs["master"])
        with branch_1.config_writer() as w:
            w.set_value("remote", "unconventional/remote/name/1")
            w.set_value("merge", "branch-1-renamed")
        git_repo.head.reference = branch_1  # type: ignore
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove("file1.txt", working_tree=True)
        git_repo.index.commit(
            "Remove file1.txt",
            commit_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        )

        branch_6 = git_repo.create_head("branch-6", remote_1.refs["master"])
        git_repo.head.reference = branch_6  # type: ignore
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove(["file6.txt"], working_tree=True)
        git_repo.index.commit("Remove file6.txt")

        branch_7 = git_repo.create_head("branch-7", remote_1.refs["master"])
        with branch_7.config_writer() as w:
            w.set_value("remote", "unconventional/remote/name/1")
            w.set_value("merge", "branch-7")
        git_repo.head.reference = branch_7  # type: ignore
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove(["file7.txt"], working_tree=True)
        git_repo.index.commit(
            "Remove file7.txt",
            commit_date=datetime.datetime(2024, 2, 1, tzinfo=datetime.timezone.utc),
        )

        git_repo.head.reference = main  # type: ignore
        git_repo.head.reset(index=True, working_tree=True)

        with mock_target_branch("branch-1"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_1.refs["branch-1-renamed"].commit
            )

        with mock_target_branch("branch-2"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_1.refs["branch-2"].commit
            )

        with mock_target_branch("branch-3"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_1.refs["branch-3"].commit
            )

        with mock_target_branch("branch-4"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_2.refs["branch-4"].commit
            )

        with mock_target_branch("branch-5"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_2.refs["branch-5"].commit
            )

        with mock_target_branch("branch-6"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == branch_6.commit
            )

        with mock_target_branch("branch-7"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == branch_7.commit
            )

        with mock_target_branch("nonexistent-branch"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == main.commit
            )

        with mock_target_branch(None):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == main.commit
            )


def test_get_changed_files(git_repo: git.Repo):
    f: Union[BufferedReader, TextIO]

    assert git_repo.working_tree_dir is not None

    def mock_os_walk(top: Union[str, os.PathLike[str]]):
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
        assert copyright.get_changed_files(Mock()) == {
            "top.txt": None,
            "subdir1/subdir2/sub.txt": None,
        }

    def fn(filename: str) -> str:
        assert git_repo.working_tree_dir is not None
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename: str, contents: str):
        with open(fn(filename), "w") as f:
            f.write(contents)

    def file_contents(verbed: str) -> str:
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
        assert copyright.get_changed_files(Mock()) == {
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
    git_repo.head.reference = pr_branch  # type: ignore
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
        blob.path: blob  # type: ignore
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
        changed_files = copyright.get_changed_files(Mock())
    assert {
        path: old_blob.path if old_blob else None
        for path, old_blob in changed_files.items()
    } == changed | superfluous

    for new, old in changed.items():
        if old:
            with open(fn(new), "rb") as f:
                new_contents = f.read()
            old_contents = old_files[old].data_stream.read()  # type: ignore
            assert new_contents != old_contents
            assert changed_files[new].data_stream.read() == old_contents  # type: ignore

    for new, old in superfluous.items():
        if old:
            with open(fn(new), "rb") as f:
                new_contents = f.read()
            old_contents = old_files[old].data_stream.read()  # type: ignore
            assert new_contents == old_contents
            assert changed_files[new].data_stream.read() == old_contents  # type: ignore


def test_get_changed_files_multiple_merge_bases(git_repo: git.Repo):
    def fn(filename: str) -> str:
        assert git_repo.working_tree_dir is not None
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename: str, contents: str):
        with open(fn(filename), "w") as f:
            f.write(contents)

    write_file("file1.txt", "File 1\n")
    write_file("file2.txt", "File 2\n")
    write_file("file3.txt", "File 3\n")
    git_repo.index.add(["file1.txt", "file2.txt", "file3.txt"])
    git_repo.index.commit("Initial commit")

    branch_1 = git_repo.create_head("branch-1", "master")
    git_repo.head.reference = branch_1  # type: ignore
    git_repo.index.reset(index=True, working_tree=True)
    write_file("file1.txt", "File 1 modified\n")
    git_repo.index.add("file1.txt")
    git_repo.index.commit(
        "Modify file1.txt",
        commit_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    )

    branch_2 = git_repo.create_head("branch-2", "master")
    git_repo.head.reference = branch_2  # type: ignore
    git_repo.index.reset(index=True, working_tree=True)
    write_file("file2.txt", "File 2 modified\n")
    git_repo.index.add("file2.txt")
    git_repo.index.commit(
        "Modify file2.txt",
        commit_date=datetime.datetime(2024, 2, 1, tzinfo=datetime.timezone.utc),
    )

    branch_1_2 = git_repo.create_head("branch-1-2", "master")
    git_repo.head.reference = branch_1_2  # type: ignore
    git_repo.index.reset(index=True, working_tree=True)
    write_file("file1.txt", "File 1 modified\n")
    write_file("file2.txt", "File 2 modified\n")
    git_repo.index.add(["file1.txt", "file2.txt"])
    git_repo.index.commit(
        "Merge branches branch-1 and branch-2",
        parent_commits=[branch_1.commit, branch_2.commit],
        commit_date=datetime.datetime(2024, 3, 1, tzinfo=datetime.timezone.utc),
    )

    branch_3 = git_repo.create_head("branch-3", "master")
    git_repo.head.reference = branch_3  # type: ignore
    git_repo.index.reset(index=True, working_tree=True)
    write_file("file1.txt", "File 1 modified\n")
    write_file("file2.txt", "File 2 modified\n")
    git_repo.index.add(["file1.txt", "file2.txt"])
    git_repo.index.commit(
        "Merge branches branch-1 and branch-2",
        parent_commits=[branch_1.commit, branch_2.commit],
        commit_date=datetime.datetime(2024, 4, 1, tzinfo=datetime.timezone.utc),
    )
    write_file("file3.txt", "File 3 modified\n")
    git_repo.index.add("file3.txt")
    git_repo.index.commit(
        "Modify file3.txt",
        commit_date=datetime.datetime(2024, 5, 1, tzinfo=datetime.timezone.utc),
    )

    with patch("os.getcwd", Mock(return_value=git_repo.working_tree_dir)), patch(
        "rapids_pre_commit_hooks.copyright.get_target_branch",
        Mock(return_value="branch-1-2"),
    ):
        changed_files = copyright.get_changed_files(Mock())
    assert {
        path: old_blob.path if old_blob else None
        for path, old_blob in changed_files.items()
    } == {
        "file1.txt": "file1.txt",
        "file2.txt": "file2.txt",
        "file3.txt": "file3.txt",
    }


def test_normalize_git_filename():
    assert copyright.normalize_git_filename("file.txt") == "file.txt"
    assert copyright.normalize_git_filename("sub/file.txt") == "sub/file.txt"
    assert copyright.normalize_git_filename("sub//file.txt") == "sub/file.txt"
    assert copyright.normalize_git_filename("sub/../file.txt") == "file.txt"
    assert copyright.normalize_git_filename("./file.txt") == "file.txt"
    assert copyright.normalize_git_filename("../file.txt") is None
    assert (
        copyright.normalize_git_filename(os.path.join(os.getcwd(), "file.txt"))
        == "file.txt"
    )
    assert (
        copyright.normalize_git_filename(
            os.path.join("..", os.path.basename(os.getcwd()), "file.txt")
        )
        == "file.txt"
    )


def test_find_blob(git_repo: git.Repo):
    assert git_repo.working_tree_dir is not None

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
    assert copyright.find_blob(git_repo.head.commit.tree, "nonexistent/sub.txt") is None


@freeze_time("2024-01-18")
def test_check_copyright(git_repo: git.Repo):
    def fn(filename: str) -> str:
        assert git_repo.working_tree_dir is not None
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename: str, contents: str):
        with open(fn(filename), "w") as f:
            f.write(contents)

    def file_contents(num: int) -> str:
        return dedent(
            rf"""\
            Copyright (c) 2021-2023 NVIDIA CORPORATION
            File {num}
            """
        )

    def file_contents_modified(num: int) -> str:
        return dedent(
            rf"""\
            Copyright (c) 2021-2023 NVIDIA CORPORATION
            File {num} modified
            """
        )

    write_file("file1.txt", file_contents(1))
    write_file("file2.txt", file_contents(2))
    write_file("file3.txt", file_contents(3))
    write_file("file4.txt", file_contents(4))
    git_repo.index.add(["file1.txt", "file2.txt", "file3.txt", "file4.txt"])
    git_repo.index.commit("Initial commit")

    branch_1 = git_repo.create_head("branch-1", "master")
    git_repo.head.reference = branch_1  # type: ignore
    git_repo.head.reset(index=True, working_tree=True)
    write_file("file1.txt", file_contents_modified(1))
    git_repo.index.add(["file1.txt"])
    git_repo.index.commit("Update file1.txt")

    branch_2 = git_repo.create_head("branch-2", "master")
    git_repo.head.reference = branch_2  # type: ignore
    git_repo.head.reset(index=True, working_tree=True)
    write_file("file2.txt", file_contents_modified(2))
    git_repo.index.add(["file2.txt"])
    git_repo.index.commit("Update file2.txt")

    pr = git_repo.create_head("pr", "branch-1")
    git_repo.head.reference = pr  # type: ignore
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

    def mock_target_branch_upstream_commit(target_branch: str):
        def func(repo: git.Repo, args: argparse.Namespace) -> git.Commit:
            assert target_branch == args.target_branch
            return repo.heads[target_branch].commit

        return patch(
            "rapids_pre_commit_hooks.copyright.get_target_branch_upstream_commit", func
        )

    def mock_apply_copyright_check():
        return patch("rapids_pre_commit_hooks.copyright.apply_copyright_check", Mock())

    #############################
    # branch-1 is target branch
    #############################

    mock_args = Mock(target_branch="branch-1", batch=False)

    with mock_repo_cwd(), mock_target_branch_upstream_commit("branch-1"):
        copyright_checker = copyright.check_copyright(mock_args)

    linter = Linter("file1.txt", file_contents_modified(1))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_not_called()

    linter = Linter("file5.txt", file_contents(2))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(2))

    linter = Linter("file3.txt", file_contents_modified(3))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(3))

    linter = Linter("file4.txt", file_contents_modified(4))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(4))

    linter = Linter("file6.txt", file_contents(6))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, None)

    #############################
    # branch-2 is target branch
    #############################

    mock_args = Mock(target_branch="branch-2", batch=False)

    with mock_repo_cwd(), mock_target_branch_upstream_commit("branch-2"):
        copyright_checker = copyright.check_copyright(mock_args)

    linter = Linter("file1.txt", file_contents_modified(1))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(1))

    linter = Linter("./file1.txt", file_contents_modified(1))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(1))

    linter = Linter("../file1.txt", file_contents_modified(1))
    with mock_apply_copyright_check() as apply_copyright_check:
        with pytest.warns(
            copyright.ConflictingFilesWarning,
            match=r'File "\.\./file1\.txt" is outside of current directory\. Not '
            r"running linter on it\.$",
        ):
            copyright_checker(linter, mock_args)
        apply_copyright_check.assert_not_called()

    linter = Linter("file5.txt", file_contents(2))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(2))

    linter = Linter("file3.txt", file_contents_modified(3))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(3))

    linter = Linter("file4.txt", file_contents_modified(4))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, file_contents(4))

    linter = Linter("file6.txt", file_contents(6))
    with mock_apply_copyright_check() as apply_copyright_check:
        copyright_checker(linter, mock_args)
        apply_copyright_check.assert_called_once_with(linter, None)

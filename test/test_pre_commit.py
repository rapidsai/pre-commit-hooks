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
import shutil
import subprocess
import sys
from functools import cache
from typing import Generator, Optional, Union

import git
import pytest
import yaml
from packaging.version import Version
from rapids_metadata.metadata import RAPIDSMetadata
from rapids_metadata.remote import fetch_latest

REPO_DIR = os.path.join(os.path.dirname(__file__), "..")
with open(os.path.join(REPO_DIR, ".pre-commit-hooks.yaml")) as f:
    ALL_HOOKS = [hook["id"] for hook in yaml.safe_load(f)]
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


@cache
def all_metadata() -> RAPIDSMetadata:
    return fetch_latest()


@contextlib.contextmanager
def set_cwd(cwd: Union[str, os.PathLike[str]]) -> Generator:
    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@pytest.fixture
def git_repo(tmp_path: str) -> git.Repo:
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as w:
        w.set_value("user", "name", "RAPIDS Test Fixtures")
        w.set_value("user", "email", "testfixtures@rapids.ai")
    return repo


def run_pre_commit(
    git_repo: git.Repo, hook_name: str, expected_status: str, exc: Optional[type]
) -> None:
    assert git_repo.working_tree_dir is not None

    def list_files(top: str) -> Generator[str, None, None]:
        for dirpath, _, filenames in os.walk(top):
            for filename in filenames:
                yield filename if top == dirpath else os.path.join(
                    os.path.relpath(top, dirpath), filename
                )

    example_dir = os.path.join(EXAMPLES_DIR, hook_name, expected_status)
    master_dir = os.path.join(example_dir, "master")
    shutil.copytree(master_dir, git_repo.working_tree_dir, dirs_exist_ok=True)

    with open(os.path.join(git_repo.working_tree_dir, "VERSION"), "w") as f:
        f.write(f"{max(all_metadata().versions.keys(), key=Version)}\n")
    git_repo.index.add("VERSION")

    git_repo.index.add(list(list_files(master_dir)))
    git_repo.index.commit(
        "Initial commit",
        commit_date=datetime.datetime(2023, 2, 1, tzinfo=datetime.timezone.utc),
    )

    branch_dir = os.path.join(example_dir, "branch")
    if os.path.exists(branch_dir):
        git_repo.head.reference = git_repo.create_head(  # type: ignore
            "branch", git_repo.head.commit
        )
        git_repo.index.remove(list(list_files(master_dir)), working_tree=True)
        shutil.copytree(branch_dir, git_repo.working_tree_dir, dirs_exist_ok=True)
        git_repo.index.add(list(list_files(branch_dir)))
        git_repo.index.commit(
            "Make some changes",
            commit_date=datetime.datetime(2024, 2, 1, tzinfo=datetime.timezone.utc),
        )

    with set_cwd(git_repo.working_tree_dir), pytest.raises(
        exc
    ) if exc else contextlib.nullcontext():
        subprocess.check_call(
            [sys.executable, "-m", "pre_commit", "try-repo", REPO_DIR, hook_name, "-a"],
            env={**os.environ, "TARGET_BRANCH": "master"},
        )


@pytest.mark.parametrize(
    "hook_name",
    ALL_HOOKS,
)
def test_pre_commit_pass(git_repo: git.Repo, hook_name: str) -> None:
    run_pre_commit(git_repo, hook_name, "pass", None)


@pytest.mark.parametrize(
    "hook_name",
    ALL_HOOKS,
)
def test_pre_commit_fail(git_repo: git.Repo, hook_name: str) -> None:
    run_pre_commit(git_repo, hook_name, "fail", subprocess.CalledProcessError)

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

import contextlib
import datetime
import json
import os.path
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import cache
from textwrap import dedent

import git
import pytest
import yaml
from packaging.version import Version
from rapids_metadata.remote import fetch_latest

HOOKS_REPO_DIR = os.path.join(os.path.dirname(__file__), "..")
with open(os.path.join(HOOKS_REPO_DIR, ".pre-commit-hooks.yaml")) as f:
    ALL_HOOKS = [hook["id"] for hook in yaml.safe_load(f)]
HOOKS_REPO = git.Repo(HOOKS_REPO_DIR)
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


maybe_skip: Callable = pytest.mark.skipif(
    any(HOOKS_REPO.head.commit.diff(other=None)),
    reason="Hooks repo has modified files that haven't been committed",
)


@cache
def all_metadata():
    return fetch_latest()


@contextlib.contextmanager
def set_cwd(cwd):
    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@pytest.fixture
def git_repo(tmp_path):
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as w:
        w.set_value("user", "name", "RAPIDS Test Fixtures")
        w.set_value("user", "email", "testfixtures@rapids.ai")
    return repo


def run_pre_commit(git_repo, hook_name, expected_status, exc):
    def list_files(top):
        for dirpath, _, filenames in os.walk(top):
            for filename in filenames:
                yield (
                    filename
                    if top == dirpath
                    else os.path.join(os.path.relpath(dirpath, top), filename)
                )

    example_dir = os.path.join(EXAMPLES_DIR, hook_name, expected_status)
    master_dir = os.path.join(example_dir, "master")
    shutil.copytree(master_dir, git_repo.working_tree_dir, dirs_exist_ok=True)

    with open(os.path.join(git_repo.working_tree_dir, "VERSION"), "w") as f:
        f.write(f"{max(all_metadata().versions.keys(), key=Version)}\n")
    try:
        f = open(os.path.join(example_dir, "metadata.yaml"))
    except FileNotFoundError:
        args_text = ""
    else:
        with f:
            metadata = yaml.safe_load(f)
        args_text = f"args: {json.dumps(metadata['args'])}"
    with open(
        os.path.join(git_repo.working_tree_dir, ".pre-commit-config.yaml"), "w"
    ) as f:
        f.write(
            dedent(
                f"""
                repos:
                - repo: "{HOOKS_REPO_DIR}"
                  rev: "{HOOKS_REPO.head.commit.hexsha}"
                  hooks:
                    - id: {hook_name}
                      {args_text}
                """
            )
        )

    git_repo.index.add("VERSION", ".pre-commit-config.yaml")

    git_repo.index.add(list(list_files(master_dir)))
    git_repo.index.commit(
        "Initial commit",
        commit_date=datetime.datetime(
            2023, 2, 1, tzinfo=datetime.timezone.utc
        ),
    )

    branch_dir = os.path.join(example_dir, "branch")
    if os.path.exists(branch_dir):
        git_repo.head.reference = git_repo.create_head(
            "branch", git_repo.head.commit
        )
        git_repo.index.remove(list(list_files(master_dir)), working_tree=True)
        shutil.copytree(
            branch_dir, git_repo.working_tree_dir, dirs_exist_ok=True
        )
        git_repo.index.add(list(list_files(branch_dir)))
        git_repo.index.commit(
            "Make some changes",
            commit_date=datetime.datetime(
                2024, 2, 1, tzinfo=datetime.timezone.utc
            ),
        )

    with (
        set_cwd(git_repo.working_tree_dir),
        pytest.raises(exc) if exc else contextlib.nullcontext(),
    ):
        subprocess.check_call(
            [sys.executable, "-m", "pre_commit", "run", hook_name, "-a"],
            env={
                **os.environ,
                "TARGET_BRANCH": "master",
                "RAPIDS_TEST_YEAR": "2024",
            },
        )


@pytest.mark.parametrize(
    "hook_name",
    ALL_HOOKS,
)
@maybe_skip
def test_pre_commit_pass(git_repo, hook_name):
    run_pre_commit(git_repo, hook_name, "pass", None)


@pytest.mark.parametrize(
    "hook_name",
    ALL_HOOKS,
)
@maybe_skip
def test_pre_commit_fail(git_repo, hook_name):
    run_pre_commit(git_repo, hook_name, "fail", subprocess.CalledProcessError)

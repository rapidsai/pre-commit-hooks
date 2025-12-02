# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import datetime
import json
import os.path
import shutil
import subprocess
import sys
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
    repo = git.Repo.init(tmp_path, initial_branch="main")
    with repo.config_writer() as w:
        w.set_value("user", "name", "RAPIDS Test Fixtures")
        w.set_value("user", "email", "testfixtures@rapids.ai")
    return repo


@pytest.mark.skipif(
    any(HOOKS_REPO.head.commit.diff(other=None)),
    reason="Hooks repo has modified files that haven't been committed",
)
@pytest.mark.parametrize(
    ["expected_status", "context"],
    [
        pytest.param("pass", contextlib.nullcontext(), id="pass"),
        pytest.param(
            "fail", pytest.raises(subprocess.CalledProcessError), id="fail"
        ),
    ],
)
@pytest.mark.parametrize(
    "hook_name",
    ALL_HOOKS,
)
def test_pre_commit(git_repo, hook_name, expected_status, context):
    def list_files(top):
        for dirpath, _, filenames in os.walk(top):
            for filename in filenames:
                yield (
                    filename
                    if top == dirpath
                    else os.path.join(os.path.relpath(dirpath, top), filename)
                )

    example_dir = os.path.join(EXAMPLES_DIR, hook_name, expected_status)
    main_dir = os.path.join(example_dir, "main")
    shutil.copytree(main_dir, git_repo.working_tree_dir, dirs_exist_ok=True)

    try:
        f = open(os.path.join(example_dir, "metadata.yaml"))
    except FileNotFoundError:
        args_text = ""
        write_version_file = False
    else:
        with f:
            metadata = yaml.safe_load(f)
        try:
            args = metadata["args"]
        except KeyError:
            args_text = ""
        else:
            args_text = f"args: {json.dumps(args)}"
        write_version_file = metadata.get("write_version_file", False)

    if write_version_file:
        with open(
            os.path.join(git_repo.working_tree_dir, "VERSION"), "w"
        ) as f:
            f.write(f"{max(all_metadata().versions.keys(), key=Version)}\n")
        git_repo.index.add("VERSION")

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
    git_repo.index.add(".pre-commit-config.yaml")

    git_repo.index.add(list(list_files(main_dir)))
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
        git_repo.index.remove(list(list_files(main_dir)), working_tree=True)
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
        context,
    ):
        subprocess.check_call(
            [sys.executable, "-m", "pre_commit", "run", hook_name, "-a"],
            env={
                **os.environ,
                "RAPIDS_COPYRIGHT_FORCE_SPDX": "0",
                "TARGET_BRANCH": "main",
                "RAPIDS_TEST_YEAR": "2024",
            },
        )

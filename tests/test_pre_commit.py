# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import datetime
import json
import os.path
import pathlib
import shutil
import subprocess
import sys
from functools import cache
from itertools import chain
from textwrap import dedent

import git
import pytest
import yaml
from packaging.version import Version
from rapids_metadata.remote import fetch_latest

TESTS_DIR = pathlib.Path(__file__).parent
HOOKS_REPO_DIR = TESTS_DIR / ".."
with open(HOOKS_REPO_DIR / ".pre-commit-hooks.yaml") as f:
    ALL_HOOKS = [hook["id"] for hook in yaml.safe_load(f)]
EXAMPLES_DIR = TESTS_DIR / "examples"


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


@pytest.fixture(scope="session")
def committed_hooks_repo(tmpdir_factory):
    hooks_repo = git.Repo(HOOKS_REPO_DIR)
    changed_files = {
        *chain.from_iterable(
            (
                *((f.a_path,) if f.a_path is not None else ()),
                *((f.b_path,) if f.b_path is not None else ()),
            )
            for f in hooks_repo.head.commit.diff(other=None)
        ),
        *hooks_repo.untracked_files,
    }
    if any(changed_files):
        new_repo_dir = tmpdir_factory.mktemp("hooks_repo")
        new_repo = git.Repo.init(new_repo_dir, initial_branch="main")
        with new_repo.config_writer() as w:
            w.set_value("user", "name", "RAPIDS Test Fixtures")
            w.set_value("user", "email", "testfixtures@rapids.ai")
        new_repo.create_remote("origin", hooks_repo.git_dir).fetch(
            hooks_repo.head.commit.hexsha, depth=1
        )
        new_repo.git.checkout(hooks_repo.head.commit.hexsha)

        for file in changed_files:
            (new_repo_dir / file).dirpath().ensure(dir=True)
            try:
                shutil.copy(
                    HOOKS_REPO_DIR / file,
                    new_repo_dir / file,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                os.remove(new_repo_dir / file)
                new_repo.index.remove(file)
            else:
                new_repo.index.add(file)
        commit = new_repo.index.commit("Update with uncommitted changes")
        return (new_repo.git_dir, commit.hexsha)
    else:
        return (hooks_repo.git_dir, hooks_repo.head.commit.hexsha)


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
def test_pre_commit(
    git_repo, committed_hooks_repo, hook_name, expected_status, context
):
    def list_files(top):
        for dirpath, _, filenames in os.walk(top):
            for filename in filenames:
                yield (
                    filename
                    if top == dirpath
                    else os.path.join(os.path.relpath(dirpath, top), filename)
                )

    hooks_repo, hooks_commit = committed_hooks_repo

    example_dir = os.path.join(EXAMPLES_DIR, hook_name, expected_status)
    main_dir = os.path.join(example_dir, "main")

    # Skip test if example directory doesn't exist
    if not os.path.exists(main_dir):
        pytest.skip(f"No {expected_status} example for {hook_name}")

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
                - repo: "{hooks_repo}"
                  rev: "{hooks_commit}"
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
                "SKIP_CODERABBIT_IF_MISSING": "true",
            },
        )

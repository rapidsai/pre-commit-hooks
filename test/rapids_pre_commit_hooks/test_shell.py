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

import bashlex  # type: ignore

from rapids_pre_commit_hooks.lint import Linter
from rapids_pre_commit_hooks.shell.verify_conda_yes import VerifyCondaYesVisitor


def run_shell_linter(content, cls):
    linter = Linter("test.sh", content)
    visitor = cls(linter, None)
    parts = bashlex.parse(content)
    for part in parts:
        visitor.visit(part)
    return linter


def test_verify_conda_yes():
    CONTENT = r"""
conda install -y pkg1
conda install --yes pkg2 pkg3
conda install pkg4
conda -h install
conda --help install
if true; then
  conda --no-plugins install pkg5
fi
# conda install pkg6
conda -V
conda
conda clean -y
conda clean
conda create -y
conda create
conda remove -y
conda remove
conda uninstall -y
conda uninstall
conda update -y
conda update
conda upgrade -y
conda upgrade
conda search
conda install $pkg1 "$pkg2"
"""
    expected_linter = Linter("test.sh", CONTENT)
    expected_linter.add_warning((53, 66), "add -y argument").add_replacement(
        (66, 66), " -y"
    )
    expected_linter.add_warning((126, 152), "add -y argument").add_replacement(
        (152, 152), " -y"
    )
    expected_linter.add_warning((212, 223), "add -y argument").add_replacement(
        (223, 223), " -y"
    )
    expected_linter.add_warning((240, 252), "add -y argument").add_replacement(
        (252, 252), " -y"
    )
    expected_linter.add_warning((269, 281), "add -y argument").add_replacement(
        (281, 281), " -y"
    )
    expected_linter.add_warning((301, 316), "add -y argument").add_replacement(
        (316, 316), " -y"
    )
    expected_linter.add_warning((333, 345), "add -y argument").add_replacement(
        (345, 345), " -y"
    )
    expected_linter.add_warning((363, 376), "add -y argument").add_replacement(
        (376, 376), " -y"
    )
    expected_linter.add_warning((390, 403), "add -y argument").add_replacement(
        (403, 403), " -y"
    )

    linter = run_shell_linter(CONTENT, VerifyCondaYesVisitor)
    assert linter.warnings == expected_linter.warnings

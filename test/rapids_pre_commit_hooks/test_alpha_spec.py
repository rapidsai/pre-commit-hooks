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

from itertools import chain
from textwrap import dedent
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import yaml

from rapids_pre_commit_hooks import alpha_spec, lint


def test_anchor_preserving_loader():
    loader = alpha_spec.AnchorPreservingLoader("- &a A\n- *a")
    try:
        root = loader.get_single_node()
    finally:
        loader.dispose()
    assert loader.document_anchors == [{"a": root.value[0]}]


@pytest.mark.parametrize(
    ["name", "is_suffixed"],
    [
        *chain(
            *(
                [
                    (f"{p}-cu11", True),
                    (f"{p}-cu12", True),
                    (f"{p}-cuda", False),
                ]
                for p in alpha_spec.RAPIDS_CUDA_SUFFIXED_PACKAGES
            )
        ),
        *chain(
            *(
                [
                    (f"{p}-cu11", False),
                    (f"{p}-cu12", False),
                    (f"{p}-cuda", False),
                ]
                for p in alpha_spec.RAPIDS_NON_CUDA_SUFFIXED_PACKAGES
            )
        ),
    ],
)
def test_is_rapids_cuda_suffixed_package(name, is_suffixed):
    assert alpha_spec.is_rapids_cuda_suffixed_package(name) == is_suffixed


@pytest.mark.parametrize(
    ["package", "content", "mode", "replacement"],
    [
        *chain(
            *(
                [
                    (p, p, "development", f"{p}>=0.0.0a0"),
                    (p, p, "release", None),
                    (p, f"{p}>=0.0.0a0", "development", None),
                    (p, f"{p}>=0.0.0a0", "release", p),
                ]
                for p in alpha_spec.RAPIDS_ALPHA_SPEC_PACKAGES
            )
        ),
        *chain(
            *(
                [
                    (f"{p}-cu12", f"{p}-cu12", "development", f"{p}-cu12>=0.0.0a0"),
                    (f"{p}-cu11", f"{p}-cu11", "release", None),
                    (f"{p}-cu12", f"{p}-cu12>=0.0.0a0", "development", None),
                    (f"{p}-cu11", f"{p}-cu11>=0.0.0a0", "release", f"{p}-cu11"),
                ]
                for p in alpha_spec.RAPIDS_CUDA_SUFFIXED_PACKAGES
            )
        ),
        *chain(
            *(
                [
                    (f"{p}-cu12", f"{p}-cu12", "development", None),
                    (f"{p}-cu12", f"{p}-cu12>=0.0.0a0", "release", None),
                ]
                for p in alpha_spec.RAPIDS_NON_CUDA_SUFFIXED_PACKAGES
            )
        ),
        ("cuml", "cuml>=24.04,<24.06", "development", "cuml>=24.04,<24.06,>=0.0.0a0"),
        ("cuml", "cuml>=24.04,<24.06,>=0.0.0a0", "release", "cuml>=24.04,<24.06"),
        (
            "cuml",
            "&cuml cuml>=24.04,<24.06,>=0.0.0a0",
            "release",
            "&cuml cuml>=24.04,<24.06",
        ),
        ("packaging", "packaging", "development", None),
        (None, "--extra-index-url=https://pypi.nvidia.com", "development", None),
        (None, "--extra-index-url=https://pypi.nvidia.com", "release", None),
        (None, "gcc_linux-64=11.*", "development", None),
        (None, "gcc_linux-64=11.*", "release", None),
    ],
)
def test_check_package_spec(package, content, mode, replacement):
    args = Mock(mode=mode)
    linter = lint.Linter("dependencies.yaml", content)
    loader = alpha_spec.AnchorPreservingLoader(content)
    try:
        composed = loader.get_single_node()
    finally:
        loader.dispose()
    alpha_spec.check_package_spec(
        linter, args, loader.document_anchors[0], set(), composed
    )
    if replacement is None:
        assert linter.warnings == []
    else:
        expected_linter = lint.Linter("dependencies.yaml", content)
        expected_linter.add_warning(
            (composed.start_mark.index, composed.end_mark.index),
            f"{'add' if mode == 'development' else 'remove'} "
            f"alpha spec for RAPIDS package {package}",
        ).add_replacement(
            (composed.start_mark.index, composed.end_mark.index), replacement
        )
        assert linter.warnings == expected_linter.warnings


def test_check_package_spec_anchor():
    CONTENT = dedent(
        """\
        - &cudf cudf>=24.04,<24.06
        - *cudf
        - cuml>=24.04,<24.06
        - rmm>=24.04,<24.06
        """
    )
    args = Mock(mode="development")
    linter = lint.Linter("dependencies.yaml", CONTENT)
    loader = alpha_spec.AnchorPreservingLoader(CONTENT)
    try:
        composed = loader.get_single_node()
    finally:
        loader.dispose()
    used_anchors = set()

    expected_linter = lint.Linter("dependencies.yaml", CONTENT)
    expected_linter.add_warning(
        (2, 26), "add alpha spec for RAPIDS package cudf"
    ).add_replacement((2, 26), "&cudf cudf>=24.04,<24.06,>=0.0.0a0")

    alpha_spec.check_package_spec(
        linter, args, loader.document_anchors[0], used_anchors, composed.value[0]
    )
    assert linter.warnings == expected_linter.warnings
    assert used_anchors == {"cudf"}

    alpha_spec.check_package_spec(
        linter, args, loader.document_anchors[0], used_anchors, composed.value[1]
    )
    assert linter.warnings == expected_linter.warnings
    assert used_anchors == {"cudf"}

    expected_linter.add_warning(
        (37, 55), "add alpha spec for RAPIDS package cuml"
    ).add_replacement((37, 55), "cuml>=24.04,<24.06,>=0.0.0a0")
    alpha_spec.check_package_spec(
        linter, args, loader.document_anchors[0], used_anchors, composed.value[2]
    )
    assert linter.warnings == expected_linter.warnings
    assert used_anchors == {"cudf"}

    expected_linter.add_warning(
        (58, 75), "add alpha spec for RAPIDS package rmm"
    ).add_replacement((58, 75), "rmm>=24.04,<24.06,>=0.0.0a0")
    alpha_spec.check_package_spec(
        linter, args, loader.document_anchors[0], used_anchors, composed.value[3]
    )
    assert linter.warnings == expected_linter.warnings
    assert used_anchors == {"cudf"}


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
                - package_a
                - &package_b package_b
                """
            ),
            [0, 1],
        ),
        (
            "null",
            [],
        ),
    ],
)
def test_check_packages(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_package_spec", Mock()
    ) as mock_check_package_spec:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_packages(linter, args, anchors, used_anchors, composed)
    assert mock_check_package_spec.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i]) for i in indices
    ]


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
                - output_types: [pyproject, conda]
                  packages:
                    - package_a
                - output_types: [conda]
                  packages:
                    - package_b
                - packages:
                    - package_c
                  output_types: pyproject
                """
            ),
            [(0, 1), (1, 1), (2, 0)],
        ),
    ],
)
def test_check_common(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_packages", Mock()
    ) as mock_check_packages:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_common(linter, args, anchors, used_anchors, composed)
    assert mock_check_packages.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i].value[j][1])
        for i, j in indices
    ]


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
                - matrix:
                    arch: x86_64
                  packages:
                    - package_a
                - packages:
                    - package_b
                  matrix:
                """
            ),
            [(0, 1), (1, 0)],
        ),
    ],
)
def test_check_matrices(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_packages", Mock()
    ) as mock_check_packages:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_matrices(linter, args, anchors, used_anchors, composed)
    assert mock_check_packages.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i].value[j][1])
        for i, j in indices
    ]


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
                - output_types: [pyproject, conda]
                  matrices:
                    - matrix:
                        arch: x86_64
                      packages:
                        - package_a
                - output_types: [conda]
                  matrices:
                    - matrix:
                        arch: x86_64
                      packages:
                        - package_b
                - matrices:
                    - matrix:
                        arch: x86_64
                      packages:
                        - package_c
                  output_types: pyproject
                """
            ),
            [(0, 1), (1, 1), (2, 0)],
        ),
    ],
)
def test_check_specific(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_matrices", Mock()
    ) as mock_check_matrices:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_specific(linter, args, anchors, used_anchors, composed)
    assert mock_check_matrices.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i].value[j][1])
        for i, j in indices
    ]


@pytest.mark.parametrize(
    ["content", "common_indices", "specific_indices"],
    [
        (
            dedent(
                """\
                set_a:
                  common:
                    - output_types: [pyproject]
                      packages:
                        - package_a
                  specific:
                    - output_types: [pyproject]
                      matrices:
                        - matrix:
                            arch: x86_64
                          packages:
                            - package_b
                set_b:
                  specific:
                    - output_types: [pyproject]
                      matrices:
                        - matrix:
                            arch: x86_64
                          packages:
                            - package_c
                  common:
                    - output_types: [pyproject]
                      packages:
                        - package_d
                """
            ),
            [(0, 0), (1, 1)],
            [(0, 1), (1, 0)],
        ),
    ],
)
def test_check_dependencies(content, common_indices, specific_indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_common", Mock()
    ) as mock_check_common, patch(
        "rapids_pre_commit_hooks.alpha_spec.check_specific", Mock()
    ) as mock_check_specific:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_dependencies(linter, args, anchors, used_anchors, composed)
    assert mock_check_common.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i][1].value[j][1])
        for i, j in common_indices
    ]
    assert mock_check_specific.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i][1].value[j][1])
        for i, j in specific_indices
    ]


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
            files: {}
            channels: []
            dependencies: {}
            """
            ),
            [2],
        ),
    ],
)
def test_check_root(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_dependencies", Mock()
    ) as mock_check_dependencies:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_root(linter, args, anchors, used_anchors, composed)
    assert mock_check_dependencies.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i][1]) for i in indices
    ]


def test_check_alpha_spec():
    CONTENT = "dependencies: []"
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_root", Mock()
    ) as mock_check_root, patch(
        "rapids_pre_commit_hooks.alpha_spec.AnchorPreservingLoader", MagicMock()
    ) as mock_anchor_preserving_loader:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", CONTENT)
        alpha_spec.check_alpha_spec(linter, args)
    mock_anchor_preserving_loader.assert_called_once_with(CONTENT)
    mock_check_root.assert_called_once_with(
        linter,
        args,
        mock_anchor_preserving_loader().document_anchors[0],
        set(),
        mock_anchor_preserving_loader().get_single_node(),
    )


def test_check_alpha_spec_integration():
    CONTENT = dedent(
        """\
        dependencies:
          test:
            common:
              - output_types: pyproject
                packages:
                  - cudf>=24.04,<24.06
        """
    )
    REPLACED = "cudf>=24.04,<24.06"

    args = Mock(mode="development")
    linter = lint.Linter("dependencies.yaml", CONTENT)
    alpha_spec.check_alpha_spec(linter, args)

    start = CONTENT.find(REPLACED)
    end = start + len(REPLACED)
    pos = (start, end)

    expected_linter = lint.Linter("dependencies.yaml", CONTENT)
    expected_linter.add_warning(
        pos, "add alpha spec for RAPIDS package cudf"
    ).add_replacement(pos, "cudf>=24.04,<24.06,>=0.0.0a0")
    assert linter.warnings == expected_linter.warnings

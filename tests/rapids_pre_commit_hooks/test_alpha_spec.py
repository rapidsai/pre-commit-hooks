# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os.path
from itertools import chain
from textwrap import dedent
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import yaml
from packaging.version import Version
from rapids_metadata.metadata import (
    RAPIDSMetadata,
    RAPIDSRepository,
    RAPIDSVersion,
)

from rapids_pre_commit_hooks import alpha_spec, lint

latest_version, latest_metadata = max(
    alpha_spec.all_metadata().versions.items(),
    key=lambda item: Version(item[0]),
)


@contextlib.contextmanager
def set_cwd(cwd):
    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@pytest.mark.parametrize(
    [
        "version_file",
        "version_file_contents",
        "version_arg",
        "expected_version",
        "raises",
    ],
    [
        ("VERSION", "24.06", None, "24.06", contextlib.nullcontext()),
        ("RAPIDS_VERSION", "24.06", None, "24.06", contextlib.nullcontext()),
        ("VERSION", "24.06", "24.08", "24.08", contextlib.nullcontext()),
        ("VERSION", "24.08", "24.06", "24.06", contextlib.nullcontext()),
        ("VERSION", None, "24.06", "24.06", contextlib.nullcontext()),
        ("VERSION", None, "24.10", None, pytest.raises(KeyError)),
        ("VERSION", None, None, None, pytest.raises(FileNotFoundError)),
    ],
)
def test_get_rapids_version(
    tmp_path,
    version_file,
    version_file_contents,
    version_arg,
    expected_version,
    raises,
):
    MOCK_METADATA = RAPIDSMetadata(
        versions={
            "24.06": RAPIDSVersion(
                repositories={
                    "repo1": RAPIDSRepository(),
                },
            ),
            "24.08": RAPIDSVersion(
                repositories={
                    "repo2": RAPIDSRepository(),
                },
            ),
        },
    )
    with (
        set_cwd(tmp_path),
        patch(
            "rapids_pre_commit_hooks.alpha_spec.all_metadata",
            Mock(return_value=MOCK_METADATA),
        ),
    ):
        if version_file_contents:
            with open(version_file, "w") as f:
                f.write(f"{version_file_contents}\n")
        args = Mock(
            rapids_version=version_arg, rapids_version_file=version_file
        )
        with raises:
            version = alpha_spec.get_rapids_version(args)
            if expected_version:
                assert version == MOCK_METADATA.versions[expected_version]


def test_anchor_preserving_loader():
    loader = alpha_spec.AnchorPreservingLoader("- &a A\n- *a")
    try:
        root = loader.get_single_node()
    finally:
        loader.dispose()
    assert loader.document_anchors == [{"a": root.value[0]}]


@pytest.mark.parametrize(
    ["name", "stripped_name"],
    [
        *chain(
            *(
                [
                    (p, p),
                    (f"{p}-cu11", p),
                    (f"{p}-cu12", p),
                    (f"{p}-cuda", f"{p}-cuda"),
                ]
                for p in latest_metadata.cuda_suffixed_packages
            )
        ),
        *chain(
            *(
                [
                    (p, p),
                    (f"{p}-cu11", f"{p}-cu11"),
                    (f"{p}-cu12", f"{p}-cu12"),
                    (f"{p}-cuda", f"{p}-cuda"),
                ]
                for p in latest_metadata.all_packages
                - latest_metadata.cuda_suffixed_packages
            )
        ),
    ],
)
@patch(
    "rapids_pre_commit_hooks.alpha_spec.get_rapids_version",
    Mock(return_value=latest_metadata),
)
def test_strip_cuda_suffix(name, stripped_name):
    assert alpha_spec.strip_cuda_suffix(Mock(), name) == stripped_name


@pytest.mark.parametrize(
    [
        "used_anchors_before",
        "node_index",
        "descend",
        "anchor",
        "used_anchors_after",
    ],
    [
        (
            set(),
            0,
            True,
            "anchor1",
            {"anchor1"},
        ),
        (
            {"anchor1"},
            1,
            True,
            "anchor2",
            {"anchor1", "anchor2"},
        ),
        (
            set(),
            2,
            True,
            None,
            set(),
        ),
        (
            {"anchor1", "anchor2"},
            0,
            False,
            "anchor1",
            {"anchor1", "anchor2"},
        ),
        (
            {"anchor1", "anchor2"},
            1,
            False,
            "anchor2",
            {"anchor1", "anchor2"},
        ),
    ],
)
def test_check_and_mark_anchor(
    used_anchors_before,
    node_index,
    descend,
    anchor,
    used_anchors_after,
):
    NODES = [Mock() for _ in range(3)]
    ANCHORS = {
        "anchor1": NODES[0],
        "anchor2": NODES[1],
    }
    used_anchors = set(used_anchors_before)
    actual_descend, actual_anchor = alpha_spec.check_and_mark_anchor(
        ANCHORS, used_anchors, NODES[node_index]
    )
    assert actual_descend == descend
    assert actual_anchor == anchor
    assert used_anchors == used_anchors_after


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
                for p in latest_metadata.prerelease_packages
            )
        ),
        *chain(
            *(
                [
                    (
                        f"{p}-cu12",
                        f"{p}-cu12",
                        "development",
                        f"{p}-cu12>=0.0.0a0",
                    ),
                    (f"{p}-cu11", f"{p}-cu11", "release", None),
                    (f"{p}-cu12", f"{p}-cu12>=0.0.0a0", "development", None),
                    (
                        f"{p}-cu11",
                        f"{p}-cu11>=0.0.0a0",
                        "release",
                        f"{p}-cu11",
                    ),
                ]
                for p in latest_metadata.prerelease_packages
                & latest_metadata.cuda_suffixed_packages
            )
        ),
        *chain(
            *(
                [
                    (f"{p}-cu12", f"{p}-cu12", "development", None),
                    (f"{p}-cu12", f"{p}-cu12>=0.0.0a0", "release", None),
                ]
                for p in latest_metadata.prerelease_packages
                & (
                    latest_metadata.all_packages
                    - latest_metadata.cuda_suffixed_packages
                )
            )
        ),
        (
            "cuml",
            "cuml>=24.04,<24.06",
            "development",
            "cuml>=24.04,<24.06,>=0.0.0a0",
        ),
        (
            "cuml",
            "cuml>=24.04,<24.06,>=0.0.0a0",
            "release",
            "cuml>=24.04,<24.06",
        ),
        (
            "cuml",
            "&cuml cuml>=24.04,<24.06,>=0.0.0a0",
            "release",
            "&cuml cuml>=24.04,<24.06",
        ),
        ("packaging", "packaging", "development", None),
        (
            None,
            "--extra-index-url=https://pypi.nvidia.com",
            "development",
            None,
        ),
        (None, "--extra-index-url=https://pypi.nvidia.com", "release", None),
        (None, "gcc_linux-64=11.*", "development", None),
        (None, "gcc_linux-64=11.*", "release", None),
    ],
)
@patch(
    "rapids_pre_commit_hooks.alpha_spec.get_rapids_version",
    Mock(return_value=latest_metadata),
)
def test_check_package_spec(package, content, mode, replacement):
    args = Mock(mode=mode)
    linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
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
        expected_linter = lint.Linter(
            "dependencies.yaml", content, "verify-alpha-spec"
        )
        expected_linter.add_warning(
            (composed.start_mark.index, composed.end_mark.index),
            f"{'add' if mode == 'development' else 'remove'} "
            f"alpha spec for RAPIDS package {package}",
        ).add_replacement(
            (composed.start_mark.index, composed.end_mark.index), replacement
        )
        assert linter.warnings == expected_linter.warnings


@patch(
    "rapids_pre_commit_hooks.alpha_spec.get_rapids_version",
    Mock(return_value=latest_metadata),
)
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
    linter = lint.Linter("dependencies.yaml", CONTENT, "verify-alpha-spec")
    loader = alpha_spec.AnchorPreservingLoader(CONTENT)
    try:
        composed = loader.get_single_node()
    finally:
        loader.dispose()
    used_anchors = set()

    expected_linter = lint.Linter(
        "dependencies.yaml", CONTENT, "verify-alpha-spec"
    )
    expected_linter.add_warning(
        (2, 26), "add alpha spec for RAPIDS package cudf"
    ).add_replacement((2, 26), "&cudf cudf>=24.04,<24.06,>=0.0.0a0")

    alpha_spec.check_package_spec(
        linter,
        args,
        loader.document_anchors[0],
        used_anchors,
        composed.value[0],
    )
    assert linter.warnings == expected_linter.warnings
    assert used_anchors == {"cudf"}

    alpha_spec.check_package_spec(
        linter,
        args,
        loader.document_anchors[0],
        used_anchors,
        composed.value[1],
    )
    assert linter.warnings == expected_linter.warnings
    assert used_anchors == {"cudf"}

    expected_linter.add_warning(
        (37, 55), "add alpha spec for RAPIDS package cuml"
    ).add_replacement((37, 55), "cuml>=24.04,<24.06,>=0.0.0a0")
    alpha_spec.check_package_spec(
        linter,
        args,
        loader.document_anchors[0],
        used_anchors,
        composed.value[2],
    )
    assert linter.warnings == expected_linter.warnings
    assert used_anchors == {"cudf"}

    expected_linter.add_warning(
        (58, 75), "add alpha spec for RAPIDS package rmm"
    ).add_replacement((58, 75), "rmm>=24.04,<24.06,>=0.0.0a0")
    alpha_spec.check_package_spec(
        linter,
        args,
        loader.document_anchors[0],
        used_anchors,
        composed.value[3],
    )
    assert linter.warnings == expected_linter.warnings
    assert used_anchors == {"cudf"}


@pytest.mark.parametrize(
    ["content", "indices", "use_anchor"],
    [
        (
            dedent(
                """\
                - package_a
                - &package_b package_b
                """
            ),
            [0, 1],
            True,
        ),
        (
            "null",
            [],
            False,
        ),
    ],
)
def test_check_packages(content, indices, use_anchor):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_package_spec", Mock()
    ) as mock_check_package_spec:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
        composed = yaml.compose(content)
        anchors = {"anchor": composed}
        used_anchors = set()
        alpha_spec.check_packages(
            linter, args, anchors, used_anchors, composed
        )
        assert used_anchors == ({"anchor"} if use_anchor else set())
        alpha_spec.check_packages(
            linter, args, anchors, used_anchors, composed
        )
    assert mock_check_package_spec.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i])
        for i in indices
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
        linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_common(linter, args, anchors, used_anchors, composed)
    assert mock_check_packages.mock_calls == [
        call(
            linter, args, anchors, used_anchors, composed.value[i].value[j][1]
        )
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
        linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_matrices(
            linter, args, anchors, used_anchors, composed
        )
    assert mock_check_packages.mock_calls == [
        call(
            linter, args, anchors, used_anchors, composed.value[i].value[j][1]
        )
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
        linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_specific(
            linter, args, anchors, used_anchors, composed
        )
    assert mock_check_matrices.mock_calls == [
        call(
            linter, args, anchors, used_anchors, composed.value[i].value[j][1]
        )
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
def test_check_dependencies(
    content,
    common_indices,
    specific_indices,
):
    with (
        patch(
            "rapids_pre_commit_hooks.alpha_spec.check_common", Mock()
        ) as mock_check_common,
        patch(
            "rapids_pre_commit_hooks.alpha_spec.check_specific", Mock()
        ) as mock_check_specific,
    ):
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_dependencies(
            linter, args, anchors, used_anchors, composed
        )
    assert mock_check_common.mock_calls == [
        call(
            linter,
            args,
            anchors,
            used_anchors,
            composed.value[i][1].value[j][1],
        )
        for i, j in common_indices
    ]
    assert mock_check_specific.mock_calls == [
        call(
            linter,
            args,
            anchors,
            used_anchors,
            composed.value[i][1].value[j][1],
        )
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
        linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
        anchors = Mock()
        used_anchors = Mock()
        composed = yaml.compose(content)
        alpha_spec.check_root(linter, args, anchors, used_anchors, composed)
    assert mock_check_dependencies.mock_calls == [
        call(linter, args, anchors, used_anchors, composed.value[i][1])
        for i in indices
    ]


def test_check_alpha_spec():
    CONTENT = "dependencies: []"
    with (
        patch(
            "rapids_pre_commit_hooks.alpha_spec.check_root", Mock()
        ) as mock_check_root,
        patch(
            "rapids_pre_commit_hooks.alpha_spec.AnchorPreservingLoader",
            MagicMock(),
        ) as mock_anchor_preserving_loader,
    ):
        args = Mock()
        linter = lint.Linter("dependencies.yaml", CONTENT, "verify-alpha-spec")
        alpha_spec.check_alpha_spec(linter, args)
    mock_anchor_preserving_loader.assert_called_once_with(CONTENT)
    mock_check_root.assert_called_once_with(
        linter,
        args,
        mock_anchor_preserving_loader().document_anchors[0],
        set(),
        mock_anchor_preserving_loader().get_single_node(),
    )


def test_check_alpha_spec_integration(tmp_path):
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

    args = Mock(
        mode="development", rapids_version=None, rapids_version_file="VERSION"
    )
    linter = lint.Linter("dependencies.yaml", CONTENT, "verify-alpha-spec")
    with open(os.path.join(tmp_path, "VERSION"), "w") as f:
        f.write(f"{latest_version}\n")
    with set_cwd(tmp_path):
        alpha_spec.check_alpha_spec(linter, args)

    start = CONTENT.find(REPLACED)
    end = start + len(REPLACED)
    span = (start, end)

    expected_linter = lint.Linter(
        "dependencies.yaml", CONTENT, "verify-alpha-spec"
    )
    expected_linter.add_warning(
        span, "add alpha spec for RAPIDS package cudf"
    ).add_replacement(span, "cudf>=24.04,<24.06,>=0.0.0a0")
    assert linter.warnings == expected_linter.warnings

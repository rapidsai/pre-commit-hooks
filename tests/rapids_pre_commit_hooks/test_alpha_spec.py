# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os.path
from itertools import chain
from unittest.mock import Mock, patch

import pytest
from packaging.version import Version
from rapids_metadata.metadata import (
    RAPIDSMetadata,
    RAPIDSRepository,
    RAPIDSVersion,
)

from rapids_pre_commit_hooks import alpha_spec, lint
from rapids_pre_commit_hooks.utils.yaml import (
    Anchor,
    AnchorType,
    load_with_anchors,
)
from rapids_pre_commit_hooks_test_utils import (
    parse_named_spans,
    zip_expected_warnings,
)

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


class TestAlphaSpecHandler:
    @pytest.mark.parametrize(
        ["anchor", "packages_is_reference_anchor"],
        [
            pytest.param(
                None,
                False,
                id="no-anchor",
            ),
            pytest.param(
                Anchor(AnchorType.DEFINITION, "anchor"),
                False,
                id="anchor-definition",
            ),
            pytest.param(
                Anchor(AnchorType.REFERENCE, "anchor"),
                True,
                id="anchor-reference",
            ),
        ],
    )
    def test_handle_packages(self, anchor, packages_is_reference_anchor):
        handler = alpha_spec.AlphaSpecHandler(Mock(), Mock())

        item_context = alpha_spec.AlphaSpecHandler.ItemContext()
        with handler.handle_packages(
            item_context, anchor, Mock(), Mock()
        ) as packages_context:
            assert packages_context.parent_context is item_context
            assert (
                packages_context.packages_is_reference_anchor
                == packages_is_reference_anchor
            )

    @pytest.mark.parametrize(
        [
            "package",
            "anchor",
            "content",
            "mode",
            "packages_is_reference_anchor",
            "replacement",
        ],
        [
            *chain(
                *(
                    [
                        pytest.param(
                            p,
                            None,
                            p,
                            "development",
                            False,
                            f"{p}>=0.0.0a0",
                            id=f"{p}-development-no-suffix",
                        ),
                        pytest.param(
                            p,
                            None,
                            p,
                            "release",
                            False,
                            None,
                            id=f"{p}-release-no-suffix",
                        ),
                        pytest.param(
                            p,
                            None,
                            f"{p}>=0.0.0a0",
                            "development",
                            False,
                            None,
                            id=f"{p}-development-suffix",
                        ),
                        pytest.param(
                            p,
                            None,
                            f"{p}>=0.0.0a0",
                            "release",
                            False,
                            p,
                            id=f"{p}-release-suffix",
                        ),
                    ]
                    for p in latest_metadata.prerelease_packages
                )
            ),
            *chain(
                *(
                    [
                        pytest.param(
                            f"{p}-cu12",
                            None,
                            f"{p}-cu12",
                            "development",
                            False,
                            f"{p}-cu12>=0.0.0a0",
                            id=f"{p}-cu12-development-no-suffix",
                        ),
                        pytest.param(
                            f"{p}-cu11",
                            None,
                            f"{p}-cu11",
                            "release",
                            False,
                            None,
                            id=f"{p}-cu11-release-no-suffix",
                        ),
                        pytest.param(
                            f"{p}-cu12",
                            None,
                            f"{p}-cu12>=0.0.0a0",
                            "development",
                            False,
                            None,
                            id=f"{p}-cu12-development-suffix",
                        ),
                        pytest.param(
                            f"{p}-cu11",
                            None,
                            f"{p}-cu11>=0.0.0a0",
                            "release",
                            False,
                            f"{p}-cu11",
                            id=f"{p}-cu11-release-suffix",
                        ),
                    ]
                    for p in latest_metadata.prerelease_packages
                    & latest_metadata.cuda_suffixed_packages
                )
            ),
            *chain(
                *(
                    [
                        pytest.param(
                            f"{p}-cu12",
                            None,
                            f"{p}-cu12",
                            "development",
                            False,
                            None,
                            id=f"{p}-cu12-development-no-suffix",
                        ),
                        pytest.param(
                            f"{p}-cu12",
                            None,
                            f"{p}-cu12>=0.0.0a0",
                            "release",
                            False,
                            None,
                            id=f"{p}-cu12-release-suffix",
                        ),
                    ]
                    for p in latest_metadata.prerelease_packages
                    & (
                        latest_metadata.all_packages
                        - latest_metadata.cuda_suffixed_packages
                    )
                )
            ),
            pytest.param(
                "cuml",
                None,
                "cuml>=24.04,<24.06",
                "development",
                False,
                "cuml>=24.04,<24.06,>=0.0.0a0",
                id="version-range-development-no-suffix",
            ),
            pytest.param(
                "cuml",
                None,
                "cuml>=24.04,<24.06,>=0.0.0a0",
                "release",
                False,
                "cuml>=24.04,<24.06",
                id="version-range-release-suffix",
            ),
            pytest.param(
                "cuml",
                Anchor(AnchorType.DEFINITION, "cuml"),
                "&cuml cuml>=24.04,<24.06",
                "development",
                False,
                "&cuml cuml>=24.04,<24.06,>=0.0.0a0",
                id="anchor-definition-development-no-suffix",
            ),
            pytest.param(
                "cuml",
                Anchor(AnchorType.REFERENCE, "cuml"),
                "&cuml cuml>=24.04,<24.06",
                "development",
                False,
                None,
                id="anchor-reference-development-no-suffix",
            ),
            pytest.param(
                "cuml",
                Anchor(AnchorType.DEFINITION, "cuml"),
                "&cuml cuml>=24.04,<24.06,>=0.0.0a0",
                "release",
                False,
                "&cuml cuml>=24.04,<24.06",
                id="anchor-definition-release-suffix",
            ),
            pytest.param(
                "cuml",
                Anchor(AnchorType.REFERENCE, "cuml"),
                "&cuml cuml>=24.04,<24.06,>=0.0.0a0",
                "release",
                False,
                None,
                id="anchor-reference-release-suffix",
            ),
            pytest.param(
                "cuml",
                None,
                "cuml>=24.04,<24.06",
                "development",
                True,
                None,
                id="packages-is-anchor-reference",
            ),
            pytest.param(
                "packaging",
                None,
                "packaging",
                "development",
                False,
                None,
                id="non-rapids-package",
            ),
            pytest.param(
                None,
                None,
                "--extra-index-url=https://pypi.nvidia.com",
                "development",
                False,
                None,
                id="extra-index-url-development",
            ),
            pytest.param(
                None,
                None,
                "--extra-index-url=https://pypi.nvidia.com",
                "release",
                False,
                None,
                id="extra-index-url-release",
            ),
            pytest.param(
                None,
                None,
                "gcc_linux-64=11.*",
                "development",
                False,
                None,
                id="conda-package-development",
            ),
            pytest.param(
                None,
                None,
                "gcc_linux-64=11.*",
                "release",
                False,
                None,
                id="conda-package-release",
            ),
        ],
    )
    @patch(
        "rapids_pre_commit_hooks.alpha_spec.get_rapids_version",
        Mock(return_value=latest_metadata),
    )
    def test_handle_package(
        self,
        package,
        anchor,
        content,
        mode,
        packages_is_reference_anchor,
        replacement,
    ):
        args = Mock(mode=mode)
        linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
        composed, _ = load_with_anchors(content)
        handler = alpha_spec.AlphaSpecHandler(linter, args)
        item_context = alpha_spec.AlphaSpecHandler.ItemContext(
            has_python_output_type=True
        )
        handler.handle_package(
            alpha_spec.AlphaSpecHandler.PackagesContext(
                item_context, packages_is_reference_anchor
            ),
            anchor,
            composed,
        )
        for package_anchor, package_node in item_context.packages:
            handler._check_package(package_anchor, package_node)
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
                (composed.start_mark.index, composed.end_mark.index),
                replacement,
            )
            assert linter.warnings == expected_linter.warnings


def test_check_alpha_spec():
    CONTENT = "dependencies: []"
    with (
        patch(
            "rapids_pre_commit_hooks.alpha_spec.AlphaSpecHandler", Mock()
        ) as mock_alpha_spec_handler,
        patch(
            "rapids_pre_commit_hooks.alpha_spec.traverse_dependencies_yaml",
            Mock(),
        ) as mock_traverse_dependencies_yaml,
    ):
        args = Mock()
        linter = lint.Linter("dependencies.yaml", CONTENT, "verify-alpha-spec")
        alpha_spec.check_alpha_spec(linter, args)
    mock_alpha_spec_handler.assert_called_once()
    mock_traverse_dependencies_yaml.assert_called_once_with(
        mock_alpha_spec_handler(), CONTENT
    )


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types: pyproject
            +         packages: &packages
            +           - &cudf cudf>=24.04,<24.06
            :             ~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            :             ~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
            +           - *cudf
            """,
            [
                {
                    "warning": "add alpha spec for RAPIDS package cudf",
                    "replacements": ["&cudf cudf>=24.04,<24.06,>=0.0.0a0"],
                }
            ],
            id="python-output",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types: conda
            +         packages: &packages
            +           - &cudf cudf>=24.04,<24.06
            +           - *cudf
            """,
            [],
            id="non-python-output",
        ),
    ],
)
def test_check_alpha_spec_integration(tmp_path, content, warnings):
    content, spans = parse_named_spans(content, dict)

    args = Mock(
        mode="development", rapids_version=None, rapids_version_file="VERSION"
    )
    linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
    with open(os.path.join(tmp_path, "VERSION"), "w") as f:
        f.write(f"{latest_version}\n")
    with set_cwd(tmp_path):
        alpha_spec.check_alpha_spec(linter, args)

    assert linter.warnings == zip_expected_warnings(
        spans.get("warnings", []), warnings
    )

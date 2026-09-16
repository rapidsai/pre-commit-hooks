# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from packaging.requirements import InvalidRequirement, Requirement

from ..utils.dependencies_yaml import (
    Handler,
    is_python_output_type,
)
from ..utils.yaml import Anchor, is_reference_anchor

if TYPE_CHECKING:
    import argparse
    from collections.abc import Generator

    import yaml

    from ..lint import Linter


def is_nvidia_library_package(req: "Requirement") -> bool:
    if req.name == "cuda-toolkit":
        return True
    nvidia_library_packages = {
        "nvidia-cublas",
        "nvidia-cuda-cccl",
        "nvidia-cuda-crt",
        "nvidia-cuda-culibos",
        "nvidia-cuda-cuobjdump",
        "nvidia-cuda-cupti",
        "nvidia-cuda-cuxxfilt",
        "nvidia-cuda-nvcc",
        "nvidia-cuda-nvdisasm",
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-opencl",
        "nvidia-cuda-profiler-api",
        "nvidia-cuda-runtime",
        "nvidia-cuda-sanitizer-api",
        "nvidia-cuda-tileiras",
        "nvidia-cudla",
        "nvidia-cudss",
        "nvidia-cufft",
        "nvidia-cufile",
        "nvidia-curand",
        "nvidia-cusolver",
        "nvidia-cusparse",
        "nvidia-libnvcomp",
        "nvidia-npp",
        "nvidia-nvfatbin",
        "nvidia-nvjitlink",
        "nvidia-nvjpeg",
        "nvidia-nvml-dev",
        "nvidia-nvptxcompiler",
        "nvidia-nvtx",
        "nvidia-nvvm",
    }
    if (
        match := re.search(r"^(?P<package>[a-z-]+)(?:-cu[0-9]+)?$", req.name)
    ) and match.group("package") in nvidia_library_packages:
        return True
    return False


def is_cupy_ctk_package(req: "Requirement") -> bool:
    return bool(
        re.search(r"^cupy-cuda[0-9]+x$", req.name) and "ctk" in req.extras
    )


class UseCUDAWheelsHandler(Handler):
    @dataclass
    class CommonContext:
        common_key: "yaml.Node"

    @dataclass
    class CommonOrMatricesItemContext:
        has_python_output_type: bool = False
        has_use_cuda_wheels: bool = False
        use_cuda_wheels_node: "Optional[yaml.Node]" = None
        suspicious_packages: "list[tuple[yaml.Node, str]]" = field(
            default_factory=list
        )

    @dataclass
    class ItemContext:
        has_python_output_type: bool = False
        matrices_item_contexts: "list[UseCUDAWheelsHandler.CommonOrMatricesItemContext]" = field(  # noqa: E501
            default_factory=list
        )

    @dataclass
    class PackagesContext:
        parent_context: "UseCUDAWheelsHandler.CommonOrMatricesItemContext"
        packages_is_reference_anchor: bool

    def __init__(self, linter: "Linter", args: "argparse.Namespace"):
        self.linter = linter
        self.args = args

    def handle_output_type(
        self,
        output_types_context: (
            "UseCUDAWheelsHandler.CommonOrMatricesItemContext | "
            "UseCUDAWheelsHandler.ItemContext"
        ),
        item: "yaml.Node",
    ) -> None:
        if is_python_output_type(item.value):
            output_types_context.has_python_output_type = True

    @contextlib.contextmanager
    def handle_common(
        self,
        dependency_set_context: "Any",  # noqa: ARG002
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[UseCUDAWheelsHandler.CommonContext]":
        yield UseCUDAWheelsHandler.CommonContext(key)

    @contextlib.contextmanager
    def handle_common_item(
        self,
        common_context: "UseCUDAWheelsHandler.CommonContext",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[UseCUDAWheelsHandler.CommonOrMatricesItemContext]":
        context = UseCUDAWheelsHandler.CommonOrMatricesItemContext()
        yield context

        if context.has_python_output_type:
            for node, name in context.suspicious_packages:
                w = self.linter.add_warning(
                    (node.start_mark.index, node.end_mark.index),
                    f'package "{name}" in common dependency set',
                )
                w.add_note(
                    (
                        common_context.common_key.start_mark.index,
                        common_context.common_key.end_mark.index,
                    ),
                    "place in a specific dependency set with "
                    'use_cuda_wheels: "true" instead',
                )

    @contextlib.contextmanager
    def handle_specific_item(
        self,
        specific_context: "Any",  # noqa: ARG002
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[UseCUDAWheelsHandler.ItemContext]":
        context = UseCUDAWheelsHandler.ItemContext()
        yield context

        if context.has_python_output_type:
            for matrices_item_context in context.matrices_item_contexts:
                self._warn_for_matrices_item(matrices_item_context)

    @contextlib.contextmanager
    def handle_matrices_item(
        self,
        matrices_context: "UseCUDAWheelsHandler.ItemContext",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[UseCUDAWheelsHandler.CommonOrMatricesItemContext]":
        context = UseCUDAWheelsHandler.CommonOrMatricesItemContext()
        yield context

        matrices_context.matrices_item_contexts.append(context)

    def _warn_for_matrices_item(
        self,
        context: "UseCUDAWheelsHandler.CommonOrMatricesItemContext",
    ) -> None:
        if context.has_use_cuda_wheels:
            return
        for node, name in context.suspicious_packages:
            w = self.linter.add_warning(
                (node.start_mark.index, node.end_mark.index),
                f'package "{name}" in specific dependency set without '
                'use_cuda_wheels: "true"',
            )
            if context.use_cuda_wheels_node:
                w.add_note(
                    (
                        context.use_cuda_wheels_node.start_mark.index,
                        context.use_cuda_wheels_node.end_mark.index,
                    ),
                    "place in a specific dependency set with "
                    'use_cuda_wheels: "true" instead',
                )

    def handle_matrix(
        self,
        matrices_item_context: "UseCUDAWheelsHandler.CommonOrMatricesItemContext",  # noqa: E501
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.nullcontext[UseCUDAWheelsHandler.CommonOrMatricesItemContext]":  # noqa: E501
        matrices_item_context.use_cuda_wheels_node = key
        return contextlib.nullcontext(matrices_item_context)

    def handle_matrix_item(
        self,
        matrix_context: "UseCUDAWheelsHandler.CommonOrMatricesItemContext",
        key: "yaml.Node",
        value: "yaml.Node",
    ) -> None:
        if key.value == "use_cuda_wheels":
            matrix_context.use_cuda_wheels_node = value
            if value.value == "true":
                matrix_context.has_use_cuda_wheels = True

    def handle_packages(
        self,
        common_or_matrices_item_context: "UseCUDAWheelsHandler.CommonOrMatricesItemContext",  # noqa: E501
        anchor: "Optional[Anchor]",
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.nullcontext[UseCUDAWheelsHandler.PackagesContext]":
        if common_or_matrices_item_context.use_cuda_wheels_node is None:
            common_or_matrices_item_context.use_cuda_wheels_node = key
        context = UseCUDAWheelsHandler.PackagesContext(
            common_or_matrices_item_context, is_reference_anchor(anchor)
        )
        return contextlib.nullcontext(context)

    def handle_package(
        self,
        packages_context: "UseCUDAWheelsHandler.PackagesContext",
        anchor: "Optional[Anchor]",  # noqa: ARG002
        item: "yaml.Node",
    ) -> None:
        if packages_context.packages_is_reference_anchor:
            return

        try:
            req = Requirement(item.value)
        except InvalidRequirement:
            return
        if is_nvidia_library_package(req):
            packages_context.parent_context.suspicious_packages.append(
                (item, req.name)
            )
        elif is_cupy_ctk_package(req):
            packages_context.parent_context.suspicious_packages.append(
                (item, f"{req.name}[ctk]")
            )

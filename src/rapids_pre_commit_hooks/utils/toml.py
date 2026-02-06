# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import copy
import uuid
from typing import TYPE_CHECKING

import tomlkit

if TYPE_CHECKING:
    from ..lint import Span


def find_value_span(
    document: "tomlkit.TOMLDocument",
    key: tuple[str, ...],
    *,
    append: bool,
) -> "Span":
    """
    Find the exact span of a key in a stringified TOML document.

    Parameters
    ----------
    document : tomlkit.TOMLDocument
        TOML content
    key : tuple[str, ...]
        Tuple of strings, of any length.
        Items are evaluated in order as keys to subset into ``document``.
        For example, to reference the 'license' value in the [project] table
        in a pyproject.toml, ``key = ("project", "license",)``.
    append : bool
        If ``True``, returns the location where new text will be added.
        If ``False``, returns the span of characters to be replaced.

    Returns
    -------
    span : tuple[int, int]
        Span of the key and its value in the document.
        e.g., ``(20, 35)`` = "the 20th-35th characters, including newlines"
          * element 0: number of characters from beginning of the document to
                       beginning of the section indicated by ``key``
          * element 1: final character to replace
    """
    copied_document = copy.deepcopy(document)
    placeholder = uuid.uuid4()
    placeholder_toml = tomlkit.string(str(placeholder))
    placeholder_repr = placeholder_toml.as_string()

    # tomlkit does not provide "mark" information to determine where exactly in
    # the document a value is located, so instead we replace it with a
    # placeholder and look for that in the new document.
    node = copied_document
    while len(key) > (0 if append else 1):
        node = node[key[0]]  # type: ignore[assignment]
        key = key[1:]

    if append:
        node.add(str(placeholder), placeholder_toml)
        value_to_find = f"{placeholder} = {placeholder_repr}"
        begin_pos = copied_document.as_string().find(value_to_find)
        return begin_pos, begin_pos

    # otherwise, if replacing without appending
    old_value = node[key[0]]
    placeholder_value, value_to_find = (
        (
            {str(placeholder): placeholder_toml},
            f"{placeholder} = {placeholder_repr}",
        )
        if isinstance(old_value, tomlkit.items.Table)
        else (str(placeholder), placeholder_repr)
    )
    node[key[0]] = placeholder_value
    begin_pos = copied_document.as_string().find(value_to_find)
    end_pos = begin_pos + len(old_value.as_string())
    return begin_pos, end_pos

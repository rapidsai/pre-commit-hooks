# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import re
from textwrap import dedent
from typing import TYPE_CHECKING

from rapids_pre_commit_hooks.lint import Lines

if TYPE_CHECKING:
    from collections.abc import Generator

    NamedRanges = (
        dict[str, "tuple[int, int] | NamedRanges"]
        | list["tuple[int, int] | NamedRanges"]
    )
    _NamedRanges = dict[str | int, "tuple[int, int] | _NamedRanges"]


_RANGE_LINE_RE: re.Pattern = re.compile(
    r"(?P<range>\^|~+)"
    r"(?P<path>"
    r"(?:[0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\.(?:[0-9]+|[a-zA-Z_][a-zA-Z0-9_]*))*"
    r")"
)


class ParseError(RuntimeError):
    pass


def parse_named_ranges(
    content: str, root_type: type | None = None
) -> "tuple[str, NamedRanges | None]":
    assert root_type is dict or root_type is list or root_type is None
    lines = Lines(dedent(content))
    content = ""
    named_ranges: "_NamedRanges | None" = None

    def get_last_collection(path: tuple[int | str, ...]) -> "_NamedRanges":
        nonlocal named_ranges
        last_collection: "_NamedRanges | None" = named_ranges
        for item in path[:-1]:
            if last_collection is None:
                last_collection = named_ranges = {}
            try:
                next_collection = last_collection[item]
            except KeyError:
                next_collection = last_collection[item] = {}
            if not isinstance(next_collection, dict):
                raise ParseError
            last_collection = next_collection
        if named_ranges is None:
            named_ranges = last_collection = {}
        else:
            assert last_collection is not None
        return last_collection

    start_of_last_line = 0
    end_of_last_line = 0
    for this_pos, next_pos in itertools.pairwise(
        itertools.chain(lines.pos, [(len(lines.content), -1)])
    ):
        line = lines.content[this_pos[0] : this_pos[1]]
        first_two_chars = line[0:2]

        if first_two_chars in {"+ ", "+"}:
            start_of_last_line = len(content)
            end_of_last_line = (
                start_of_last_line
                + this_pos[1]
                - this_pos[0]
                - len(first_two_chars)
            )
            content += lines.content[
                this_pos[0] + len(first_two_chars) : next_pos[0]
            ]
        elif first_two_chars in {": ", ":"}:
            directive_line = line[2:]
            if (pound := directive_line.find("#")) >= 0:
                directive_line = directive_line[:pound]
            end = 0
            for match in _RANGE_LINE_RE.finditer(directive_line):
                if any(
                    filter(
                        lambda c: c != " ", directive_line[end : match.start()]
                    )
                ):
                    raise ParseError
                end = match.end()

                if match.group("range") == "^":
                    range_end = start_of_last_line + match.start("range")
                elif (
                    match.end("range")
                    == end_of_last_line - start_of_last_line + 1
                ):
                    range_end = len(content)
                else:
                    range_end = start_of_last_line + match.end("range")

                range = (start_of_last_line + match.start("range"), range_end)

                if range[0] > end_of_last_line or range[1] > len(content):
                    raise ParseError

                def parse_path_item(item: str) -> str | int:
                    try:
                        return int(item)
                    except ValueError:
                        return item

                path = tuple(
                    map(parse_path_item, match.group("path").split("."))
                )
                last_collection = get_last_collection(path)

                try:
                    existing_range = last_collection[path[-1]]
                except KeyError:
                    last_collection[path[-1]] = range
                else:
                    if not isinstance(existing_range, tuple):
                        raise ParseError
                    if range[0] == existing_range[1]:
                        last_collection[path[-1]] = (
                            existing_range[0],
                            range[1],
                        )
                    elif range[1] == existing_range[0]:
                        last_collection[path[-1]] = (
                            range[0],
                            existing_range[1],
                        )
                    else:
                        raise ParseError

            if any(
                filter(
                    lambda c: c != " ",
                    directive_line[end : len(directive_line)],
                )
            ):
                raise ParseError
        elif line != "" or next_pos[1] >= 0:
            raise ParseError

    def ensure_list_filled(
        collection: "list[None | tuple[int, int] | NamedRanges]",
    ) -> "Generator[tuple[int, int] | NamedRanges]":
        for item in collection:
            if item is None:
                raise ParseError
            yield item

    def postprocess(named_ranges: "_NamedRanges") -> "NamedRanges":
        collection: """
            dict[str, tuple[int, int] | NamedRanges] |
            list[None | tuple[int, int] | NamedRanges] | None
        """ = None
        for k, v in named_ranges.items():
            if isinstance(k, str):
                if collection is None:
                    collection = {}
                if not isinstance(collection, dict):
                    raise ParseError
                collection[k] = postprocess(v) if isinstance(v, dict) else v
            elif isinstance(k, int):
                if collection is None:
                    collection = []
                if not isinstance(collection, list):
                    raise ParseError
                if len(collection) - 1 < k:
                    collection.extend([None] * (k - len(collection) + 1))
                collection[k] = postprocess(v) if isinstance(v, dict) else v

        if isinstance(collection, list):
            return list(ensure_list_filled(collection))

        assert collection is not None
        return collection

    postprocessed = (
        (None if root_type is None else root_type())
        if named_ranges is None
        else postprocess(named_ranges)
    )
    if root_type is not None and not isinstance(postprocessed, root_type):
        raise ParseError
    return content, postprocessed

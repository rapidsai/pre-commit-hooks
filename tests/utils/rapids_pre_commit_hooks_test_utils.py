# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import re
from textwrap import dedent
from typing import TYPE_CHECKING

from rapids_pre_commit_hooks.lint import Lines

if TYPE_CHECKING:
    from typing import TypeGuard

    NamedRanges = (
        dict[str, "tuple[int, int] | NamedRanges"]
        | list["tuple[int, int] | NamedRanges"]
    )
    _NamedRanges = dict[str | int, "tuple[int, int] | _NamedRanges"]


_RANGE_LINE_RE: re.Pattern = re.compile(
    r"(?P<range>\^|>|!|~+)"
    r"(?P<path>"
    r"(?:[0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\.(?:[0-9]+|[a-zA-Z_][a-zA-Z0-9_]*))*"
    r")"
)


class ParseError(RuntimeError):
    pass


def _parse_path_item(item: str) -> str | int:
    try:
        return int(item)
    except ValueError:
        return item


def parse_named_ranges(
    content: str, root_type: type | None = None
) -> "tuple[str, NamedRanges | None]":
    assert root_type is dict or root_type is list or root_type is None
    lines = Lines(dedent(content))
    content = ""
    named_ranges: "_NamedRanges | None" = None
    in_progress_large_groups: dict[tuple[int | str, ...], int] = {}

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
    newline = False
    for this_pos, next_pos in itertools.pairwise(
        itertools.chain(lines.pos, [(len(lines.content), -1)])
    ):
        line = lines.content[this_pos[0] : this_pos[1]]
        first_two_chars = line[0:2]

        if first_two_chars in {"+ ", "+"}:
            newline = True
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
        elif first_two_chars in {"> ", ">"}:
            newline = False
            start_of_last_line = len(content)
            end_of_last_line = (
                start_of_last_line
                + this_pos[1]
                - this_pos[0]
                - len(first_two_chars)
            )
            content += line[len(first_two_chars) :]
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

                path = tuple(
                    map(_parse_path_item, match.group("path").split("."))
                )

                if match.group("range") == ">":
                    if path in in_progress_large_groups:
                        raise ParseError
                    in_progress_large_groups[path] = (
                        start_of_last_line + match.start("range")
                    )
                else:
                    range_start = start_of_last_line + match.start("range")
                    if match.group("range") == "^":
                        range_end = range_start
                    elif match.group("range") == "!":
                        range_end = range_start
                        try:
                            range_start = in_progress_large_groups[path]
                        except KeyError as e:
                            raise ParseError from e
                        del in_progress_large_groups[path]
                    elif (
                        match.end("range")
                        == end_of_last_line - start_of_last_line + 1
                    ):
                        if not newline:
                            raise ParseError
                        range_end = len(content)
                    else:
                        range_end = start_of_last_line + match.end("range")

                    range = (range_start, range_end)

                    if range[0] > end_of_last_line or range[1] > len(content):
                        raise ParseError

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

    if any(in_progress_large_groups):
        raise ParseError

    def is_list_filled(
        collection: "list[None | tuple[int, int] | NamedRanges]",
    ) -> "TypeGuard[list[tuple[int, int] | NamedRanges]]":
        return all(map(lambda i: i is not None, collection))

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

        if isinstance(collection, list) and not is_list_filled(collection):
            raise ParseError

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

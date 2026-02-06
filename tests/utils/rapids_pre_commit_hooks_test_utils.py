# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import re
from textwrap import dedent
from typing import TYPE_CHECKING

from rapids_pre_commit_hooks.lint import Lines

if TYPE_CHECKING:
    from typing import TypeGuard

    from rapids_pre_commit_hooks.lint import Span

    NamedSpans = dict[str, "Span | NamedSpans"] | list["Span | NamedSpans"]
    _NamedSpans = dict[str | int, "Span | _NamedSpans"]


_SPAN_LINE_RE: re.Pattern = re.compile(
    r"(?P<span>\^|>|!|~+)"
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


def parse_named_spans(
    content: str, root_type: type | None = None
) -> "tuple[str, NamedSpans | None]":
    assert root_type is dict or root_type is list or root_type is None
    lines = Lines(dedent(content))
    content = ""
    named_spans: "_NamedSpans | None" = None
    in_progress_large_groups: dict[tuple[int | str, ...], int] = {}

    def get_last_collection(path: tuple[int | str, ...]) -> "_NamedSpans":
        nonlocal named_spans
        last_collection: "_NamedSpans | None" = named_spans
        for item in path[:-1]:
            if last_collection is None:
                last_collection = named_spans = {}
            try:
                next_collection = last_collection[item]
            except KeyError:
                next_collection = last_collection[item] = {}
            if not isinstance(next_collection, dict):
                raise ParseError
            last_collection = next_collection
        if named_spans is None:
            named_spans = last_collection = {}
        else:
            assert last_collection is not None
        return last_collection

    start_of_last_line = 0
    end_of_last_line = 0
    newline = False
    for this_span, next_span in itertools.pairwise(
        itertools.chain(lines.spans, [(len(lines.content), -1)])
    ):
        line = lines.content[this_span[0] : this_span[1]]
        first_two_chars = line[0:2]

        if first_two_chars in {"+ ", "+"}:
            newline = True
            start_of_last_line = len(content)
            end_of_last_line = (
                start_of_last_line
                + this_span[1]
                - this_span[0]
                - len(first_two_chars)
            )
            content += lines.content[
                this_span[0] + len(first_two_chars) : next_span[0]
            ]
        elif first_two_chars in {"> ", ">"}:
            newline = False
            start_of_last_line = len(content)
            end_of_last_line = (
                start_of_last_line
                + this_span[1]
                - this_span[0]
                - len(first_two_chars)
            )
            content += line[len(first_two_chars) :]
        elif first_two_chars in {": ", ":"}:
            directive_line = line[2:]
            if (pound := directive_line.find("#")) >= 0:
                directive_line = directive_line[:pound]
            end = 0
            for match in _SPAN_LINE_RE.finditer(directive_line):
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

                if match.group("span") == ">":
                    if path in in_progress_large_groups:
                        raise ParseError
                    in_progress_large_groups[path] = (
                        start_of_last_line + match.start("span")
                    )
                else:
                    span_start = start_of_last_line + match.start("span")
                    if match.group("span") == "^":
                        span_end = span_start
                    elif match.group("span") == "!":
                        span_end = span_start
                        try:
                            span_start = in_progress_large_groups.pop(path)
                        except KeyError as e:
                            raise ParseError from e
                    elif (
                        match.end("span")
                        == end_of_last_line - start_of_last_line + 1
                    ):
                        if not newline:
                            raise ParseError
                        span_end = len(content)
                    else:
                        span_end = start_of_last_line + match.end("span")

                    span = (span_start, span_end)

                    if max(*span) > len(content):
                        raise ParseError

                    last_collection = get_last_collection(path)

                    try:
                        existing_span = last_collection[path[-1]]
                    except KeyError:
                        last_collection[path[-1]] = span
                    else:
                        if not isinstance(existing_span, tuple):
                            raise ParseError
                        if span[0] == existing_span[1]:
                            last_collection[path[-1]] = (
                                existing_span[0],
                                span[1],
                            )
                        elif span[1] == existing_span[0]:
                            last_collection[path[-1]] = (
                                span[0],
                                existing_span[1],
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
        elif line != "" or next_span[1] >= 0:
            raise ParseError

    if any(in_progress_large_groups):
        raise ParseError

    def is_list_filled(
        collection: "list[None | Span | NamedSpans]",
    ) -> "TypeGuard[list[Span | NamedSpans]]":
        return all(map(lambda i: i is not None, collection))

    def postprocess(named_spans: "_NamedSpans") -> "NamedSpans":
        collection: """
            dict[str, "Span | NamedSpans"] |
            list[None | "Span | NamedSpans"] | None
        """ = None
        for k, v in named_spans.items():
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
        if named_spans is None
        else postprocess(named_spans)
    )
    if root_type is not None and not isinstance(postprocessed, root_type):
        raise ParseError
    return content, postprocessed

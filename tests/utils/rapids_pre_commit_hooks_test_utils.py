# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import re
import warnings
from textwrap import dedent
from typing import TYPE_CHECKING

from rapids_pre_commit_hooks.utils.yaml import node_has_type
from rapids_pre_commit_hooks.lint import Lines, LintWarning, Note, Replacement

if TYPE_CHECKING:
    from typing import Optional, TypeGuard, TypedDict

    import yaml

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


class ParseWarning(RuntimeWarning):
    pass


def _parse_path_item(item: str) -> str | int:
    try:
        return int(item)
    except ValueError:
        return item


def parse_named_spans(
    content: str, root_type: type | None = None
) -> "tuple[str, NamedSpans | None]":
    """Parse a document with named spans.

    This function parses a DSL that allows the developer to write a document
    with named spans interspersed. These named spans can be used to easily pick
    out parts of the document and their exact locations to be used for warning
    locations, replacement locations, and note locations.

    The DSL works as follows:

    - A line beginning with ``+`` adds content to the document with a newline
      at the end.
    - A line beginning with ``>`` adds content to the document with no newline
      at the end.
    - A line beginning with ``:`` adds named spans to the document.

    The named span syntax is as follows:

    - Named spans require one or more components. These spans can be arranged
      hierarchically by specifying the component for each level in the
      hierarchy. Each component can be either an integer or an alphanumeric
      string. If it's an integer, the hierarchy level is a list, and if it's a
      string, the hierarchy level is a dictionary. Components are separated by
      periods and can consist of letters, numbers, and underscores.
    - A named span marked with one or more tildes (``~``) will contain the text
      underlined in the immediately preceding content line. Overlapping named
      spans can be specified by placing them on separate lines, as long as no
      new content lines are introduced between them (if they are, anything span
      lines after the new content line will mark that content line instead.)
      Single named spans can span multiple lines as long as there is no break
      or overlap between the parts.
    - A named span marked with a single caret (``^``) will contain no text but
      will be at the marked location in the immediately preceding content line.
    - A named span can start with ``>`` and end with ``!``. Both of these
      markings require the span's name immediately after. This syntax can be
      used to specify lengthy multi-line spans without adding tons of tildes.
    """
    assert root_type is dict or root_type is list or root_type is None
    lines = Lines(dedent(content))
    content = ""
    named_spans: "_NamedSpans | None" = None
    in_progress_large_spans: dict[tuple[int | str, ...], tuple[int, int]] = {}
    content_line = 0

    def path_tuple_to_str(path: tuple[int | str, ...]) -> str:
        if len(path) == 0:
            return "<root>"
        return ".".join(map(str, path))

    def get_last_collection(path: tuple[int | str, ...]) -> "_NamedSpans":
        nonlocal named_spans
        last_collection: "_NamedSpans | None" = named_spans
        for i, item in enumerate(path[:-1]):
            if last_collection is None:
                last_collection = named_spans = {}
            next_collection = last_collection.setdefault(item, {})
            if not isinstance(next_collection, dict):
                raise ParseError(
                    f'Path "{path_tuple_to_str(path[: i + 1])}" is a span, '
                    "but attempted to access "
                    f'"{path_tuple_to_str(path[: i + 2])}"'
                )
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
            content_line += 1
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
                non_space = list(
                    filter(
                        lambda c: c != " ", directive_line[end : match.start()]
                    )
                )
                if any(non_space):
                    raise ParseError(
                        f'Invalid directive line character: "{non_space[0]}"'
                    )
                end = match.end()

                path = tuple(
                    map(_parse_path_item, match.group("path").split("."))
                )

                if match.group("span") == ">":
                    if path in in_progress_large_spans:
                        raise ParseError(
                            f'Large span "{match.group("path")}" already in '
                            "progress"
                        )
                    in_progress_large_spans[path] = (
                        start_of_last_line + match.start("span"),
                        content_line,
                    )
                else:
                    span_start = start_of_last_line + match.start("span")
                    if match.group("span") == "^":
                        span_end = span_start
                    elif match.group("span") == "!":
                        span_end = span_start
                        try:
                            span_start, span_start_line = (
                                in_progress_large_spans.pop(path)
                            )
                        except KeyError as e:
                            raise ParseError(
                                f'Large span "{match.group("path")}" not '
                                "started yet"
                            ) from e
                        if span_start_line == content_line:
                            warnings.warn(
                                f'Large span "{match.group("path")}" '
                                "is on a single line, consider using ~ "
                                "notation instead",
                                ParseWarning,
                            )
                    elif (
                        match.end("span")
                        == end_of_last_line - start_of_last_line + 1
                    ):
                        if not newline:
                            raise ParseError(
                                f'End of span "{match.group("path")}" '
                                "overruns previous line"
                            )
                        span_end = len(content)
                    else:
                        span_end = start_of_last_line + match.end("span")

                    span = (span_start, span_end)

                    if max(*span) > len(content):
                        raise ParseError(
                            f'End of span "{match.group("path")}" overruns '
                            "previous line"
                        )

                    last_collection = get_last_collection(path)

                    try:
                        existing_span = last_collection[path[-1]]
                    except KeyError:
                        last_collection[path[-1]] = span
                    else:
                        if not isinstance(existing_span, tuple):
                            raise ParseError(
                                f'Path "{match.group("path")}" is not a span'
                            )
                        if span[0] == existing_span[1]:
                            span_start, span_end = last_collection[
                                path[-1]
                            ] = (
                                existing_span[0],
                                span[1],
                            )
                        elif span[1] == existing_span[0]:
                            span_start, span_end = last_collection[
                                path[-1]
                            ] = (
                                span[0],
                                existing_span[1],
                            )
                        else:
                            raise ParseError(
                                "Attempted to create non-contiguous span "
                                f'"{match.group("path")}"'
                            )

                        if "~" in match.group("span"):
                            content_lines = Lines(content)
                            span_line_start = content_lines.line_for_pos(
                                span_start
                            )
                            if (
                                len(content_lines.spans) > span_line_start + 2
                                and span_end
                                > content_lines.spans[span_line_start + 2][0]
                            ):
                                span_line_end = content_lines.line_for_pos(
                                    span_end
                                )
                                warnings.warn(
                                    f'Span "{match.group("path")}" spans '
                                    f"{span_line_end - span_line_start + 1} "
                                    "lines, consider using >/! notation "
                                    "instead",
                                    ParseWarning,
                                )

            non_space = list(
                filter(
                    lambda c: c != " ",
                    directive_line[end : len(directive_line)],
                )
            )
            if any(non_space):
                raise ParseError(
                    f'Invalid directive line character: "{non_space[0]}"'
                )
        elif line != "":
            raise ParseError(f'Invalid line start: "{first_two_chars}"')
        elif next_span[1] >= 0:
            raise ParseError("Only the last line can be blank")

    if any(in_progress_large_spans):
        spans = '", "'.join(
            map(path_tuple_to_str, sorted(in_progress_large_spans.keys()))
        )
        raise ParseError(f'Unfinished large spans: "{spans}"')

    def get_unfilled_items(
        collection: "list[None | Span | NamedSpans]",
    ) -> list[int]:
        return [i for i, item in enumerate(collection) if item is None]

    def is_list_filled(
        _collection: "list[None | Span | NamedSpans]",
        unfilled: list[int],
    ) -> "TypeGuard[list[Span | NamedSpans]]":
        return len(unfilled) == 0

    def postprocess(
        path: tuple[int | str, ...], named_spans: "_NamedSpans"
    ) -> "NamedSpans":
        collection: """
            dict[str, "Span | NamedSpans"] |
            list[None | "Span | NamedSpans"] | None
        """ = None
        for k, v in named_spans.items():
            child_path = (*path, k)
            if isinstance(k, str):
                if collection is None:
                    collection = {}
                if not isinstance(collection, dict):
                    raise ParseError(
                        f'Path "{path_tuple_to_str(path)}" is not a dict, but '
                        f'got a string key "{k}"'
                    )
                collection[k] = (
                    postprocess(child_path, v) if isinstance(v, dict) else v
                )
            elif isinstance(k, int):
                if collection is None:
                    collection = []
                if not isinstance(collection, list):
                    raise ParseError(
                        f'Path "{path_tuple_to_str(path)}" is not a list, but '
                        f"got an integer key {k}"
                    )
                if len(collection) - 1 < k:
                    collection.extend([None] * (k - len(collection) + 1))
                collection[k] = (
                    postprocess(child_path, v) if isinstance(v, dict) else v
                )

        if isinstance(collection, list):
            unfilled = get_unfilled_items(collection)
            if not is_list_filled(collection, unfilled):
                raise ParseError(
                    f'List "{path_tuple_to_str(path)}" is missing items at '
                    f"the following indices: {', '.join(map(str, unfilled))}"
                )

        assert collection is not None
        return collection

    postprocessed = (
        (None if root_type is None else root_type())
        if named_spans is None
        else postprocess((), named_spans)
    )
    if root_type is not None and not isinstance(postprocessed, root_type):
        raise ParseError(
            f"Expected root type to be {root_type.__name__}, got "
            f"{type(postprocessed).__name__}"
        )
    return content, postprocessed


if TYPE_CHECKING:

    class ExpectedWarningSpan(TypedDict):
        warning: "Span"
        notes: "list[Span]"
        replacements: "list[Span]"

    class ExpectedWarning(TypedDict):
        warning: str
        notes: list[str]
        replacements: list[str]


def zip_expected_warnings(
    warning_spans: "list[ExpectedWarningSpan]",
    warnings: "list[ExpectedWarning]",
) -> "list[LintWarning]":
    return [
        LintWarning(
            warning_span["warning"],
            warning["warning"],
            notes=[
                Note(
                    note_span,
                    note,
                )
                for note_span, note in zip(
                    warning_span.get("notes", []),
                    warning.get("notes", []),
                    strict=True,
                )
            ],
            replacements=[
                Replacement(
                    replacement_span,
                    replacement,
                )
                for replacement_span, replacement in zip(
                    warning_span.get("replacements", []),
                    warning.get("replacements", []),
                    strict=True,
                )
            ],
        )
        for warning_span, warning in zip(warning_spans, warnings, strict=True)
    ]


def find_yaml_node_for_span(
    node: "yaml.Node", span: "Span"
) -> "Optional[yaml.Node]":
    if (node.start_mark.index, node.end_mark.index) == span:
        return node
    if node_has_type(node, "map"):
        for key, value in node.value:
            if found := find_yaml_node_for_span(key, span):
                return found
            if found := find_yaml_node_for_span(value, span):
                return found
    if node_has_type(node, "seq"):
        for item in node.value:
            if found := find_yaml_node_for_span(item, span):
                return found
    return None

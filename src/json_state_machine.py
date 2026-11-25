
"""Validate JSON prefixes using a function parameter schema."""

import json
import re
import string


NUMBER_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)"
    r"(?:\.[0-9]*)?"
    r"(?:[eE][+-]?[0-9]*)?"
)

NUMBER_COMPLETE = re.compile(
    r"-?(?:0|[1-9][0-9]*)"
    r"(?:\.[0-9]+)?"
    r"(?:[eE][+-]?[0-9]+)?"
)


def consume_literal(
    text: str,
    position: int,
    expected: str,
) -> tuple[bool, int, bool]:
    """Consume a required JSON fragment."""
    remaining = text[position:]

    if len(remaining) < len(expected):
        valid = expected.startswith(remaining)
        return valid, len(text), False

    if not remaining.startswith(expected):
        return False, position, False

    return True, position + len(expected), True


def consume_number(
    text: str,
    position: int,
) -> tuple[bool, int, bool]:
    """Consume a JSON number or an unfinished number prefix."""
    end = position

    while end < len(text) and text[end] not in ",}":
        end += 1

    number_text = text[position:end]

    if end == len(text):
        valid = (
            number_text in ("", "-")
            or NUMBER_PREFIX.fullmatch(number_text) is not None
        )
        return valid, end, False

    valid = NUMBER_COMPLETE.fullmatch(number_text) is not None

    return valid, end, valid


def consume_string(
    text: str,
    position: int,
) -> tuple[bool, int, bool]:
    """Consume a JSON string, including escape sequences."""
    if position == len(text):
        return True, position, False

    if text[position] != '"':
        return False, position, False

    index = position + 1
    escaped = False
    unicode_left = 0

    while index < len(text):
        char = text[index]
        index += 1

        if unicode_left > 0:
            if char not in string.hexdigits:
                return False, index, False

            unicode_left -= 1
            continue

        if escaped:
            escaped = False

            if char == "u":
                unicode_left = 4
            elif char not in '"\\/bfnrt':
                return False, index, False

            continue

        if char == "\\":
            escaped = True
        elif char == '"':
            return True, index, True
        elif ord(char) < 32:
            return False, index, False

    return True, index, False


def consume_boolean(text: str, position: int) -> tuple[bool, int, bool]:
    """Consume a JSON boolean or an incomplete boolean prefix."""
    remaining = text[position:]
    for literal in ("true", "false"):
        if remaining.startswith(literal):
            return True, position + len(literal), True

    if remaining and any(
        literal.startswith(remaining)
        for literal in ("true", "false")
    ):
        return True, len(text), False

    if remaining == "":
        return True, position, False

    return False, position, False


def check_json_prefix(
    text: str,
    schema: list[tuple[str, str]],
) -> tuple[bool, bool]:
    """Check whether text is a valid JSON prefix or object."""
    if not schema:
        return text in ("", "{", "{}"), text == "{}"

    position = 0

    for index, (name, value_type) in enumerate(schema):
        key = json.dumps(name)

        if index == 0:
            expected = "{" + key + ":"
        else:
            expected = ", " + key + ":"

        valid, position, finished = consume_literal(
            text, position, expected
        )

        if not valid:
            return False, False

        if not finished:
            return True, False

        while position < len(text) and text[position] in " \t\n\r":
            position += 1

        if value_type == "number":
            valid, position, finished = consume_number(
                text, position
            )
        elif value_type == "string":
            valid, position, finished = consume_string(
                text, position
            )
        elif value_type == "boolean":
            valid, position, finished = consume_boolean(text, position)
        else:
            raise ValueError(
                f"Unsupported type: {value_type}"
            )

        if not valid:
            return False, False

        if not finished:
            return True, False

    valid, position, finished = consume_literal(
        text, position, "}"
    )

    complete = valid and finished and position == len(text)

    return valid and position == len(text), complete

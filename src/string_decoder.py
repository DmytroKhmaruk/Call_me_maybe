import json
import string
from llm_sdk import Small_LLM_Model


def check_string_prefix(text: str) -> tuple[bool, bool]:
    """Check whether text is a valid JSON string prefix.
    Return (is_valid_prefix, is_complete).
    """
    if not text.startswith('"'):
        return False, False

    position = 1
    escaped = False
    unicode_left = 0
    closed = False
    while position < len(text):
        char = text[position]
        position += 1

        if closed:
            if char.isspace():
                continue
            if char == "}" and not text[position:].strip():
                return True, True
            return False, False

        if unicode_left:
            if char not in string.hexdigits:
                return False, False
            unicode_left -= 1
            continue

        if escaped:
            escaped = False
            if char == "u":
                unicode_left = 4
            elif char not in '"\\/bfnrt':
                return False, False
            continue

        if char == "\\":
            escaped = True
        elif char == '"':
            closed = True
        elif ord(char) < 32:
            return False, False

    return True, False


def generate_string_argument(model: Small_LLM_Model, prompt: str,
                             max_tokens: int = 80) -> str:
    """Generate a JSON string value using token constraints."""
    input_ids = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        ranked_ids = sorted(
            range(len(logits)),
            key=lambda i: logits[i],
            reverse=True,
        )
        chosen_id: int | None = None
        complete = False
        chosen_text = ""
        for token_id in ranked_ids:
            candidate_ids = generated_ids + [token_id]
            candidate_text = model.decode(candidate_ids)
            valid, finished = check_string_prefix(candidate_text)
            if valid:
                chosen_id = token_id
                complete = finished
                chosen_text = candidate_text
                break

        if chosen_id is None:
            raise ValueError("No valid string token found")

        generated_ids.append(chosen_id)
        input_ids.append(chosen_id)
        if complete:
            parsed = json.loads('{"value": ' + chosen_text)
            value = parsed["value"]

            if not isinstance(value, str):
                raise ValueError("Expected a string value")

            return value

    raise ValueError("String generation was not completed")

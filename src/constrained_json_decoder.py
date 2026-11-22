import json

from llm_sdk import Small_LLM_Model

from .json_state_machine import check_json_prefix
from .models import FunctionDefinition


def generate_constrained_json(
    model: Small_LLM_Model, user_prompt: str,
    function: FunctionDefinition, max_tokens: int = 100
) -> dict[str, int | float | str]:
    """Generate numeric parameters constrained by a JSON schema."""
    names = list(function.parameters)
    schema = [
        (name, definition.type)
        for name, definition in function.parameters.items()
    ]
    if not names:
        return {}

    for definition in function.parameters.values():
        if definition.type not in ("number", "string"):
            raise ValueError(f"Unsupported parameter type: {definition.type}")

    prompt = (
        "You are a function calling assistant.\n"
        "Convert the request into JSON arguments.\n"
        "Follow the function description and parameter types.\n"
        "Generate valid JSON only.\n"
        "Preserve the exact source text.\n"
        "For regex parameters, distinguish literal patterns "
        "from character classes and repeated matches.\n\n"

        "Example 1:\n"
        "Request: Replace every digit in 'Room 82' with X.\n"
        "JSON: "
        '{"source_string": "Room 82", '
        '"regex": "\\\\d", '
        '"replacement": "X"}\n\n'

        "Example 2:\n"
        "Request: Replace all lowercase vowels in "
        "'banana' with underscores.\n"
        "JSON: "
        '{"source_string": "banana", '
        '"regex": "[aeiou]", '
        '"replacement": "_"}\n\n'

        "Example 3:\n"
        "Request: Replace the word 'red' in "
        "'red car' with blue.\n"
        "JSON: "
        '{"source_string": "red car", '
        '"regex": "red", '
        '"replacement": "blue"}\n\n'

        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {', '.join(names)}\n"
        f"Request: {user_prompt}\n"
        "JSON: "
    )
    print("\n========== LLM PROMPT ==========")
    print(prompt)
    print("================================\n")

    input_ids = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []
    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        if '"regex": "' in model.decode(generated_ids):
            current_text = model.decode(generated_ids)

            regex_prefix = current_text.split('"regex": "', 1)[1]

            if regex_prefix == "":
                print("\n=== REGEX FIRST TOKEN ===")

                ranked = sorted(
                    range(len(logits)),
                    key=lambda token_id: logits[token_id],
                    reverse=True,
                )

                print("Top 15 raw tokens:")

                for token_id in ranked[:15]:
                    token_text = model.decode([token_id])

                    print(
                        f"{token_text!r}: {logits[token_id]:.4f}"
                    )

                print("\nTokens starting with '[':")

                count = 0

                for token_id in ranked:
                    token_text = model.decode([token_id])

                    if token_text.startswith("["):
                        print(
                            f"{token_text!r}: "
                            f"{logits[token_id]:.4f}"
                        )
                        count += 1

                    if count == 5:
                        break

        ranked_ids = sorted(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
            reverse=True
        )
        selected_id: int | None = None
        selected_text = ""
        is_complete = False
        current_text = model.decode(generated_ids)
        for token_id in ranked_ids:
            candidate_ids = generated_ids + [token_id]
            candidate_text = model.decode(candidate_ids)

            if candidate_text == current_text:
                continue

            valid, complete = check_json_prefix(candidate_text, schema)

            if valid:
                selected_id = token_id
                selected_text = candidate_text
                is_complete = complete
                break

        if selected_id is None:
            raise ValueError("No valid next token found")

        generated_ids.append(selected_id)
        input_ids.append(selected_id)
        print(f"Step {step + 1}: {selected_text!r}")

        if is_complete:
            result = json.loads(selected_text)

            if list(result) != names:
                raise ValueError("Incorrect parameter names")

            for name, definition in function.parameters.items():
                value = result[name]

                if definition.type == "number":
                    if type(value) not in (int, float):
                        raise ValueError(f"Invalid number: {name}")

                elif definition.type == "string":
                    if not isinstance(value, str):
                        raise ValueError(f"Invalid string: {name}")

            return result

    raise ValueError("JSON generation exceeded the token limit")

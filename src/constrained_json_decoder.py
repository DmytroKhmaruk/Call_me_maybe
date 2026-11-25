import json
from llm_sdk import Small_LLM_Model
from .json_state_machine import check_json_prefix
from .models import FunctionDefinition
from .token_selector import select_valid_token


def generate_constrained_json(
    model: Small_LLM_Model, user_prompt: str,
    function: FunctionDefinition, max_tokens: int = 100
) -> dict[str, int | float | str | bool]:
    """Generate schema-constrained JSON function arguments."""
    names = list(function.parameters)
    schema = [
        (name, definition.type)
        for name, definition in function.parameters.items()
    ]
    if not names:
        return {}

    for definition in function.parameters.values():
        if definition.type not in ("number", "string", "boolean"):
            raise ValueError(f"Unsupported parameter type: {definition.type}")

    parameter_schema = "\n".join(
        f"- {name}: type={definition.type}"
        for name, definition in function.parameters.items()
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise function calling assistant.\n"
                "Extract function arguments from the request.\n"
                "Follow the function description and parameter types.\n"
                "Preserve exact numbers, signs and source text.\n"
                "For regex parameters, generate a regex pattern "
                "that describes the requested matches.\n"
                "Distinguish literal text, character classes, "
                "digit patterns, and repetition.\n"
                "When a request refers to all occurrences of a "
                "category, the regex must match every relevant "
                "occurrence.\n"
                "Request: Replace all numeric sequences in "
                "'Order 17 contains 250 items' with X.\n"
                "Regex: \\\\d+\n"
                "Request: Replace all vowels in 'banana' with X.\n"
                "Regex: [aeiou]\n"
                "Return only a valid JSON object.\n"
                "Do not provide explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Function: {function.name}\n"
                f"Description: {function.description}\n"
                f"Parameter schema:\n{parameter_schema}\n"
                f"Request: {user_prompt}"
            ),
        },
    ]

    prompt = ""

    for message in messages:
        prompt += (
            f"<|im_start|>{message['role']}\n"
            f"{message['content']}<|im_end|>\n"
        )

    prompt += (
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    )

    input_ids = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        current_text = model.decode(generated_ids)

        def is_valid_token(token_id: int) -> bool:
            candidate_ids = generated_ids + [token_id]
            candidate_text = model.decode(candidate_ids)

            if candidate_text == current_text:
                return False

            valid, _ = check_json_prefix(
                candidate_text,
                schema,
            )

            return valid

        selected_id = select_valid_token(
            logits,
            is_valid_token,
            top_k=128
        )

        generated_ids.append(selected_id)
        input_ids.append(selected_id)

        selected_text = model.decode(generated_ids)

        _, is_complete = check_json_prefix(
            selected_text,
            schema,
        )

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
                elif definition.type == "boolean":
                    if type(value) is not bool:
                        raise ValueError(f"Invalid boolean: {name}")

            return result

    raise ValueError("JSON generation exceeded the token limit")

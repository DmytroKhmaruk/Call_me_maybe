import re
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition


def build_argument_prompt(user_prompt: str, function: FunctionDefinition,
                          parameter_name: str) -> str:
    """Build a prompt for generating a numeric argument."""
    return (
        "Extract the exact numeric value for the target parameter.\n"
        "Preserve negative signs and decimal points.\n"
        "Return only the JSON number followed by }.\n\n"
        "Example 1:\n"
        "Request: Add -7.5 and 3.25\n"
        "Target parameter: a\n"
        'JSON: {"a": -7.5}\n\n'
        "Example 2:\n"
        "Request: Add 10.5 and -4.25\n"
        "Target parameter: b\n"
        'JSON: {"b": -4.25}\n\n'
        f"{user_prompt}\n"
        f"Function description: {function.description}\n"
        f"Target parameter: {parameter_name}\n"
        "JSON: "
        + '{"' + parameter_name + '": '
    )


NUMBER_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)?"
    r"(?:\.[0-9]*)?"
    r"(?:[eE][+-]?[0-9]*)?"
)
COMPLETE_NUMBER = re.compile(
    r"-?(?:0|[1-9][0-9]*)"
    r"(?:\.[0-9]+)?"
    r"(?:[eE][+-]?[0-9]+)?"
)


def check_number_prefix(text: str) -> tuple[bool, bool]:
    """Check a JSON number prefix and its completion."""
    stripped = text.lstrip()

    if not stripped:
        return True, False

    if stripped.startswith("}"):
        return False, False

    if stripped.endswith("}"):
        number_text = stripped[:-1].strip()

        if COMPLETE_NUMBER.fullmatch(number_text):
            return True, True

        return False, False

    if stripped != stripped.rstrip():
        return (
            COMPLETE_NUMBER.fullmatch(stripped.strip()) is not None,
            False,
        )

    return NUMBER_PREFIX.fullmatch(stripped) is not None, False


def generate_number_argument(model: Small_LLM_Model, prompt: str,
                             max_tokens: int = 30) -> int | float:
    """Generate a JSON number using constrained decoding."""
    input_ids = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        ranked_ids = sorted(
            range(len(logits)),
            key=lambda i: logits[i],
            reverse=True,
        )

        if not generated_ids:
            print("\nTop 15 raw tokens:")

            for token_id in ranked_ids[:15]:
                token_text = model.decode([token_id])

                valid, finished = check_number_prefix(token_text)

                print(
                    f"ID: {token_id}, "
                    f"Token: {token_text!r}, "
                    f"Logit: {logits[token_id]:.4f}, "
                    f"Valid: {valid}"
                )

            minus_ids = model.encode("-")[0].tolist()

            print("\nMinus token candidates:")

            for token_id in minus_ids:
                print(
                    f"Token: {model.decode([token_id])!r}, "
                    f"Logit: {logits[token_id]:.4f}, "
                    f"Valid: {check_number_prefix(model.decode([token_id]))[0]}"
                )

        chosen_id: int | None = None
        chosen_text = ""
        completed = False
        current_text = model.decode(generated_ids)
        for token_id in ranked_ids:
            candidate_ids = generated_ids + [token_id]
            candidate_text = model.decode(candidate_ids)
            if candidate_text == current_text:
                continue

            valid, finished = check_number_prefix(candidate_text)
            if valid:
                chosen_id = token_id
                chosen_text = candidate_text
                completed = finished
                break

        if chosen_id is None:
            raise ValueError("No valid numeric token found")

        generated_ids.append(chosen_id)
        input_ids.append(chosen_id)
        print(f"Generated: {chosen_text!r}")
        if completed:
            number_text = chosen_text.strip()[:-1].strip()

            if not COMPLETE_NUMBER.fullmatch(number_text):
                raise ValueError("Invalid generated number")

            if "." in number_text or "e" in number_text.lower():
                return float(number_text)

            return int(number_text)

    raise ValueError("Number generation was not completed")

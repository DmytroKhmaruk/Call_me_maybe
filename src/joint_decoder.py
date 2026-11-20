import json
import re
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition


NUMBER_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)?"
    r"(?:\.[0-9]*)?"
    r"(?:[eE][+-]?[0-9]*)?"
)

NUMBER_COMPLETE = re.compile(
    r"-?(?:0|[1-9][0-9]*)"
    r"(?:\.[0-9]+)?"
    r"(?:[eE][+-]?[0-9]+)?"
)


def check_number(text: str) -> tuple[bool, bool, str | None]:
    """Check a numeric prefix or a terminated JSON number."""
    candidate = text.strip()

    if NUMBER_PREFIX.fullmatch(candidate):
        return True, False, None

    match = re.fullmatch(
        r"(-?(?:0|[1-9][0-9]*)"
        r"(?:\.[0-9]+)?"
        r"(?:[eE][+-]?[0-9]+)?)"
        r"\s*[,}]\s*",
        candidate,
    )

    if match is not None:
        return True, True, match.group(1)

    return False, False, None


def generate_number(model: Small_LLM_Model, context: str, max_tokens: int = 30
                    ) -> int | float:
    """Generate one number inside a shared JSON context."""
    generated_ids: list[int] = []
    for _ in range(max_tokens):
        generated_text = model.decode(generated_ids)
        input_ids = model.encode(context + generated_text)[0].tolist()
        logits = model.get_logits_from_input_ids(input_ids)
        
        if not generated_ids:
            print("\nFirst-token diagnostics:")

            candidates = sorted(
                range(len(logits)),
                key=lambda token_id: logits[token_id],
                reverse=True,
            )[:10]

            for token_id in candidates:
                print(
                    f"{model.decode([token_id])!r}: "
                    f"{logits[token_id]:.4f}"
                )

            minus_id = model.encode("-")[0].tolist()

            for token_id in minus_id:
                print(
                    f"Minus token {token_id}: "
                    f"{logits[token_id]:.4f}"
                )

        ranked_ids = sorted(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
            reverse=True,
        )
        chosen_id: int | None = None
        completed_number: str | None = None
        for token_id in ranked_ids:
            candidate_ids = generated_ids + [token_id]
            candidate_text = model.decode(candidate_ids)
            if candidate_text == generated_text:
                continue

            valid, finished, number_text = check_number(candidate_text)

            if valid:
                chosen_id = token_id

                if finished:
                    completed_number = number_text

                break

        if chosen_id is None:
            raise ValueError("No valid numeric token found")

        generated_ids.append(chosen_id)
        print(f"Generated: {model.decode(generated_ids)!r}")
        if completed_number is not None:
            value = json.loads(completed_number)

            if type(value) not in (int, float):
                raise ValueError("Expected a JSON number")

            return value

    raise ValueError("Numeric generation was not completed")


def generate_joint_arguments(model: Small_LLM_Model, user_prompt: str,
                             function: FunctionDefinition
                             ) -> dict[str, int | float]:
    """Generate numeric arguments in one shared context."""
    parameters: dict[str, int | float] = {}
    for definition in function.parameters.values():
        if definition.type != "number":
            raise ValueError("This decoder supports numeric parameters only")

    names = list(function.parameters)
    description = ", ".join(names)
    instruction = (
        "Extract function arguments from the request.\n"
        "Preserve the exact numbers, signs and decimal parts.\n"
        "Use the order and meaning of the request.\n"
        "Do not repeat a number unless requested.\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters in order: {description}\n"
        f"Request: {user_prompt}\n"
        "JSON: "
    )
    json_text = "{"
    for index, name in enumerate(names):
        if index > 0:
            json_text += ", "

        json_text += f"{json.dumps(name)}: "
        print(f"Generating parameter: {name}")
        value = generate_number(model, instruction + json_text)
        parameters[name] = value
        json_text += json.dumps(value)

    json_text += "}"
    parsed = json.loads(json_text)
    if parsed != parameters:
        raise ValueError("Generated parameters are inconsistent")

    print("Generated JSON:", json_text)

    return parameters

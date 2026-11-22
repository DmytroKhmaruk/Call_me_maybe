
"""Score numeric argument candidates using Qwen logits."""

import math
import re
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .file_handler import load_functions
from .models import FunctionDefinition


NUMBER_PATTERN = re.compile(
    r"(?<![\w.])"
    r"-?(?:0|[1-9][0-9]*)"
    r"(?:\.[0-9]+)?"
    r"(?:[eE][+-]?[0-9]+)?"
    r"(?![\w.])"
)


def extract_numbers(text: str) -> list[str]:
    """Extract numeric candidates from a request."""
    return list(dict.fromkeys(
        match.group(0)
        for match in NUMBER_PATTERN.finditer(text)
    ))


def score_candidate(model: Small_LLM_Model, context: str, candidate: str
                    ) -> tuple[float, float]:
    """Compute total and average token log-probabilities."""
    input_ids = model.encode(context)[0].tolist()
    candidate_ids = model.encode(candidate)[0].tolist()

    if not candidate_ids:
        raise ValueError("Empty candidate")

    total_score = 0.0
    for token_id in candidate_ids:
        logits = model.get_logits_from_input_ids(input_ids)
        maximum = max(logits)
        log_total = maximum + math.log(
            sum(math.exp(float(value) - maximum) for value in logits))
        log_probability = float(logits[token_id]) - log_total
        total_score += log_probability
        input_ids.append(token_id)

    average_score = total_score / len(candidate_ids)

    return total_score, average_score


def compare_candidates(
    model: Small_LLM_Model, user_prompt: str, function: FunctionDefinition,
    parameter_name: str, candidates: list[str],
    previous_parameters: dict[str, str]
) -> None:
    """Display candidate scores for one parameter."""
    context = (
        "Extract the exact numeric value for the target parameter.\n"
        "Preserve negative signs and decimal points.\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters in order: {', '.join(function.parameters)}\n"
        f"Already extracted: {previous_parameters}\n"
        f"Request: {user_prompt}\n"
        f"Target parameter: {parameter_name}\n"
        "Value: "
    )

    print(f"\nParameter: {parameter_name}")

    for candidate in candidates:
        total, average = score_candidate(model, context, candidate)

        print(
            f"Candidate: {candidate:>10} | "
            f"Total: {total:>9.4f} | "
            f"Average: {average:>9.4f}"
        )


def main() -> None:
    """Compare numeric candidates for two parameters."""
    functions = load_functions(
        Path("data/input/functions_definition.json")
    )

    function = next(
        function
        for function in functions
        if function.name == "fn_add_numbers"
    )

    model = Small_LLM_Model()

    user_prompt = "Add 12.5 and -4.25"

    candidates = extract_numbers(user_prompt)

    print("Candidates:", candidates)

    compare_candidates(
        model,
        user_prompt,
        function,
        "a",
        candidates,
        {},
    )

    compare_candidates(
        model,
        user_prompt,
        function,
        "b",
        candidates,
        {"a": "12.5"},
    )


if __name__ == "__main__":
    main()

from .models import FunctionDefinition
import math
from llm_sdk import Small_LLM_Model


def build_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build a prompt with available function descriptions."""
    lines = [
        "Choose the function that best matches the user request.",
        "Return only the exact function name.",
        "",
        "Available functions:",
    ]

    for function in functions:
        lines.append(f"{function.name}: {function.description}")

    lines.extend(["", f"User request: {user_prompt}", "Function name: "])
    return "\n".join(lines)


def score_function(model: Small_LLM_Model, prompt: str, function_name: str
                   ) -> float:
    """Calculate the log-probability of a function name."""
    input_ids = model.encode(prompt)[0].tolist()
    candidate_ids = model.encode(function_name)[0].tolist()
    total_score = 0.0
    for token_id in candidate_ids:
        logits = model.get_logits_from_input_ids(input_ids)
        maximum = max(logits)
        log_sum_exp = maximum + math.log(
            sum(math.exp(value - maximum) for value in logits)
        )
        token_score = logits[token_id] - log_sum_exp
        total_score += token_score
        input_ids.append(token_id)

    return total_score


def select_function(model: Small_LLM_Model, user_prompt: str,
                    functions: list[FunctionDefinition]) -> FunctionDefinition:
    """Select a function using LLM token probabilities."""
    if not functions:
        raise ValueError("No functions available")

    prompt = build_prompt(user_prompt, functions)
    best_function = functions[0]
    best_score = float("-inf")
    for function in functions:
        score = score_function(model, prompt, function.name)
        print(f"{function.name}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_function = function

    return best_function

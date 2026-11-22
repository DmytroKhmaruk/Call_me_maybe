import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .file_handler import load_functions
from .function_selector import select_function
from .constrained_json_decoder import generate_constrained_json


def main() -> None:
    """Select a function and generate its arguments."""
    try:
        functions = load_functions(
            Path("data/input/functions_definition.json")
        )

        model = Small_LLM_Model()

        user_prompt = (
            "Replace all vowels in 'Programming is fun' "
            "with asterisks using regex."
        )

        selected = select_function(
            model,
            user_prompt,
            functions,
        )

        print(f"Selected function: {selected.name}")

        parameters = generate_constrained_json(
            model,
            user_prompt,
            selected,
        )

        result = {
            "prompt": user_prompt,
            "name": selected.name,
            "parameters": parameters,
        }

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    except ValueError as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()

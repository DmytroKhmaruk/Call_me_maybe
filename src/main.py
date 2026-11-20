import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .file_handler import load_functions
from .function_selector import select_function
from .joint_decoder import generate_joint_arguments


def main() -> None:
    """Select a function and generate its arguments."""
    try:
        functions = load_functions(
            Path("data/input/functions_definition.json")
        )

        model = Small_LLM_Model()

        user_prompt = "Add 12.5 and -4.25"

        selected = select_function(
            model,
            user_prompt,
            functions,
        )

        print(f"Selected function: {selected.name}")

        parameters = generate_joint_arguments(
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

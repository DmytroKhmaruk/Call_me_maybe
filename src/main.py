from pathlib import Path
from .file_handler import load_functions, load_prompts


def main() -> None:
    input_dir = Path("data/input")
    try:
        functions = load_functions(
            input_dir / "functions_definition.json"
        )
        prompts = load_prompts(
            input_dir / "function_calling_tests.json"
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print(f"Loaded {len(functions)} functions")
    print(f"Loaded {len(prompts)} prompts")
    print("\nAvailable functions:")
    for function in functions:
        print(f"- {function.name}")
        for name, definition in function.parameters.items():
            print(f"    {name}: {definition.type}")

    print("\nFirst prompt:")
    if prompts:
        print(prompts[0].prompt)


if __name__ == "__main__":
    main()

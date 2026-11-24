
"""Main entry point for the Call Me Maybe project."""

import argparse
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .constrained_json_decoder import generate_constrained_json
from .file_handler import load_functions, load_prompts
from .function_selector import select_function
from .output_handler import save_results
from time import perf_counter


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-based constrained function calling."
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the input prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
        help="Path to the output JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    """Process all prompts and save the function calls."""
    args = parse_arguments()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)

        if not functions:
            raise ValueError("No functions were provided")

        print(f"Loaded {len(functions)} functions")
        print(f"Loaded {len(prompts)} prompts")

        results: list[dict[str, object]] = []

        if prompts:
            model = Small_LLM_Model()
            decoding_times = []
            for index, request in enumerate(prompts, start=1):
                print(
                    f"\nProcessing {index}/{len(prompts)}: "
                    f"{request.prompt}"
                )

                selected = select_function(
                    model,
                    request.prompt,
                    functions,
                )

                print(f"Selected function: {selected.name}")
                start = perf_counter()
                parameters = generate_constrained_json(
                    model,
                    request.prompt,
                    selected,
                )
                elapsed = perf_counter() - start
                decoding_times.append(elapsed)
                print(f"Constrained decoding: {elapsed:.3f} seconds")
                result = {
                    "prompt": request.prompt,
                    "name": selected.name,
                    "parameters": parameters,
                }

                results.append(result)

        if decoding_times:
            total = sum(decoding_times)
            average = total / len(decoding_times)

            print("\n=== PERFORMANCE SUMMARY ===")
            print(f"Total decoding time: {total:.3f} s")
            print(f"Average per request: {average:.3f} s")
            print(f"Fastest request: {min(decoding_times):.3f} s")
            print(f"Slowest request: {max(decoding_times):.3f} s")

        save_results(results, args.output)

        print("\nAll prompts processed successfully!")

    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

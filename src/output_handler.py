import json
from pathlib import Path
from typing import Any


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save function calls to the specified JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")

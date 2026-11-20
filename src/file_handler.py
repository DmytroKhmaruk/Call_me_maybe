"""Load and validate function-calling input files."""
import json
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from .models import FunctionDefinition, PromptDefinition


def load_json_file(file_path: Path) -> object:
    """Read and parse a JSON file."""
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file '{file_path}': {exc}"
                         ) from exc


def load_functions(file_path: Path) -> list[FunctionDefinition]:
    """Load and validate available function definitions."""
    data = load_json_file(file_path)
    adapter = TypeAdapter(list[FunctionDefinition])
    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid function definitions: {exc}") from exc


def load_prompts(file_path: Path) -> list[PromptDefinition]:
    """Load and validate user prompts."""
    data = load_json_file(file_path)
    adapter = TypeAdapter(list[PromptDefinition])
    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid prompt data: {exc}") from exc

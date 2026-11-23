import re


def validate_regex(pattern: str) -> bool:
    """Return whether a regex pattern is syntactically valid."""
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def inspect_regex(source: str, pattern: str, replacement: str
                  ) -> dict[str, object]:
    """Inspect matches and substitutions for a regex pattern."""
    try:
        compiled = re.compile(pattern)
        matches = [match.group(0) for match in compiled.finditer(source)]
        result, count = compiled.subn(replacement, source)
        return {
            "valid": True,
            "matches": matches,
            "match_count": count,
            "result": result
        }

    except re.error as exc:
        return {"valid": False, "error": str(exc)}


def compare_regex_candidates(source: str, candidates: list[str],
                             replacement: str) -> None:
    """Display the behavior of several regex candidates."""
    for pattern in candidates:
        info = inspect_regex(source, pattern, replacement)
        print(f"\nCandidate: {pattern!r}")
        print(f"Valid: {info['valid']}")

        if info["valid"]:
            print(f"Matches: {info['matches']}")
            print(f"Count: {info['match_count']}")
            print(f"Result: {info['result']}")
        else:
            print(f"Error: {info['error']}")

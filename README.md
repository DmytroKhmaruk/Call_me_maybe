*This project has been created as part of the 42 curriculum by dkhmaruk.*

# Call Me Maybe

## Description

**Call Me Maybe** converts natural-language requests into structured function calls using a local language model. It reads available function definitions and user prompts, selects the appropriate function using model-generated token scores, and extracts arguments into a JSON object conforming to the selected function's parameter schema.

The program **does not execute functions** or calculate their return values. Its output identifies which function should be called and with which arguments.

The implementation uses **Qwen/Qwen3-0.6B** through the provided `llm_sdk` (`Small_LLM_Model`) and applies **constrained decoding** during argument generation. Rather than trusting the language model to format JSON correctly, it rejects token continuations that violate the expected JSON structure or parameter types.

## Instructions

### Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- The supplied `llm_sdk` package in the repository, alongside `src/`
- Model files accessible to `llm_sdk` (a first run may require downloading them)

Install dependencies from the repository root:

```bash
uv sync
```

Alternatively:

```bash
make install
```

### Run

With the default paths:

```bash
uv run python -m src
```

With explicit input and output paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The equivalent Makefile target is:

```bash
make run
```

Other Makefile targets include `debug`, `clean`, and `lint`. To check code style and static types, run:

```bash
make lint
```

The program reads function definitions and prompts from `data/input/` by default. The output handler creates the destination directory when necessary and writes results to `data/output/function_calls.json`, unless `--output` specifies another path.

## Example Usage

Example function definition (one item in `functions_definition.json`):

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": {"type": "number"},
    "b": {"type": "number"}
  },
  "returns": {"type": "number"}
}
```

Example input (`function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

Expected output (`function_calls.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

Every output entry contains exactly three fields: `prompt`, `name`, and `parameters`.

The supported argument types are:

| Schema type | Example JSON value | Representation |
| --- | --- | --- |
| `number` | `42.0` | Converted to a floating-point value before output |
| `integer` | `42` | Integer without a fraction or exponent |
| `string` | `"hello"` | JSON string |
| `boolean` | `true` | JSON boolean (`true` or `false`) |

Support for `integer` is an additional implementation feature. The output preserves its integer representation, while `number` arguments are converted to floating-point values.

## Project Architecture

| Module | Responsibility |
| --- | --- |
| `src/main.py` | CLI arguments, processing pipeline, error reporting, and timing summary |
| `src/file_handler.py` | Read JSON input files and validate their contents |
| `src/models.py` | Pydantic models for functions, parameters, and prompts |
| `src/function_selector.py` | Choose a function using LLM token log-probabilities |
| `src/constrained_json_decoder.py` | Build the generation context and generate schema-constrained arguments |
| `src/json_state_machine.py` | Check whether a partial JSON response can still satisfy the schema |
| `src/token_selector.py` | Select the highest-scoring valid token with a fallback search |
| `src/output_handler.py` | Serialize completed function calls to JSON |

## Algorithm Explanation

### 1. Input loading and validation

`file_handler.py` reads the input JSON files. `models.py` uses Pydantic to validate function names, descriptions, parameter declarations, and user prompts. Invalid or missing input files produce error messages instead of being silently accepted.

### 2. Function selection

`function_selector.py` constructs a prompt containing the user request and available function descriptions. It scores each candidate function name token-by-token using `Small_LLM_Model.get_logits_from_input_ids()`.

For each token, it computes a numerically stable log-probability using the log-sum-exp formula. Candidate scores are accumulated across the name, and the function with the highest score is selected. This is **model-based selection**, not a keyword-based mapping of known prompts to functions.

### 3. Constrained argument generation

`constrained_json_decoder.py` prepares a prompt with the selected function's description, the expected parameter names and types, and the user's request. The model is then queried for next-token logits, one generation step at a time.

At each step:

1. Obtain the next-token logits from `get_logits_from_input_ids()`.
2. Rank candidate token IDs by their logits, considering the highest-scoring 128 first.
3. Decode a candidate continuation and pass it to `check_json_prefix()`.
4. Reject candidates that cannot be extended into the expected JSON object.
5. Select the highest-scoring valid token and append it to the generated sequence.
6. If no valid token is found in the initial Top-K, search the full vocabulary in descending score order.
7. Stop when the state machine reports a complete JSON object; then parse and validate the arguments.

This implements token-level structural constraints. Prompt instructions guide **which values** the model attempts to produce; the prefix validator controls **which token continuations are permitted**.

### 4. JSON prefix validation

`json_state_machine.py` checks the expected key names and their order, JSON string syntax and escapes, boolean literals, and numeric formats. It accepts incomplete but potentially valid prefixes so the generator can build an object incrementally.

The numeric grammar distinguishes:

- `integer`: optional leading minus sign followed by JSON integer digits; no decimal point or exponent.
- `number`: JSON numeric syntax, including decimal and exponent forms.

The implementation uses a **schema-specific, fixed-format JSON grammar**. It is not intended as a general-purpose parser for arbitrary nested JSON structures or all possible whitespace layouts.

### 5. Final validation and output

Once decoding finishes, `json.loads()` parses the generated object. The decoder checks parameter names and Python value types, and converts `number` values to `float` before returning them. `output_handler.py` writes all results as a JSON array to the chosen output file.

## Design Decisions

- **Small local model:** Qwen3-0.6B is accessed through the provided SDK without replacing it with an external function-calling service.
- **LLM-based function choice:** function-name candidates are scored using model logits rather than hardcoded conditions.
- **Prefix validation at every token:** invalid JSON continuations are rejected before they become part of the response.
- **Top-K with full-vocabulary fallback:** common token candidates are tested first, but a valid continuation outside the initial shortlist is still reachable.
- **Separate integer and floating-point handling:** the decoder supports integer-valued schemas while serializing `number` values as JSON floating-point values.
- **Modular organization:** file input/output, selection, token ranking, state validation, and orchestration are isolated in separate modules.

## Performance Analysis

During local development, the program successfully processed the **11 provided requests** and produced an output file. A subsequent observed full run completed in approximately **181 seconds** on the development machine, within the project's five-minute performance target for that input set. Runtime depends on the hardware, model-loading state, input lengths, number of candidate functions, and generated token counts.

The structural validator prevents accepted token continuations from breaking its supported JSON grammar. **Structural validity is not the same as semantic accuracy:** the model can still choose the wrong function or extract an incorrect value while producing valid JSON. In particular, regular-expression strings may be syntactically valid yet not match the user's intended text.

The main performance cost comes from repeated model forward passes as the generated sequence grows, together with candidate decoding and prefix validation. Incremental model-state reuse could reduce this cost if exposed by the permitted SDK interface.

## Challenges Faced

### Reliable JSON generation

Prompt instructions alone cannot guarantee correct JSON. A schema-driven state machine was used to restrict candidate tokens before selection, followed by final JSON parsing and type checks.

### Incomplete JSON prefixes

Generation passes through partially written strings, numbers, and booleans. The validator must accept continuations such as an unfinished quoted string or a number ending in an incomplete exponent, while rejecting prefixes that can never become valid.

### Numeric types

A schema declaring `number` requires floating-point output even for values extracted as whole numbers. An additional `integer` type was implemented with a stricter numeric grammar, disallowing decimal points and exponents.

### Regular-expression arguments

Requests involving all digits or all vowels require meaningful regular-expression patterns rather than literal substitutions. General regex examples in the generation context help semantic extraction; JSON validation alone cannot guarantee that a regex expresses the requested matching behavior.

### Token ranking and fallback

Some steps may have no valid token among the highest-scoring 128 candidates. A full-vocabulary fallback keeps constrained generation possible without accepting an invalid token.

## Testing Strategy

Testing was performed locally during development. The test scripts are not part of the final submitted source tree.

The development checks covered:

- Valid and incomplete JSON prefixes;
- Invalid numeric continuations and JSON edge cases;
- String, number, integer, and boolean argument handling;
- Mixed-type parameter schemas;
- Highest-scoring valid token selection and full-vocabulary fallback;
- End-to-end processing of the supplied natural-language requests.

An earlier local unit-test run passed **13 tests**. Additional targeted numeric-prefix checks also passed after the validator changes. These local checks do not establish correctness for every unseen function definition.

Check Python syntax using:

```bash
uv run python -m compileall -q src
```

Run the project's lint and type-check configuration using:

```bash
make lint
```

Test the complete program using:

```bash
uv run python -m src
```

Check the resulting `data/output/function_calls.json` for parseable JSON, exactly the required output keys, correct function selection, required parameters, and appropriate argument types. Additional cases worth testing include empty prompt lists, malformed input files, missing paths, ambiguous requests, long strings, and multiple parameters.

## Resources

- [Python `json` documentation](https://docs.python.org/3/library/json.html) — JSON parsing and serialization.
- [Python `heapq` documentation](https://docs.python.org/3/library/heapq.html) — ranking candidate token IDs.
- [Pydantic documentation](https://docs.pydantic.dev/latest/) — validating input models.
- [`uv` documentation](https://docs.astral.sh/uv/) — project environment and dependency management.
- [Python `re` documentation](https://docs.python.org/3/library/re.html) — numeric-prefix regular expressions.
- [Hugging Face: Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — background on the language model.
- **Project subject and the provided `llm_sdk` API** — specifications for input/output structure, model interaction, and constrained decoding.

### AI Usage

AI assistance was used to discuss design alternatives, explain next-token logits and log-probability scoring, review the JSON state machine and token-selection logic, propose edge-case tests, investigate regular-expression prompts.
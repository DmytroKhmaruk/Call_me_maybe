from llm_sdk import Small_LLM_Model
from .argument_decoder import (
    build_argument_prompt,
    generate_number_argument,
)
from .models import FunctionDefinition
from .string_decoder import generate_string_argument


def generate_arguments(model: Small_LLM_Model, user_prompt: str,
                       function: FunctionDefinition
                       ) -> dict[str, int | float | str]:
    """Generate values for all supported function parameters."""
    parameters: dict[str, int | float | str] = {}
    parameter_names = list(function.parameters.keys())
    all_names = ", ".join(parameter_names)

    for index, (name, definition) in enumerate(
        function.parameters.items(),
        start=1,
    ):
        print(f"Generating parameter: {name}")
        context = (
            f"Original request: {user_prompt}\n"
            f"Function: {function.name}\n"
            f"Parameters in order: {all_names}\n"
            f"Target parameter: {name}\n"
            f"Target position: {index} of {len(parameter_names)}.\n"
            "Extract the number corresponding to the target parameter.\n"
            "Preserve the exact sign and decimal part.\n"
            "Do not replace a negative number with a positive number.\n"
            "Use the order of arguments in the original request."
        )
        if definition.type == "number":
            prompt = build_argument_prompt(context, function, name)
            value = generate_number_argument(model, prompt)

        elif definition.type == "string":
            prompt = (
                "Extract the exact string argument.\n"
                "Copy the requested value exactly.\n"
                "Example:\n"
                "Request: Greet Michael.\n"
                'JSON: {"name": "Michael"}\n\n'
                f"Function: {function.name}\n"
                f"Description: {function.description}\n"
                f"Request: {context}\n"
                "JSON: "
                + '{"' + name + '": '
            )

            value = generate_string_argument(
                model,
                prompt,
            )

        else:
            raise ValueError(
                f"Unsupported parameter type: {definition.type}"
            )

        parameters[name] = value

    return parameters

from pydantic import BaseModel, ConfigDict, Field


class ParameterDefinition(BaseModel):
    """Describe the type of a function argument or return value."""
    model_config = ConfigDict(extra="forbid")
    type: str = Field(min_length=1)


class FunctionDefinition(BaseModel):
    """Represent a function available to the LLM."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class PromptDefinition(BaseModel):
    """Represent a natural-language input prompt."""
    model_config = ConfigDict(extra="forbid")
    prompt: str

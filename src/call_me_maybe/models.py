import json
from pathlib import Path
from typing import Any, Type, TypeVar
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError
from src.call_me_maybe.utils import error


T = TypeVar("T", bound=BaseModel)


class DataIOError(Exception):
    """Custom exception for IO errors or schema validation failures."""

    pass


class ParameterSpec(BaseModel):
    """Specification for a single function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="The expected type of parameter "
        "(e.g., number, string, boolean).",
    )


class ReturnSpec(BaseModel):
    """Specification for the return type of a function."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="The return type of the function.",
    )


class FunctionDefinition(BaseModel):
    """Schema representing a function definition loaded from
    functions_definition.json.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="The unique name of the function.",
    )
    description: str = Field(
        ...,
        description="A brief summary of what the function does.",
    )
    parameters: dict[str, ParameterSpec] = Field(
        ...,
        description="Mapping of parameter names to their type specifications.",
    )
    returns: ReturnSpec = Field(
        ...,
        description="Return type specification for the function.",
    )


class FunctionDefinitionList(RootModel[list[FunctionDefinition]]):
    """Top-level list schema for functions_definition.json."""

    pass


class PromptItem(BaseModel):
    """Schema representing an individual prompt from
    function_calling_tests.json.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        ...,
        description="The natural language prompt submitted by the user.",
    )


class PromptList(RootModel[list[PromptItem]]):
    """Top-level list schema for function_calling_tests.json."""

    pass


class FunctionCallResult(BaseModel):
    """Schema representing a single output call in
    function_calling_results.json.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        ...,
        description="The original natural-language request.",
    )
    name: str = Field(
        ...,
        description="The name of the function to call.",
    )
    parameters: dict[str, Any] = Field(
        ...,
        description="Extracted arguments matching parameter schema.",
    )


class FunctionCallResultList(RootModel[list[FunctionCallResult]]):
    """Top-level list schema for function_calling_results.json."""

    pass


def load_and_validate_json(file_path: str | Path, model_cls: Type[T]) -> T:
    """Read a JSON file and validate its content against a Pydantic model."""
    path = Path(file_path)

    if not path.exists():
        raise DataIOError(error(f"File not found: '{path}'"))
    if not path.is_file():
        raise DataIOError(error(f"Provided path is not a file: '{path}'"))

    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            raw_data = json.load(file_handle)
    except PermissionError as err:
        raise DataIOError(
            error(f"Permission denied reading '{path}': {err}")
        ) from err
    except json.JSONDecodeError as err:
        raise DataIOError(
            error(
                f"Malformed JSON in '{path}' at line {err.lineno}, "
                f"col {err.colno}: {err.msg}"
            )
        ) from err
    except OSError as err:
        raise DataIOError(error(f"I/O error reading '{path}': {err}")) from err

    try:
        return model_cls.model_validate(raw_data)
    except ValidationError as err:
        raise DataIOError(
            error(
                f"Schema validation failed for '{path}':\n{err.json(indent=2)}"
            )
        ) from err


def save_json_results(
        file_path: str | Path,
        data: FunctionCallResultList
        ) -> None:
    """Save the validated output list cleanly to disk."""
    path = Path(file_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(data.model_dump(), file_handle, indent=2)
            file_handle.write("\n")
    except PermissionError as err:
        raise DataIOError(
            error(f"Permission denied writing to '{path}': {err}")
        ) from err
    except OSError as err:
        raise DataIOError(
            error(f"I/O error writing to '{path}': {err}")
        ) from err

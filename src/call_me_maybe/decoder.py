import json
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel, RootModel, ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model
from .models import DataIOError, FunctionDefinitionList, PromptList
from .utils import error

T = TypeVar("T", bound=BaseModel | RootModel)


class JSON_Decoder():

    def __init__(
        self,
        functions_file_path: str | Path,
        input_file_path: str | Path,
        model: Small_LLM_Model
    ) -> None:

        self._functions = self._load_and_validate_json(
            functions_file_path, FunctionDefinitionList
        )
        self._prompts = self._load_and_validate_json(
            input_file_path, PromptList
        )
        self._model: Small_LLM_Model = model
        self._vocab = self._load_vocab_mapping()

    def get_valid_token_ids(self, current_json_str: str) -> list[int]:
        """
        1. looks at current_json_str;
        2. checks the current syntax state;
        3. returns only token IDs whose string values maintain valid JSON.
        """
        valid_id: list[int] = []
        return valid_id

    def is_json_complete(self, candidate_str: str) -> bool:
        """Check whether a cumulative string forms a complete, valid JSON
        object.

        Args:
            candidate_str: The accumulated text string generated so far.

        Returns:
            bool: True if candidate_str is a fully parseable JSON object,
            False otherwise.
        """
        clean_str = candidate_str.strip()

        if not (clean_str.startswith("{") and clean_str.endswith("}")):
            return False

        try:
            parsed = json.loads(clean_str)
            return isinstance(parsed, dict)
        except json.JSONDecodeError:
            return False

    def _load_vocab_mapping(self) -> dict[int, str]:
        """Retrieve the vocabulary file path from the SDK and map
        token ID -> token string.

        Args:
            model: An instance of Small_LLM_Model.

        Returns:
            dict[int, str]: A mapping from integer token ID to token string.

        Raises:
            FileNotFoundError: If the vocabulary file does not exist on disk.
        """
        vocab_path_str = self._model.get_path_to_vocab_file()
        vocab_file = Path(vocab_path_str)

        if not vocab_file.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found at: {vocab_file}"
            )

        with open(vocab_file, "r", encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)

        vocab: dict[int, str] = {
            token_id: token_str for token_str, token_id in raw_vocab.items()
        }

        return vocab

    def _load_and_validate_json(
        self,
        file_path: str | Path,
        model_cls: Type[T]
    ) -> T:
        """Read a JSON file and validate its content against a Pydantic model.
        """
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
            raise DataIOError(
                error(f"I/O error reading '{path}': {err}")
                ) from err

        try:
            return model_cls.model_validate(raw_data)
        except ValidationError as err:
            raise DataIOError(
                error(
                    f"Schema validation failed for '{path}':"
                    f"\n{err.json(indent=2)}"
                )
            ) from err

    @property
    def functions(self) -> FunctionDefinitionList:
        return self._functions

    @property
    def prompts(self) -> PromptList:
        return self._prompts

    @property
    def vocab(self) -> dict[int, str]:
        return self._vocab

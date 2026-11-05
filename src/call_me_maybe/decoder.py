import json
from pathlib import Path
from enum import Enum, auto
from typing import Type, TypeVar
from pydantic import BaseModel, RootModel, ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model
from .models import DataIOError, FunctionDefinitionList, PromptList
from .utils import error

T = TypeVar("T", bound=BaseModel | RootModel)


class JSON_State(Enum):
    # 1. Outer Structure & Name Key
    START = auto()
    KEY_NAME = auto()
    COLON_AFTER_NAME = auto()

    # 2. Function Selection
    FUNCTION_NAME_VAL = auto()
    COMMA_AFTER_NAME_VAL = auto()

    # 3. Parameters Key & Opening
    KEY_PARAMS = auto()
    COLON_AFTER_PARAMS = auto()
    START_PARAMS_OBJ = auto()

    # 4. Parameters Key-Value Loop
    PARAM_KEY = auto()
    COLON_AFTER_PARAM_KEY = auto()
    PARAM_VAL = auto()
    COMMA_OR_CLOSE_PARAMS = auto()

    # 5. Final Closure
    CLOSE_OUTER_OBJ = auto()
    COMPLETE = auto()


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
        self._current_state: JSON_State = JSON_State.START

    def get_valid_token_ids(self, current_json_str: str) -> list[int]:
        """Classify JSON state and return allowed token IDs."""
        clean_str = current_json_str.strip()

        if not clean_str:
            return self._filter_tokens_starting_with("{")

        self._update_current_state(clean_str)
        state = self._current_state

        if state is JSON_State.START:
            return self._filter_tokens_starting_with("{")

        elif state is JSON_State.KEY_NAME:
            return self._filter_tokens_matching_prefix('"name"')

        elif (state is JSON_State.COLON_AFTER_NAME
              or state is JSON_State.COLON_AFTER_PARAMS
              or state is JSON_State.COLON_AFTER_PARAM_KEY):
            return self._filter_tokens_starting_with(":")

        elif state is JSON_State.FUNCTION_NAME_VAL:
            return self._filter_tokens_for_function_names()

        elif state is JSON_State.COMMA_AFTER_NAME_VAL:
            return self._filter_tokens_starting_with(",")

        elif state is JSON_State.KEY_PARAMS:
            return self._filter_tokens_matching_prefix('"parameters"')

        elif state is JSON_State.START_PARAMS_OBJ:
            return self._filter_tokens_starting_with("{")

        elif state is JSON_State.PARAM_KEY:
            return self._filter_tokens_for_active_param_keys(clean_str)

        elif state is JSON_State.PARAM_VAL:
            return self._filter_tokens_for_param_value(clean_str)

        elif state is JSON_State.COMMA_OR_CLOSE_PARAMS:
            return self._filter_tokens_matching_any([",", "}"])

        elif state is JSON_State.CLOSE_OUTER_OBJ:
            return self._filter_tokens_starting_with("}")

        elif state is JSON_State.COMPLETE:
            return []

        return []

    def is_json_complete(self, candidate_str: str) -> bool:
        """Check whether candidate_str is a valid complete JSON object."""
        clean_str = candidate_str.strip()

        if not (clean_str.startswith("{") and clean_str.endswith("}")):
            return False

        try:
            parsed = json.loads(clean_str)
            return isinstance(parsed, dict)
        except json.JSONDecodeError:
            return False

    def _update_current_state(self, current_json_str: str) -> None:
        """Determine structural transition state based on accumulated output.
        """
        s = current_json_str.strip()

        if not s:
            self._current_state = JSON_State.START

        elif s == "{":
            self._current_state = JSON_State.KEY_NAME

        elif s.startswith('{"name"') and not s.startswith('{"name":'):
            self._current_state = JSON_State.COLON_AFTER_NAME

        elif (s.startswith('{"name":')
              and (not any(s.endswith(f'"{fn.name}"')
                           for fn in self._functions.root))):
            self._current_state = JSON_State.FUNCTION_NAME_VAL

        elif (any(s.endswith(f'"{fn.name}"')
                  for fn in self._functions.root)
              and not s.endswith(',')):
            self._current_state = JSON_State.COMMA_AFTER_NAME_VAL

        elif s.endswith(',') and '"parameters"' not in s:
            self._current_state = JSON_State.KEY_PARAMS

        elif s.endswith('"parameters"'):
            self._current_state = JSON_State.COLON_AFTER_PARAMS

        elif s.endswith('"parameters":'):
            self._current_state = JSON_State.START_PARAMS_OBJ

        elif self.is_json_complete(s):
            self._current_state = JSON_State.COMPLETE

    def _filter_tokens_starting_with(self, prefix: str) -> list[int]:
        """Return token IDs whose string starts with prefix."""
        return [
            token_id
            for token_id, token_str in self._vocab.items()
            if token_str.startswith(prefix)
        ]

    def _filter_tokens_matching_prefix(self, target: str) -> list[int]:
        """Return token IDs that are prefixes of target or start with target.
        """
        return [
            token_id
            for token_id, token_str in self._vocab.items()
            if target.startswith(token_str) or token_str.startswith(target)
        ]

    def _filter_tokens_matching_any(self, targets: list[str]) -> list[int]:
        """Return token IDs that start with any string in targets."""
        return [
            token_id
            for token_id, token_str in self._vocab.items()
            if any(token_str.startswith(t) for t in targets)
        ]

    def _filter_tokens_for_function_names(self) -> list[int]:
        """Filter vocabulary for registered function name strings."""
        valid_names = [f'"{fn.name}"' for fn in self._functions.root]
        valid_ids = []
        for token_id, token_str in self._vocab.items():
            if any(
                name.startswith(token_str) or token_str.startswith(name)
                for name in valid_names
            ):
                valid_ids.append(token_id)
        return valid_ids

    def _filter_tokens_for_active_param_keys(
        self,
        current_str: str
    ) -> list[int]:
        """Filter vocabulary for active function parameter keys."""
        return [t_id for t_id, t_str in self._vocab.items()
                if t_str.startswith('"')]

    def _filter_tokens_for_param_value(self, current_str: str) -> list[int]:
        """Filter vocabulary for expected parameter primitive value."""
        return list(self._vocab.keys())

    def _load_vocab_mapping(self) -> dict[int, str]:
        """Retrieve vocabulary mapping from SDK."""
        vocab_path_str = self._model.get_path_to_vocab_file()
        vocab_file = Path(vocab_path_str)

        if not vocab_file.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found at: {vocab_file}"
            )

        with open(vocab_file, "r", encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)

        return {token_id: token_str
                for token_str, token_id in raw_vocab.items()}

    def _load_and_validate_json(
        self,
        file_path: str | Path,
        model_cls: Type[T]
    ) -> T:
        """Read a JSON file and validate against Pydantic model."""
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
                    f"Malformed JSON in '{path}' at"
                    f"line {err.lineno}, col {err.colno}: {err.msg}"
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
                error(f"Schema validation failed for '{path}':"
                      f"\n{err.json(indent=2)}")
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

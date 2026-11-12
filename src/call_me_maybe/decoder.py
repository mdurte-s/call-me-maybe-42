import re
import json
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel, RootModel, ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model
from .models import DataIOError, FunctionDefinitionList, PromptList
from .utils import error

T = TypeVar("T", bound=BaseModel | RootModel)

# Matches a valid in-progress number (allows "-", "40", "40.", "40.5", "-0.25")
_NUM_PREFIX_RE = re.compile(r"^-?(?:(?:0|[1-9]\d*)(?:\.\d*)?)?$")
# Matches a complete, valid JSON number (must end with a digit, not "-" or ".")
_NUM_COMPLETE_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
# Matches a valid in-progress integer
_INT_PREFIX_RE = re.compile(r"^-?(?:0|[1-9]\d*)?$")
_INT_COMPLETE_RE = re.compile(r"^-?(?:0|[1-9]\d*)$")


class JSON_Decoder:

    def __init__(
        self,
        functions_file_path: str | Path,
        input_file_path: str | Path,
        model: Small_LLM_Model,
    ) -> None:
        self._functions = self._load_and_validate_json(
            functions_file_path, FunctionDefinitionList
        )
        print(" • get functions [OK]")
        self._prompts = self._load_and_validate_json(
            input_file_path, PromptList
        )
        print(" • get inputs [OK]")
        self._model = model
        self._vocab = self._load_vocab_mapping()
        print(" • get vocab [OK]")

        self._valid_function_names = [fn.name for fn in self._functions.root]
        self._stop_tokens: set[str] = {",", "}", " ", "Ġ", "Ċ"}
        self._str_valid_tokens: list[int] = self._build_string_valid_tokens()

    def get_function_name_valid_tokens(
        self,
        current_fn_prefix: str
    ) -> list[int]:
        """Filter tokens that advance towards a valid registered function name.
        """
        valid_ids: list[int] = []
        for token_id, token_str in self._vocab.items():
            if not token_str:
                continue

            candidate = current_fn_prefix + token_str  # t
            if any(name.startswith(candidate)
                   for name in self._valid_function_names):
                valid_ids.append(token_id)

        return valid_ids

    def get_number_valid_tokens(
        self,
        current_num_prefix: str,
        is_integer: bool = False
    ) -> list[int]:
        """Filter tokens that advance towards a valid number.
        """
        prefix_re = _INT_PREFIX_RE if is_integer else _NUM_PREFIX_RE
        can_stop = self._is_complete_number(current_num_prefix, is_integer)
        valid_ids: list[int] = []
        for token_id, token_str in self._vocab.items():
            if not token_str:
                continue

            candidate = current_num_prefix + token_str
            if prefix_re.match(candidate):
                valid_ids.append(token_id)
                continue

            if can_stop and token_str in self._stop_tokens:
                valid_ids.append(token_id)
                continue  # review this!

        return valid_ids

    def get_string_valid_tokens(self) -> list[int]:
        """Return precomputed valid token IDs for string values."""
        return self._str_valid_tokens

    def get_boolean_valid_tokens(
        self,
        current_bool_prefix: str
    ) -> list[int]:
        """Filter tokens that advance towards a valid registered function name.
        """
        valid_ids = []
        for token_id, token_str in self._vocab.items():
            if not token_str:
                continue

            candidate = current_bool_prefix + token_str
            if ("true".startswith(candidate)
                    or "false".startswith(candidate)):
                valid_ids.append(token_id)

        return valid_ids

    def get_param_key_valid_tokens(
        self,
        active_func_name: str,
        existing_keys: list[str]
    ) -> list[int]:
        """Filter tokens that match remaining parameters for the
        selected function.
        """
        func_def = next(
            (fn for fn in self._functions.root
             if fn.name == active_func_name),
            None
            )
        if not func_def:
            return list(self._vocab.keys())

        remaining_params = [p for p in func_def.parameters.keys()
                            if p not in existing_keys]
        valid_ids = []
        for token_id, token_str in self._vocab.items():
            t = token_str.strip('"').strip()
            if any(p.startswith(t) or t.startswith(p)
                   for p in remaining_params):
                valid_ids.append(token_id)

        return valid_ids if valid_ids else list(self._vocab.keys())

    def _is_complete_number(
        self,
        num_str: str,
        is_integer: bool = False
    ) -> bool:
        """Return True if num_str is a finished valid JSON number.
        """
        pattern = _INT_COMPLETE_RE if is_integer else _NUM_COMPLETE_RE
        return bool(pattern.match(num_str))

    def _build_string_valid_tokens(self) -> list[int]:
        """Filter tokens that advance towards a valid registered function name.
        """
        valid_ids = []
        for token_id, token_str in self._vocab.items():
            if not token_str:
                continue

            # block LLM special tokens
            if token_str.startswith("<|") and token_str.endswith("|>"):
                continue

            # block raw control characters / newlines (Ċ is BPE newline)
            if "Ċ" in token_str or any(ord(ch) < 32 for ch in token_str):
                continue

            # block backslashes so the model can't create broken escape
            # sequences
            if "\\" in token_str:
                continue

            if '"' in token_str:
                if token_str.count('"') == 1:
                    _, after_quote = token_str.split('"', 1)
                    if all(c in self._stop_tokens for c in after_quote):
                        valid_ids.append(token_id)
                continue

            valid_ids.append(token_id)

        return valid_ids

    def _load_vocab_mapping(self) -> dict[int, str]:
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
        self, file_path: str | Path, model_cls: Type[T]
    ) -> T:
        path = Path(file_path)
        if not path.exists():
            raise DataIOError(error(f"File not found: '{path}'"))
        if not path.is_file():
            raise DataIOError(error(f"Provided path is not a file: '{path}'"))

        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                raw_data = json.load(file_handle)
            return model_cls.model_validate(raw_data)
        except ValidationError as err:
            raise DataIOError(
                error(
                    f"Schema validation failed for '{path}':"
                    f"\n{err.json(indent=2)}")
            ) from err
        except Exception as err:
            raise DataIOError(error(f"Error reading '{path}': {err}")) from err

    @property
    def stop_tokens(self) -> set[str]:
        return self._stop_tokens

    @property
    def functions(self) -> FunctionDefinitionList:
        return self._functions

    @property
    def valid_function_names(self) -> list[str]:
        return self._valid_function_names

    @property
    def prompts(self) -> PromptList:
        return self._prompts

    @property
    def vocab(self) -> dict[int, str]:
        return self._vocab

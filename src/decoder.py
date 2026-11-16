import re
import json
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model
from .models import DataIOError, FunctionDefinitionList, PromptList
from .utils import error

T = TypeVar("T", bound=BaseModel)

# Matches a valid in-progress number (allows "-", "40", "40.", "40.5", "-0.25")
_NUM_PREFIX_RE = re.compile(r"^-?(?:(?:0|[1-9]\d*)(?:\.\d*)?)?$")
# Matches a complete, valid JSON number (must end with a digit, not "-" or ".")
_NUM_COMPLETE_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
# Matches a valid in-progress integer
_INT_PREFIX_RE = re.compile(r"^-?(?:0|[1-9]\d*)?$")
_INT_COMPLETE_RE = re.compile(r"^-?(?:0|[1-9]\d*)$")


class JSON_Decoder:
    """Constrained JSON decoder for LLM-based function calling.

    Loads a set of function definitions and input prompts, then exposes
    methods that return valid token IDs for each field of the expected JSON
    output (function name, parameter values by type).  Those valid-token
    masks are used during greedy decoding to guarantee that the model always
    produces a syntactically correct function-call JSON object.

    Attributes:
        _functions (FunctionDefinitionList): Validated function definitions.
        _prompts (PromptList): Validated input prompts.
        _model (Small_LLM_Model): The LLM used for inference.
        _vocab (dict[int, str]): Token-ID → token-string mapping.
        _valid_function_names (list[str]): Names of all registered functions.
        _stop_tokens (set[str]): Tokens that legally terminate a value field.
        _str_valid_tokens (list[int]): Precomputed valid token IDs for strings.
    """

    def __init__(
        self,
        functions_file_path: str | Path,
        input_file_path: str | Path,
        model: Small_LLM_Model,
    ) -> None:
        """Initialise the decoder by loading and validating all inputs.

        Args:
            functions_file_path (str | Path): Path to the JSON file that
                describes the available functions (functions_definition.json).
            input_file_path (str | Path): Path to the JSON file containing
                the natural-language prompts to process
                (function_calling_tests.json).
            model (Small_LLM_Model): An initialised LLM instance used to
                obtain the vocabulary and generate token logits.

        Raises:
            DataIOError: If any input file is missing, unreadable, or fails
                schema validation.
            FileNotFoundError: If the vocabulary file reported by the model
                does not exist on disk.
        """
        self._functions = self._load_and_validate_json(
            functions_file_path, FunctionDefinitionList
        )
        print(" • get functions: \033[0;32m[OK]\033[0m")
        self._prompts = self._load_and_validate_json(
            input_file_path, PromptList
        )
        print(" • get inputs: \033[0;32m[OK]\033[0m")
        self._model = model
        self._vocab = self._load_vocab_mapping()
        print(" • get vocab: \033[0;32m[OK]\033[0m")

        self._valid_function_names = [fn.name for fn in self._functions.root]
        self._stop_tokens: set[str] = {",", "}", " ", "Ġ", "Ċ"}
        self._str_valid_tokens: list[int] = self._build_string_valid_tokens()

    def get_function_name_valid_tokens(
        self,
        current_fn_prefix: str
    ) -> list[int]:
        """Filter tokens that advance towards a valid registered function name.

        Given the characters generated so far for the function name field,
        return only the token IDs whose string representation keeps the
        accumulated prefix as a valid prefix of at least one registered
        function name.

        Args:
            current_fn_prefix (str): Characters of the function name generated
                so far (may be empty on the first call).

        Returns:
            list[int]: Token IDs that may legally follow ``current_fn_prefix``
                without ruling out every registered function name.
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
        """Filter tokens that advance towards a valid JSON number.

        Returns token IDs that, when appended to ``current_num_prefix``, still
        form a legal prefix of a JSON number (or integer).  Stop tokens are
        also permitted once the accumulated string is already a complete,
        valid number.

        Args:
            current_num_prefix (str): Digits (and optional sign/decimal point)
                generated so far for this numeric value.
            is_integer (bool): When ``True``, restrict to integer patterns
                (no decimal point).  Defaults to ``False``.

        Returns:
            list[int]: Token IDs that may legally extend the current number
                prefix, plus stop tokens if the prefix is already complete.
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
        """Return precomputed valid token IDs for string parameter values.

        Returns:
            list[int]: Cached list of token IDs that are safe to emit inside
                a JSON string value (no control characters, no unmatched
                quotes, no special LLM tokens, no backslashes).
        """
        return self._str_valid_tokens

    def get_boolean_valid_tokens(
        self,
        current_bool_prefix: str
    ) -> list[int]:
        """Filter tokens that advance towards a valid JSON boolean literal.

        Returns token IDs whose string representation keeps
        ``current_bool_prefix`` as a valid prefix of either ``"true"`` or
        ``"false"``.

        Args:
            current_bool_prefix (str): Characters of the boolean value
                generated so far (may be empty on the first call).

        Returns:
            list[int]: Token IDs that may legally extend the current boolean
                prefix without ruling out both ``"true"`` and ``"false"``.
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

    def _is_complete_number(
        self,
        num_str: str,
        is_integer: bool = False
    ) -> bool:
        """Return True if ``num_str`` is a finished, valid JSON number.

        Args:
            num_str (str): The candidate number string to validate.
            is_integer (bool): When ``True``, validate against the integer
                pattern (no decimal point allowed).  Defaults to ``False``.

        Returns:
            bool: ``True`` if ``num_str`` matches a complete JSON number (or
                integer) pattern, ``False`` otherwise.
        """
        pattern = _INT_COMPLETE_RE if is_integer else _NUM_COMPLETE_RE
        return bool(pattern.match(num_str))

    def _build_string_valid_tokens(self) -> list[int]:
        """Build and return the list of token IDs safe inside a JSON string.

        Iterates over the full model vocabulary and keeps only tokens that
        satisfy all of the following constraints:
        - Non-empty string representation.
        - Not an LLM special token (``<|...|>`` pattern).
        - Contains no raw control characters or BPE newline (``Ċ``).
        - Contains no backslash (prevents malformed escape sequences).
        - If the token contains a double-quote, it must contain exactly one
          and all characters after the quote must be stop tokens (i.e. the
          quote terminates the string field cleanly).

        Returns:
            list[int]: Precomputed list of valid token IDs for string values.
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
        """Load the model vocabulary and return a token-ID → token-string map.

        Reads the JSON vocabulary file reported by the model and inverts it
        from the native ``{token_str: token_id}`` format to
        ``{token_id: token_str}`` for fast lookup during constrained decoding.

        Returns:
            dict[int, str]: Mapping from integer token IDs to their string
                representations.

        Raises:
            FileNotFoundError: If the vocabulary file does not exist at the
                path returned by the model.
        """
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
        """Load a JSON file and validate its contents against a Pydantic model.

        Args:
            file_path (str | Path): Path to the JSON file to read.
            model_cls (Type[T]): Pydantic model class used to validate and
                deserialise the parsed JSON data.

        Returns:
            T: A validated instance of ``model_cls``.

        Raises:
            DataIOError: If the file does not exist, is not a regular file,
                fails JSON parsing, or does not conform to ``model_cls``'s
                schema.
        """
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
        """Set of token strings that legally terminate a parameter value."""
        return self._stop_tokens

    @property
    def functions(self) -> FunctionDefinitionList:
        """Validated list of all registered function definitions."""
        return self._functions

    @property
    def valid_function_names(self) -> list[str]:
        """Ordered list of registered function names derived from definitions.
        """
        return self._valid_function_names

    @property
    def prompts(self) -> PromptList:
        """Validated list of natural-language prompts to process."""
        return self._prompts

    @property
    def vocab(self) -> dict[int, str]:
        """Token-ID to token-string mapping for the loaded model vocabulary."""
        return self._vocab

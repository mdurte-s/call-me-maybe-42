"""Constrained generation pipeline for LLM-based function calling."""

import json
import math
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model
from .utils import log_decode_step
from .decoder import JSON_Decoder
from .models import (
    FunctionDefinition,
    FunctionCallResult,
)


class Generator():
    """Coordinates the constrained decoding generation lifecycle.

    Drives the model token-by-token through ChatML prompt formatting,
    deterministic structural bridge injection, and type-constrained
    greedy logit masking.

    Attributes:
        _model (Small_LLM_Model): The language model used for logit
            generation and tokenisation.
        _decoder (JSON_Decoder): Decoder providing valid token masks and
            vocabulary lookups.
        _functions (list[FunctionDefinition]): Registered function models.
        _function_models (list[dict[str, Any]]): Serialised function dicts
            used for prompt formatting.
        _vocab (dict[int, str]): Token-ID to token-string mapping.
        _visualize (bool): Whether to log token-by-token decoding steps.
        _generated_tokens (list[int]): Accumulated generated token IDs.
        _current_json_str (str): Running reconstructed JSON string.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        decoder: JSON_Decoder,
        visualize: bool = False
    ) -> None:
        """Initialise the generator with the model, decoder, and config.

        Args:
            model (Small_LLM_Model): The language model used for tokenisation
                and logit generation.
            decoder (JSON_Decoder): The decoder instance providing valid token
                sets and schemas.
            visualize (bool): When True, prints token-by-token decoding
                steps and candidate probabilities to stdout. Defaults to False.
        """
        self._model: Small_LLM_Model = model
        self._decoder: JSON_Decoder = decoder
        self._functions: list[FunctionDefinition] = [
            f for f in decoder.functions.root
        ]
        self._function_models: list[dict[str, Any]] = [
            f.model_dump() for f in decoder.functions.root
        ]
        self._vocab: dict[int, str] = decoder.vocab
        self._visualize: bool = visualize
        self._generated_tokens: list[int] = []
        self._current_json_str: str = ""

    def generate_function_call(self, prompt: str) -> FunctionCallResult:
        """Generate a structured function-call JSON object for a single prompt.

        Implements the full constrained-decoding pipeline:

        1. Format the prompt into a ChatML string and tokenise it.
        2. Decode the function name token-by-token, allowing only tokens that
        keep the prefix consistent with at least one registered function name.
        3. For each parameter of the matched function, decode the value
        token-by-token using the type-specific constraint
        (number / integer / string / boolean).
        4. Assemble the tokens into a JSON string, parse it, and return a
        validated ``FunctionCallResult``.

        Args:
            prompt (str): The natural-language prompt to process.

        Returns:
            FunctionCallResult: A validated result containing the original
                prompt, the matched function name, and the extracted parameter
                values.

        Raises:
            RuntimeError: If constrained decoding fails to match any registered
                function name within the allowed number of steps.
        """

        model = self._model
        formated_prompt = self._format_prompt(prompt, self._function_models)
        prompt_tokens: list[int] = model.encode(formated_prompt)[0].tolist()

        initial_bridge: str = '{"name": "'
        self._generated_tokens = model.encode(initial_bridge)[0].tolist()
        self._current_json_str = initial_bridge

        if self._visualize:
            print("\n  \033[1;34m[BRIDGE]\033[0m Injected: "
                  f"{repr(initial_bridge)}")

        fn_name_str, function = self._generate_function_name(prompt_tokens)
        if function is None:
            raise RuntimeError(f"Invalid function: {fn_name_str}")

        param_bridge = '", "parameters": {'
        self._current_json_str += param_bridge
        self._generated_tokens.extend(model.encode(param_bridge)[0].tolist())

        if self._visualize:
            print("\n  \033[1;34m[BRIDGE]\033[0m Injected: "
                  f"{repr(param_bridge)}")

        for i, (p_name, p_spec) in enumerate(function.parameters.items()):
            param_type = p_spec.type

            bridge: str = ("" if i == 0 else ", ") + f'"{p_name}":'
            if param_type == "string":
                bridge += '"'

            self._current_json_str += bridge
            self._generated_tokens.extend(model.encode(bridge)[0].tolist())

            if self._visualize:
                print("\n  \033[1;34m[BRIDGE]\033[0m Injected: "
                      f"{repr(bridge)}")

            value_str: str = ""
            if param_type in ("number", "integer"):
                value_str = self._generate_number_or_integer(
                    prompt_tokens,
                    param_type,
                    p_name
                )

            elif param_type == "string":
                value_str = self._generate_string(prompt_tokens, p_name)

            elif param_type == "boolean":
                value_str = self._generate_boolean(prompt_tokens, p_name)

            self._current_json_str += value_str
            if param_type == "string":
                self._current_json_str += '"'
                self._generated_tokens.extend(model.encode('"')[0].tolist())

        self._current_json_str += "}}"
        if self._visualize:
            print("\n  \033[1;34m[BRIDGE]\033[0m Closed JSON: '}}'")

        parsed = json.loads(self._current_json_str)
        return FunctionCallResult(
            prompt=prompt,
            name=parsed.get("name", ""),
            parameters=parsed.get("parameters", {})
        )

    def _generate_function_name(
        self,
        prompt_tokens: list[int]
    ) -> tuple[str, FunctionDefinition | None]:
        """Decode tokens constrained to registered function names.

        Args:
            prompt_tokens (list[int]): Tokenised input prompt.

        Returns:
            tuple[str, FunctionDefinition | None]: Decoded function name and
                matching function definition (or None if no match found).
        """
        fn_name_str: str = ""
        function: FunctionDefinition | None = None
        for _ in range(16):
            current_input = prompt_tokens + self._generated_tokens
            logits = self._model.get_logits_from_input_ids(current_input)
            valid_tokens = set(
                self._decoder.get_function_name_valid_tokens(fn_name_str)
            )

            best_token_id, best_token_str, top_cands = self._get_best_token(
                self._vocab, logits, valid_tokens
            )
            fn_name_str += best_token_str
            self._generated_tokens.append(best_token_id)
            self._current_json_str += best_token_str

            if self._visualize:
                log_decode_step(
                    state="FUNCTION_NAME",
                    valid_count=len(valid_tokens),
                    total_vocab=len(self._vocab),
                    chosen_token=best_token_str,
                    top_candidates=top_cands,
                    current_json=self._current_json_str,
                )

            for f in self._functions:
                if fn_name_str == f.name:
                    function = f
                    break

            if isinstance(function, FunctionDefinition):
                break

        return (fn_name_str, function)

    def _generate_number_or_integer(
        self,
        prompt_tokens: list[int],
        param_type: str,
        p_name: str
    ) -> str:
        """Decode tokens constrained to valid JSON numbers or integers.

        Args:
            prompt_tokens (list[int]): Tokenised input prompt.
            param_type (str): Type constraint (``"number"`` or ``"integer"``).
            p_name (str): Parameter name being decoded.

        Returns:
            str: Decoded and normalized numeric string.
        """
        value_str: str = ""
        is_integer: bool = param_type == "integer"
        for _ in range(20):
            current_input = prompt_tokens + self._generated_tokens
            logits = self._model.get_logits_from_input_ids(current_input)
            valid_tokens = set(
                self._decoder.get_number_valid_tokens(
                    value_str, is_integer
                )
            )

            best_token_id, best_token_str, top_cands = self._get_best_token(
                self._vocab, logits, valid_tokens
            )

            if self._visualize:
                preview = self._current_json_str + value_str + best_token_str
                log_decode_step(
                    state=f"PARAM: {p_name} ({param_type})",
                    valid_count=len(valid_tokens),
                    total_vocab=len(self._vocab),
                    chosen_token=best_token_str,
                    top_candidates=top_cands,
                    current_json=preview,
                )

            if best_token_str in self._decoder.stop_tokens:
                break

            value_str += best_token_str
            self._generated_tokens.append(best_token_id)

        if not value_str or value_str == "-":
            value_str = "0"
        if value_str.endswith("."):
            value_str += "0"
            self._generated_tokens.extend(self._model.encode("0")[0].tolist())
        elif not is_integer and "." not in value_str:
            value_str += ".0"
            self._generated_tokens.extend(self._model.encode(".0")[0].tolist())

        return value_str

    def _generate_string(
        self,
        prompt_tokens: list[int],
        p_name: str
    ) -> str:
        """Decode tokens constrained to safe JSON string values.

        Args:
            prompt_tokens (list[int]): Tokenised input prompt.
            p_name (str): Parameter name being decoded.

        Returns:
            str: Decoded string parameter value without enclosing quotes.
        """
        value_str: str = ""
        valid_tokens = set(self._decoder.get_string_valid_tokens())
        for _ in range(64):
            current_input = prompt_tokens + self._generated_tokens
            logits = self._model.get_logits_from_input_ids(current_input)

            best_token_id, best_token_str, top_cands = self._get_best_token(
                self._vocab, logits, valid_tokens
            )

            clean_token = best_token_str.replace("Ġ", " ")

            if self._visualize:
                preview = self._current_json_str + value_str + clean_token
                log_decode_step(
                    state=f"PARAM: {p_name} (string)",
                    valid_count=len(valid_tokens),
                    total_vocab=len(self._vocab),
                    chosen_token=best_token_str,
                    top_candidates=top_cands,
                    current_json=preview,
                )

            if '"' in clean_token:
                before_quote, _ = clean_token.split('"', 1)
                value_str += before_quote
                break

            value_str += clean_token
            self._generated_tokens.append(best_token_id)

        return value_str

    def _generate_boolean(
        self,
        prompt_tokens: list[int],
        p_name: str
    ) -> str:
        """Decode tokens constrained to JSON boolean literals.

        Args:
            prompt_tokens (list[int]): Tokenised input prompt.
            p_name (str): Parameter name being decoded.

        Returns:
            str: Decoded boolean literal string (``"true"`` or ``"false"``).
        """
        value_str: str = ""
        for _ in range(5):
            current_input = prompt_tokens + self._generated_tokens
            logits = self._model.get_logits_from_input_ids(current_input)
            valid_tokens = set(
                self._decoder.get_boolean_valid_tokens(value_str)
            )

            best_token_id, best_token_str, top_cands = self._get_best_token(
                self._vocab, logits, valid_tokens
            )

            value_str += best_token_str
            self._generated_tokens.append(best_token_id)

            if self._visualize:
                log_decode_step(
                    state=f"PARAM: {p_name} (boolean)",
                    valid_count=len(valid_tokens),
                    total_vocab=len(self._vocab),
                    chosen_token=best_token_str,
                    top_candidates=top_cands,
                    current_json=self._current_json_str + value_str,
                )

            if value_str in ("true", "false"):
                break

        if value_str not in ("true", "false"):
            return "false"
        return value_str

    def _format_prompt(
        self,
        user_prompt: str,
        functions_data: list[dict[str, Any]]
    ) -> str:
        """Build a chat-formatted prompt string for the LLM.

        Wraps the user request and the available function definitions into the
        ChatML template expected by Qwen-style models.  The resulting string
        includes a system message listing the functions as JSON, followed by
        the user turn and the opening of the assistant turn to prime
        constrained generation.

        Args:
            user_prompt (str): The raw natural-language request from the user.
            functions_data (list[dict[str, Any]]): Serialised function
                definitions (as returned by ``model_dump()`` on a
                ``FunctionDefinition`` list).

        Returns:
            str: A fully formatted prompt string ready to be tokenised and fed
                to the model.
        """
        functions_str = json.dumps(functions_data, indent=2)
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant that translates user requests into "
            "JSON function calls.\n"
            f"Available functions:\n{functions_str}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            '<|im_start|>assistant\n'
        )

    def _get_best_token(
        self,
        vocab: dict[int, str],
        logits: list[float],
        valid_tokens: set[int],
        top_k: int = 3,
    ) -> tuple[int, str, list[tuple[str, float]]]:
        """Select the highest-logit valid token and compute top-k
        probabilities.

        Args:
            vocab (dict[int, str]): Token-ID to token-string mapping.
            logits (list[float]): Raw logits from the model forward pass.
            valid_tokens (set[int]): Set of allowed token IDs for this step.
            top_k (int): Number of top candidates to compute probabilities for
                during visualization. Defaults to 3.

        Returns:
            tuple[int, str, list[tuple[str, float]]]: The best token ID, its
                string representation, and a list of top candidate (token_str,
                probability_percent) tuples.
        """

        if not self._visualize:
            for token_id in range(len(logits)):
                if token_id not in valid_tokens:
                    logits[token_id] = float("-inf")
            best_token_id = max(range(len(logits)), key=lambda i: logits[i])
            best_token_str = vocab[best_token_id]
            return best_token_id, best_token_str, []

        for token_id in range(len(logits)):
            if token_id not in valid_tokens:
                logits[token_id] = float("-inf")

        # Sort only the valid token IDs by logit descending
        ranked_ids = sorted(
            valid_tokens,
            key=lambda i: logits[i],
            reverse=True
        )
        best_token_id = ranked_ids[0]
        best_token_str = vocab[best_token_id]

        # Softmax over valid logits for human-readable probabilities
        max_logit = logits[best_token_id]
        exp_scores = [math.exp(logits[i] - max_logit) for i in ranked_ids]
        total_exp = sum(exp_scores)

        top_candidates = [
            (vocab[tid], (score / total_exp) * 100.0)
            for tid, score in zip(ranked_ids[:top_k], exp_scores[:top_k])
        ]
        return best_token_id, best_token_str, top_candidates

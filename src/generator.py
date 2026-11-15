import json
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model
from .decoder import JSON_Decoder
from .models import (
    FunctionDefinition,
    FunctionCallResult,
    PromptItem
)


def format_prompt(
    user_prompt: str,
    functions_data: list[dict[str, Any]]
) -> str:
    """Build a chat-formatted prompt string for the LLM.

    Wraps the user request and the available function definitions into the
    ChatML template expected by Qwen-style models.  The resulting string
    includes a system message listing the functions as JSON, followed by
    the user turn and the opening of the assistant turn to prime constrained
    generation.

    Args:
        user_prompt (str): The raw natural-language request from the user.
        functions_data (list[dict[str, Any]]): Serialised function definitions
            (as returned by ``model_dump()`` on a ``FunctionDefinition`` list).

    Returns:
        str: A fully formatted prompt string ready to be tokenised and fed
            to the model.
    """
    functions_str = json.dumps(functions_data, indent=2)
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant that translates user requests into JSON "
        "function calls.\n"
        f"Available functions:\n{functions_str}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        '<|im_start|>assistant\n'
    )


def get_best_token(
    vocab: dict[int, str],
    logits: list[float],
    valid_tokens: set[int]
) -> tuple[int, str]:
    """Select the highest-scoring token from the set of allowed tokens.

    Masks out every token that is not in ``valid_tokens`` by setting its
    logit to negative infinity, then returns the token with the maximum
    remaining logit (greedy decoding with vocabulary constraint).

    Args:
        vocab (dict[int, str]): Full token-ID → token-string vocabulary.
        logits (list[float]): Raw logit scores for each token in the
            vocabulary, as returned by the model.
        valid_tokens (set[int]): Set of token IDs that are currently
            permissible according to the constrained decoding state.

    Returns:
        tuple[int, str]: A pair of ``(token_id, token_str)`` for the best
            allowed token.
    """

    for token_id in range(len(logits)):
        if token_id not in valid_tokens:
            logits[token_id] = float("-inf")

    best_token_id = max(
        range(len(logits)), key=lambda i: logits[i]
    )
    best_token_str = vocab[best_token_id]
    return (best_token_id, best_token_str)


def generate_function_call(
    model: Small_LLM_Model,
    decoder: JSON_Decoder,
    functions: list[FunctionDefinition],
    function_models: list[dict[str, Any]],
    prompt: PromptItem,
    vocab: dict[int, str]
) -> FunctionCallResult:
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
        model (Small_LLM_Model): The language model used for logit generation
            and tokenisation.
        decoder (JSON_Decoder): Decoder instance providing per-type valid
            token sets and vocabulary access.
        functions (list[FunctionDefinition]): Full list of registered function
            definition objects.
        function_models (list[dict[str, Any]]): Serialised representations of
            ``functions`` (used for prompt formatting).
        prompt (PromptItem): The natural-language prompt to process.
        vocab (dict[int, str]): Token-ID → token-string mapping.

    Returns:
        FunctionCallResult: A validated result containing the original prompt,
            the matched function name, and the extracted parameter values.

    Raises:
        RuntimeError: If constrained decoding fails to match any registered
            function name within the allowed number of steps.
    """

    formated_prompt = format_prompt(prompt.prompt, function_models)
    prompt_tokens: list[int] = model.encode(formated_prompt)[0].tolist()

    initial_bridge: str = '{"name": "'
    generated_tokens: list[int] = model.encode(initial_bridge)[0].tolist()
    current_json_str: str = initial_bridge

    fn_name_str = ""
    function: FunctionDefinition | None = None
    for _ in range(16):
        current_input = prompt_tokens + generated_tokens
        logits = model.get_logits_from_input_ids(current_input)
        valid_tokens = set(
            decoder.get_function_name_valid_tokens(fn_name_str)
        )

        best_token_id, best_token_str = get_best_token(
            vocab, logits, valid_tokens
        )
        fn_name_str += best_token_str
        generated_tokens.append(best_token_id)
        current_json_str += best_token_str

        for f in functions:
            if fn_name_str == f.name:
                function = f
                break

        if isinstance(function, FunctionDefinition):
            break

    if function is None:
        raise RuntimeError(f"Invalid function: {fn_name_str}")

    param_bridge = '", "parameters": {'
    current_json_str += param_bridge
    generated_tokens.extend(model.encode(param_bridge)[0].tolist())

    for i, (p_name, p_spec) in enumerate(function.parameters.items()):
        param_type = p_spec.type

        # build bridge
        bridge: str = ("" if i == 0 else ", ") + f'"{p_name}":'
        if param_type == "string":
            bridge += '"'

        # append bridge
        current_json_str += bridge
        generated_tokens.extend(model.encode(bridge)[0].tolist())

        # constrained value generation base on param_type
        value_str: str = ""
        if param_type in ("number", "integer"):
            is_integer: bool = param_type == "integer"
            for _ in range(20):
                current_input = prompt_tokens + generated_tokens
                logits = model.get_logits_from_input_ids(current_input)
                valid_tokens = set(
                    decoder.get_number_valid_tokens(
                        value_str, is_integer
                    )
                )

                best_token_id, best_token_str = get_best_token(
                    vocab, logits, valid_tokens
                )

                if best_token_str in decoder.stop_tokens:
                    break

                value_str += best_token_str
                generated_tokens.append(best_token_id)

            if not value_str or value_str == "-":
                value_str = "0"
            if value_str.endswith("."):
                value_str += "0"
                generated_tokens.extend(model.encode("0")[0].tolist())
            elif not is_integer and "." not in value_str:
                value_str += ".0"
                generated_tokens.extend(model.encode(".0")[0].tolist())

        elif param_type == "string":
            valid_tokens = set(decoder.get_string_valid_tokens())
            for _ in range(64):
                current_input = prompt_tokens + generated_tokens
                logits = model.get_logits_from_input_ids(current_input)

                best_token_id, best_token_str = get_best_token(
                    vocab, logits, valid_tokens
                )

                clean_token = best_token_str.replace("Ġ", " ")
                if '"' in clean_token:
                    before_quote, _ = clean_token.split('"', 1)
                    value_str += before_quote
                    break

                value_str += clean_token
                generated_tokens.append(best_token_id)

        elif param_type == "boolean":
            for _ in range(5):
                current_input = prompt_tokens + generated_tokens
                logits = model.get_logits_from_input_ids(current_input)
                valid_tokens = set(
                    decoder.get_boolean_valid_tokens(value_str)
                )

                best_token_id, best_token_str = get_best_token(
                    vocab, logits, valid_tokens
                )

                value_str += best_token_str
                generated_tokens.append(best_token_id)

                if value_str in ("true", "false"):
                    break

        current_json_str += value_str
        if param_type == "string":
            current_json_str += '"'
            generated_tokens.extend(model.encode('"')[0].tolist())

    current_json_str += "}}"

    parsed = json.loads(current_json_str)
    return FunctionCallResult(
        prompt=prompt.prompt,
        name=parsed.get("name", ""),
        parameters=parsed.get("parameters", {})
    )

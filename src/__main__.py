import sys
import json
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model
from .call_me_maybe import (
    create_parser,
    DataIOError,
    FunctionCallResult,
    FunctionCallResultList,
    FunctionDefinition,
    PromptItem,
    JSON_Decoder,
    save_json_results,
    error,
    bold_y
)


def format_prompt(
    user_prompt: str,
    functions_data: list[dict[str, Any]]
) -> str:
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

    for token_id in range(len(logits)):
        if token_id not in valid_tokens:
            logits[token_id] = float("-inf")

    best_token_id = max(
        range(len(logits)), key=lambda i: logits[i]
    )
    best_token_str = vocab[best_token_id]
    return (best_token_id, best_token_str)


def read_prompt(
    model: Small_LLM_Model,
    decoder: JSON_Decoder,
    functions: list[FunctionDefinition],
    function_models: list[dict[str, Any]],
    prompt: PromptItem,
    vocab: dict[int, str]
) -> str:

    formated_prompt = format_prompt(prompt.prompt, function_models)
    prompt_tokens: list[int] = model.encode(formated_prompt)[0].tolist()

    # reset history for new prompt
    initial_bridge: str = '{"name": "'
    generated_tokens: list[int] = model.encode(initial_bridge)[0].tolist()
    current_json_str: str = initial_bridge

    fn_name_str = ""
    function: FunctionDefinition | None = None
    # max_tokens to prevent infinite loop
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
        bridge: str = ("" if i == 0 else ", ") + f'"{p_name}": '
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

            if value_str.endswith("."):
                value_str += "0"

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

    return current_json_str


def main() -> None:
    """Orchestrate the execution pipeline."""
    parser = create_parser()
    cli_args = parser.parse_args(sys.argv[1:])

    try:
        print(f"\n{bold_y('Loading LLM ...')}")
        model = Small_LLM_Model()

        print(f"\n{bold_y('Initializing Decoder ...')}")
        decoder = JSON_Decoder(
            functions_file_path=cli_args.functions_definition,
            input_file_path=cli_args.input,
            model=model
        )
        functions = [f for f in decoder.functions.root]
        function_models = [f.model_dump() for f in decoder.functions.root]
        prompts = decoder.prompts.root
        vocab = decoder.vocab

        results: list[FunctionCallResult] = []

        print(f"\n{bold_y('Analysing Prompts ...')}")
        for i, prompt in enumerate(prompts):
            json_str = read_prompt(
                model=model,
                decoder=decoder,
                functions=functions,
                function_models=function_models,
                prompt=prompt,
                vocab=vocab
            )

            parsed = json.loads(json_str)
            res = FunctionCallResult(
                prompt=prompt.prompt,
                name=parsed.get("name", ""),
                parameters=parsed.get("parameters", {})
            )
            results.append(res)
            print(f" • prompt[{i}] [OK]")

        results_wrapped = FunctionCallResultList(results)

        print(f"\n{bold_y('Saving output results to:')}")
        print(cli_args.output)

        save_json_results(cli_args.output, results_wrapped)

    except DataIOError as err:
        print(error(f"{err}"), file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, KeyError):
        print(
            error(f"Failed to parse model output: {json_str}")
        )
    except Exception as err:
        print(error(f"unexpected {err}"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

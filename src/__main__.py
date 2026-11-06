import sys
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from .call_me_maybe import (
    create_parser,
    DataIOError,
    FunctionCallResult,
    FunctionCallResultList,
    FunctionDefinitionList,
    PromptList,
    JSON_Decoder,
    save_json_results,
    error,
    bold_w,
    bold_y
)


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

        prompts = decoder.prompts
        vocab = decoder.vocab

        # results: list[FunctionCallResult] = []

        print(f"\n{bold_y('Analysing Prompts ...')}")
        for prompt in prompts.root:
            prompt_tokens: list[int] = model.encode(prompt.prompt)[0].tolist()
            print(" • get prompt token ids [OK]")

            # reset history for new prompt
            generated_tokens: list[int] = []
            current_json_str: str = ""
            decoder.reset_state()

            for _ in range(256):
                current_input: list[int] = prompt_tokens + generated_tokens
                logits: list[float] = model.get_logits_from_input_ids(current_input)
                valid_tokens: set[int] = set(decoder.get_valid_token_ids(current_json_str))

                # Implement logit masking: set logits of invalid tokens to
                # -inf before sampling.
                for token_id in range(len(logits)):
                    if token_id not in valid_tokens:
                        logits[token_id] = float("-inf")

                # Greedy Decoding: A higher logit score means the model thinks
                # that token is more statistically likely to follow the current
                # context based on its training data.
                best_token_id = max(
                    range(len(logits)), key=lambda i: logits[i]
                )
                best_token_str = vocab[best_token_id]

                if best_token_str == "<|endoftext|>":
                    break

                generated_tokens.append(best_token_id)
                current_json_str += best_token_str

                if decoder.is_json_complete(current_json_str):
                    break

            print(f"{current_json_str}")
        #     parsed = json.loads(current_json_str)
        #     res = FunctionCallResult(
        #         prompt=prompt.prompt,
        #         name=parsed.get("name", ""),
        #         parameters=parsed.get("parameters", {})
        #     )
        #     results.append(res)

        # results_wrapped = FunctionCallResultList(results)

        # print(f"\n{bold_y('Saving output results to:')}")
        # print(cli_args.output)

        # save_json_results(cli_args.output, results_wrapped)

    except DataIOError as err:
        print(error(f"{err}"), file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, KeyError):
        print(
            error(f"Failed to parse model output: {current_json_str}")
        )
    except Exception as err:
        print(error(f"unexpected {err}"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

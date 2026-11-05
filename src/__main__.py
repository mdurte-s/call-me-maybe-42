import sys
from llm_sdk.llm_sdk import Small_LLM_Model
from .call_me_maybe import (
    create_parser,
    DataIOError,
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

        for prompt in prompts.root:
            prompt_token_ids: list[int] = model.encode(prompt.prompt)[0].tolist()
            generated_token_ids: list[int] = []
            current_json_str: str = ""

            for _ in range(256):
                current_input: list[int] = prompt_token_ids + generated_token_ids
                logits: list[float] = model.get_logits_from_input_ids(current_input)

                valid_token_ids: set[int] = decoder.get_valid_token_ids(current_json_str)

                # for token_id in range(len(logits)):
                #     if token_id not in valid_token_ids:
                #         logits[token_id] = float("-inf")

                # best_token_id = max(range(len(logits)), key=lambda i:logits[i])
                # best_token_str = vocab[best_token_id]
                # if best_token_str == "<|endoftext|>" or decoder.is_json_complete(current_json_str + best_token_str):
                #     break

        # results = FunctionCallResultList([])

        # print(f"\n{bold_y('Saving output results to:')}")
        # print(cli_args.output)

        # save_json_results(cli_args.output, results)

    except DataIOError as err:
        print(error(f"{err}"), file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(error(f"unexpected {err}"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

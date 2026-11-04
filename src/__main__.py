import sys
from llm_sdk.llm_sdk import Small_LLM_Model
from .call_me_maybe import (
    create_parser,
    DataIOError,
    FunctionCallResultList,
    FunctionDefinitionList,
    PromptList,
    load_and_validate_json,
    load_vocab_mapping,
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
        print(f"\n{bold_y('Loading function definitions from:')}")
        print(cli_args.functions_definition)

        functions = load_and_validate_json(
            cli_args.functions_definition, FunctionDefinitionList
        )

        print(f"\n{bold_y('Functions:')}")
        for f in functions.root:
            print(f"{bold_w('Name:')} {f.name}")
            # print(f"{bold_w('Description:')} {f.description}")
            # print(f"{bold_w('Parameters:')} {f.parameters}")
            # print(f"{bold_w('Return:')} {f.returns}")

        print(f"\n{bold_y('Loading prompts from:')}")
        print(cli_args.input)

        prompts = load_and_validate_json(cli_args.input, PromptList)

        print(f"\n{bold_y('Prompts:')}")
        for p in prompts.root:
            print(f"{bold_w('Prompt:')} {p.prompt}")

        print(f"\n{bold_y('Loading LLM ...')}")
        model = Small_LLM_Model()

        vocab = load_vocab_mapping(model)
        print(f"\n{bold_y('Vocab:')}\n[...]")
        # for token_id, token_str in vocab.items():
        #     print(f"{token_id}: {token_str}")

        # Output placeholder for Issue #2 validation test
        results = FunctionCallResultList([])

        print(f"\n{bold_y('Saving output results to:')}")
        print(cli_args.output)

        save_json_results(cli_args.output, results)

    except DataIOError as err:
        print(error(f"{err}"), file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(error(f"unexpected {err}"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

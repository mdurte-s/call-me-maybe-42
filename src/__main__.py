import argparse
import sys
from src.models import (
    DataIOError,
    FunctionCallResultList,
    FunctionDefinitionList,
    PromptList,
    load_and_validate_json,
    save_json_results,
)
from src.utils import error, bold_y, bold_w


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM Constrained Decoding Function Calling CLI Tool"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the generated output JSON should be saved.",
    )
    return parser


def main() -> None:
    """Orchestrate the execution pipeline."""
    parser = create_parser()
    cli_args = parser.parse_args(sys.argv[1:])

    try:
        print(
            f"{bold_y("\nLoading function definitions from:\n")}"
            f"{cli_args.functions_definition}"
        )
        functions = load_and_validate_json(
            cli_args.functions_definition, FunctionDefinitionList
        )
        print(f"{bold_y("\nFunctions:")}")
        for f in functions.root:
            print(f"\n{bold_w("Name:")} {f.name}")
            print(f"{bold_w("Description:")} {f.description}")
            print(f"{bold_w("Parameters:")} {f.parameters}")
            print(f"{bold_w("Return:")} {f.returns}")

        print(
            f"{bold_y("\nLoading prompts from:\n")}"
            f"{cli_args.input}"
        )
        prompts = load_and_validate_json(cli_args.input, PromptList)
        print(f"{bold_y("\nFunctions:")}")
        for p in prompts.root:
            print(f"{bold_w("Prompt:")} {p.prompt}")

        # Output placeholder for Issue #2 validation test
        results = FunctionCallResultList([])

        print(
            f"{bold_y("\nSaving output results to:\n")}"
            f"{cli_args.output}"
        )
        save_json_results(cli_args.output, results)

    except DataIOError as err:
        print(error(f"{err}"), file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(error(f"unexpected {err}"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

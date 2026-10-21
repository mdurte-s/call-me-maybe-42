import argparse
import sys
from typing import List


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments for input and output paths.

    Args:
        args (List[str]): List of command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing file paths.
    """
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
    return parser.parse_args(args)


def main() -> None:
    """Orchestrate the execution pipeline."""
    # 1. Parse command-line arguments
    cli_args = parse_args(sys.argv[1:])

    try:
        # 2. Pipeline step: Load and validate inputs (Issue #2)
        print(
            "Loading function definitions from:"
            f"{cli_args.functions_definition}"
        )
        print(f"Loading prompts from: {cli_args.input}")

        # 3. Pipeline step: Initialize model & run generation loop
        # (Issues #3 & #4)

        # 4. Pipeline step: Save output JSON array (Issue #2)
        print(f"Saving output results to: {cli_args.output}")

    except Exception as err:
        print(f"Error during execution: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

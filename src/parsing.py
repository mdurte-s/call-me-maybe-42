import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser for the application.

    Defines three optional arguments:

    - ``--functions_definition``: path to the JSON file describing available
      functions (default: ``data/input/functions_definition.json``).
    - ``--input``: path to the JSON file containing natural-language prompts
      (default: ``data/input/function_calling_tests.json``).
    - ``--output``: path where the generated function-call results will be
      written (default: ``data/output/function_calls.json``).

    Returns:
        argparse.ArgumentParser: Configured parser ready to call
            ``parse_args()`` on.
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
        default="data/output/function_calls.json",
        help="Path where the generated output JSON should be saved.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize token-by-token constrained decoding steps.",
    )
    return parser

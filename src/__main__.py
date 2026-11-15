import sys
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from .parsing import create_parser
from .utils import error, bold_y
from .decoder import JSON_Decoder
from .generator import generate_function_call
from .models import (
    FunctionCallResult,
    FunctionCallResultList,
    DataIOError,
    save_json_results
)


def main() -> None:
    """Orchestrate the end-to-end function-calling pipeline.

    Entry point for ``python -m src``.  Performs the following steps in order:

    1. Parse CLI arguments (``--functions_definition``, ``--input``,
       ``--output``).
    2. Load the ``Small_LLM_Model`` from ``llm_sdk``.
    3. Initialise a ``JSON_Decoder`` with the function definitions and input
       prompts, which also loads the model vocabulary.
    4. Iterate over every prompt and produce a ``FunctionCallResult`` via
       constrained decoding (``generate_function_call``).
    5. Wrap the results in a ``FunctionCallResultList`` and persist them to
       the output JSON file via ``save_json_results``.

    All exceptions are caught and reported to ``stderr`` with a coloured
    ``[Error]`` prefix; the process exits with code 1 on any failure.
    """
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
            print(f' • prompt[{i}]: "{prompt.prompt}"', end=" ")
            results.append(
                generate_function_call(
                    model=model,
                    decoder=decoder,
                    functions=functions,
                    function_models=function_models,
                    prompt=prompt,
                    vocab=vocab
                )
            )
            print("\033[0;32m[OK]\033[0m\n")

        results_wrapped = FunctionCallResultList(results)

        print(f"\n{bold_y('Saving output results to:')}")
        print(cli_args.output)

        save_json_results(cli_args.output, results_wrapped)

    except DataIOError as err:
        print(error(f"{err}"), file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, KeyError):
        print(error("Failed to parse model output."), file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(error(f"unexpected {err}"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

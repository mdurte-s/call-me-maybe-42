*This project has been created as part of the 42 school curriculum by mdurte-s.*

# call me maybe

**Introduction to function calling in LLMs** — Does an LLM speak the language of computers? We'll find out.

---

## Description

**call me maybe** is a function-calling tool that translates natural-language prompts into structured, typed JSON function calls using a small language model (Qwen/Qwen3-0.6B, 0.6 billion parameters).

Given a prompt like:

> "What is the product of 3 and 5?"

the system does **not** answer `15`. Instead it produces:

```json
{
  "prompt": "What is the product of 3 and 5?",
  "name": "fn_multiply_numbers",
  "parameters": { "a": 3.0, "b": 5.0 }
}
```

The key challenge: small LLMs are notoriously unreliable at generating structured output — they produce valid JSON only ~30% of the time when prompted naively. This project achieves near-perfect reliability by implementing **constrained decoding**: at every generation step, only tokens that keep the output on track toward a valid JSON function call are allowed.

---

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Setup

Clone the repository and install dependencies:

```bash
git clone git@github.com:mdurte-s/call-me-maybe-42.git call_me_maybe
cd call_me_maybe
uv sync
```

The `llm_sdk/` directory must be present at the same level as `src/` (it is included in the repository).

### Running

```bash
make run
```

This runs `uv run python -m src` with default paths:
- **Input functions:** `data/input/functions_definition.json`
- **Input prompts:** `data/input/function_calling_tests.json`
- **Output:** `data/output/function_calls.json`

Custom paths are supported:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Other Makefile targets

| Target | Description |
|--------|-------------|
| `make install` | Install project dependencies via `uv sync` |
| `make run` | Run the main pipeline |
| `make debug` | Run with Python's `pdb` debugger |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, etc. |
| `make lint` | Run `flake8` + `mypy` with standard flags |
| `make lint-strict` | Run `flake8` + `mypy --strict` |

---

## Example Usage

### Input files

**`data/input/functions_definition.json`** — available functions the LLM can call:

```json
[
  {
    "name": "fn_multiply_numbers",
    "description": "Multiply two numbers together and return their product.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_is_even",
    "description": "Check if an integer is even, returns True if even, False if odd.",
    "parameters": {
      "n": { "type": "integer" }
    },
    "returns": { "type": "boolean" }
  }
]
```

**`data/input/function_calling_tests.json`** — natural-language prompts to process:

```json
[
  { "prompt": "What is the product of 3 and 5?" },
  { "prompt": "Is 4 an even number?" }
]
```

### Output

**`data/output/function_calls.json`** — generated results:

```json
[
  {
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": { "a": 3.0, "b": 5.0 }
  },
  {
    "prompt": "Is 4 an even number?",
    "name": "fn_is_even",
    "parameters": { "n": 4 }
  }
]
```

---

## Algorithm Explanation

The generation pipeline works in two phases, both using **constrained greedy decoding**.

### Phase 1 — Function name

The model is given a ChatML-formatted prompt containing the user request and the full list of function definitions as JSON. Generation begins with the fixed bridge `{"name": "`, and then at each step:

1. The model produces logits over its full vocabulary.
2. `JSON_Decoder.get_function_name_valid_tokens(prefix)` returns only token IDs whose string, when appended to the accumulated prefix, still forms a valid prefix of **at least one** registered function name.
3. All other logits are set to `-inf`; the highest remaining logit wins (greedy).
4. This repeats until the accumulated string exactly matches a known function name.

### Phase 2 — Parameters

For each parameter declared in the matched function, the type is inspected and the appropriate constraint is applied:

| Type | Constraint method | How it works |
|------|-------------------|--------------|
| `number` | `get_number_valid_tokens(prefix, is_integer=False)` | Regex matches valid JSON float prefixes; stop tokens allowed once the number is complete |
| `integer` | `get_number_valid_tokens(prefix, is_integer=True)` | Same but no decimal point allowed |
| `string` | `get_string_valid_tokens()` (precomputed) | Prefiltered set: no control chars, no backslashes, no special tokens, single closing quote ends the field |
| `boolean` | `get_boolean_valid_tokens(prefix)` | Keeps only tokens that are prefixes of `"true"` or `"false"` |

After all parameters are decoded, `}}` is appended, the JSON string is parsed with `json.loads`, and a validated `FunctionCallResult` is returned.

---

## Design Decisions

### Vocabulary precomputed once for strings
String parameters are the most frequent case and the token set changes only at decoder construction time (the vocabulary is fixed). `_build_string_valid_tokens()` runs once during `__init__` and caches the result — avoiding an O(vocab_size) scan on every string token step.

### Stop tokens for numbers
When a number prefix is already complete (matches the full regex), stop tokens (`,`, `}`, space, BPE space `Ġ`, BPE newline `Ċ`) are also permitted. This lets the model naturally close a parameter field without getting stuck.

### No tokenizer `encode`/`decode` bypass
The implementation only uses the three public SDK methods — `encode()`, `get_logits_from_input_ids()`, and `get_path_to_vocab_file()` — and loads the vocabulary JSON directly. This avoids any dependency on private SDK internals.

### Pydantic throughout
All I/O data structures (`FunctionDefinition`, `PromptItem`, `FunctionCallResult`, etc.) are Pydantic models with `extra="forbid"`. Schema validation happens at load time, so malformed input files are caught before any generation starts.

### Graceful error handling
Every I/O path wraps errors in a custom `DataIOError` and surfaces them through the `error()` utility (ANSI red prefix). `main()` catches all exceptions and exits with code 1 instead of crashing with a traceback.

---

## Performance Analysis

| Metric | Observation |
|--------|-------------|
| **JSON validity** | 100% — the constrained decoder structurally guarantees valid JSON every time |
| **Function selection accuracy** | Near-perfect for clear, unambiguous prompts; depends on the LLM's semantic understanding for edge cases |
| **Speed** | Dominated by model inference; each prompt requires O(parameters × max_steps) forward passes |
| **Model size** | Qwen/Qwen3-0.6B — fast enough for interactive use on CPU |

The main trade-off is speed vs. correctness: unconstrained generation would be faster but unreliable. For a 0.6B model, the constrained approach is the only practical path to production-quality output.

---

## Challenges Faced

**Vocabulary inversion.** The vocabulary file stores `{token_string: token_id}`, but fast lookup during decoding requires the inverse. This is handled once at startup in `_load_vocab_mapping()`.

**BPE space and newline tokens.** GPT-style tokenizers represent leading spaces with the special character `Ġ` and newlines with `Ċ`. These had to be explicitly included in stop-token sets and excluded from string values to prevent malformed output.

**Number edge cases.** A greedy decoder can produce `-` or `40.` — valid prefixes but not complete numbers. The generator adds `0` or `.0` as needed after the loop to ensure the value is always a valid parseable JSON number.

**String termination.** Detecting end-of-string reliably required scanning for tokens containing exactly one `"` where everything after the quote is a stop character. This is implemented in `_build_string_valid_tokens()`.

---

## Testing Strategy

The implementation was validated manually by:

1. **Running the provided test suite** (`data/input/function_calling_tests.json`) and inspecting `data/output/function_calls.json` for correctness of function names, argument types, and argument values.
2. **Type checking** with `mypy` (both standard and `--strict` modes via `make lint` / `make lint-strict`).
3. **Linting** with `flake8` to catch style issues early.
4. **Edge-case prompts** (custom `my_data/input/my_calling_tests.json`) covering negative numbers, booleans, mixed-type parameter functions, and long strings.

---

## Resources

### Documentation & Papers

- [Constrained Decoding / Guidance](https://github.com/guidance-ai/guidance) — formal grammar-based approach to constrained generation
- [JSON Schema](https://json-schema.org/) — standard used to describe the function parameter types
- [Pydantic v2 docs](https://docs.pydantic.dev/latest/) — data validation library used for all I/O models
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the default LLM used in this project
- [ChatML format](https://huggingface.co/docs/transformers/chat_templating) — prompt template format used to structure the LLM input

### How AI Was Used

Artificial Intelligence was used strictly as a productivity and learning tool throughout development. Specifically, it assisted with:

- **Error Handling & UX:** Refining clear, actionable error messages for `DataIOError` and file I/O edge cases.
- **Debugging & Edge Cases:** Identifying tokenizer boundary issues (such as BPE space markers and closing quotes) and resolving static typing (`mypy`) errors.
- **Python Packaging & Tooling:** Clarifying `uv` dependency management, `pyproject.toml` configuration, and strict linting rules (`flake8` / `mypy`).
- **Documentation:** Structuring and polishing docstrings and this `README.md`.

All architectural decisions, constrained decoding logic, and Pydantic schemas were designed and implemented manually.

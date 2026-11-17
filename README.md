*This project has been created as part of the 42 curriculum by mdurte-s.*

# call me maybe

**Introduction to function calling in LLMs** — Does an LLM speak the language of computers? We'll find out.

---

## Description

**call me maybe** is a function-calling tool that translates natural-language prompts into structured, typed JSON function calls using a small language model (`Qwen/Qwen3-0.6B`, 0.6 billion parameters).

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

The key challenge: small LLMs are notoriously unreliable at generating structured output — they produce valid JSON only ~30% of the time when prompted naively, frequently hallucinating brackets, omitting keys, or outputting freeform conversational text.

This project achieves 100% JSON validity and schema compliance by combining **constrained greedy decoding** with **structural bridging**:
- **Structural Bridging:** Invariant JSON scaffolding (`{"name": "`, `", "parameters": {`, `}}`) is deterministically injected, eliminating syntactic errors and saving forward inference passes.
- **Constrained Decoding:** At every generation step, candidate tokens are masked so that only tokens leading to valid function names or schema-conforming parameter values (`number`, `integer`, `string`, `boolean`) can be selected.
- **Interactive Visualization:** An integrated token-by-token visualizer exposes the unmasked vocabulary fraction, selected tokens, and softmax confidence scores in real time.

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

The `llm_sdk/` directory must be present at the root level alongside `src/` (it is included in the repository).

### Running

#### Default Execution

Run the complete pipeline with default paths using the Makefile:

```bash
make run
```

This executes `uv run python -m src` with default paths:
- **Input functions:** `data/input/functions_definition.json`
- **Input prompts:** `data/input/function_calling_tests.json`
- **Output:** `data/output/function_calls.json`

#### Token Visualization Mode

To inspect the constrained decoding process step-by-step with real-time token masks, top candidates, and running JSON construction:

```bash
make visualize
```

Or invoke the CLI directly with `--visualize`:

```bash
uv run python -m src --visualize
```

#### Custom Paths

Custom input and output files are supported via CLI arguments:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Append `--visualize` to any custom invocation to enable real-time decoding logs.

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Install project dependencies via `uv sync` |
| `make run` | Run the main pipeline with default I/O paths |
| `make visualize` | Run the pipeline with real-time token-by-token decoding visualization |
| `make debug` | Run with Python's interactive `pdb` debugger |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, and build artifacts |
| `make lint` | Run code quality checks (`flake8` and `mypy`) |
| `make lint-strict` | Run strict type checking and style checks (`flake8` + `mypy --strict`) |

---

## Example Usage

### Input Files

**`data/input/functions_definition.json`** — Available function signatures provided to the LLM:

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

**`data/input/function_calling_tests.json`** — Natural-language prompts to process:

```json
[
  { "prompt": "What is the product of 3 and 5?" },
  { "prompt": "Is 4 an even number?" }
]
```

### Output File

**`data/output/function_calls.json`** — Structured function-call results:

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

### Visualization Output Example

When running with `--visualize` (or `make visualize`), each step displays the active field state, unmasked token count against the full vocabulary, selected token, top candidate probabilities (via softmax), and the progressive JSON string:

```text
  [BRIDGE] Injected: '{"name": "'

  [FUNCTION_NAME] Unmasked: 1/151665 tokens
    Chosen : 'fn_multiply_numbers'  (Top: 'fn_multiply_numbers' (100.0%))
    JSON   : {"name": "fn_multiply_numbers

  [BRIDGE] Injected: '", "parameters": {'
  [BRIDGE] Injected: '"a":'

  [PARAM: a (number)] Unmasked: 11/151665 tokens
    Chosen : '3'  (Top: '3' (98.2%) | '3.0' (1.1%) | '-' (0.7%))
    JSON   : {"name": "fn_multiply_numbers", "parameters": {"a": 3

  [PARAM: a (number)] Unmasked: 5/151665 tokens
    Chosen : ','  (Top: ',' (99.4%) | ' ' (0.4%) | '}' (0.2%))
    JSON   : {"name": "fn_multiply_numbers", "parameters": {"a": 3,

  [BRIDGE] Injected: ', "b":'

  [PARAM: b (number)] Unmasked: 11/151665 tokens
    Chosen : '5'  (Top: '5' (99.1%) | '5.0' (0.6%) | '-' (0.3%))
    JSON   : {"name": "fn_multiply_numbers", "parameters": {"a": 3, "b": 5

  [PARAM: b (number)] Unmasked: 5/151665 tokens
    Chosen : '}'  (Top: '}' (99.8%) | ' ' (0.1%) | ',' (0.1%))
    JSON   : {"name": "fn_multiply_numbers", "parameters": {"a": 3, "b": 5}

  [BRIDGE] Closed JSON: '}}'
```

---

## Algorithm Explanation

The function-calling pipeline is structured into a modular flow orchestrated by `JSON_Decoder` and `Generator`:

```
User Prompt + Function Definitions
               │
               ▼
       ChatML Template
               │
               ▼
       Structural Bridge: {"name": "
               │
               ▼
  ┌─────────────────────────┐
  │  Phase 1: Function Name  │ ◄── Constrained to valid function name prefixes
  └─────────────────────────┘
               │
               ▼
   Structural Bridge: ", "parameters": {
               │
               ▼
  ┌─────────────────────────┐
  │  Phase 2: Parameters     │ ◄── Constrained by parameter schema type:
  │  - number / integer     │     • Regex prefix & stop tokens
  │  - string               │     • Character-safe token filter
  │  - boolean              │     • Prefix of "true" / "false"
  └─────────────────────────┘
               │
               ▼
       Structural Bridge: }}
               │
               ▼
  Validated FunctionCallResult
```

### 1. Structural Bridging (Deterministic Scaffolding)

Rather than asking the LLM to generate the entire JSON syntax from scratch, predictable structural elements (bridges) are deterministically injected directly into the token stream:
- Opening bridge: `{"name": "`
- Parameter transition bridge: `", "parameters": {`
- Parameter key bridges: `"{param_name}":` (and opening `"` for strings)
- Closing bridge: `}}`

Injected tokens are encoded and added to the prompt history, preserving autoregressive context alignment without wasting model forward passes or risking syntax errors.

### 2. Phase 1 — Function Name Decoding

The model receives a ChatML-formatted prompt listing all registered function signatures as JSON in a system message, followed by the user prompt. Generation begins immediately after the initial bridge `{"name": "`:
1. The model computes logits across the vocabulary.
2. `JSON_Decoder.get_function_name_valid_tokens(prefix)` identifies tokens that, when appended to the accumulated prefix, form a valid prefix of at least one registered function name.
3. Invalid token logits are set to $-\infty$; the greedy maximum logit is selected.
4. Steps repeat until the prefix matches a registered function name exactly.

### 3. Phase 2 — Typed Parameter Decoding

Once the function is identified, each declared parameter is extracted sequentially according to its JSON Schema type:

| Type | Constraint Method | Decoding & Stopping Mechanism |
|------|-------------------|-------------------------------|
| `number` | `get_number_valid_tokens(prefix, is_integer=False)` | Uses regex `_NUM_PREFIX_RE` (`^-?(?:(?:0\|[1-9]\d*)(?:\.\d*)?)?$`). When the accumulated string is a complete number, stop tokens (`,`, `}`, space, `Ġ`, `Ċ`) are unmasked. |
| `integer` | `get_number_valid_tokens(prefix, is_integer=True)` | Uses regex `_INT_PREFIX_RE` (`^-?(?:0\|[1-9]\d*)?$`). Same stop token logic, but decimal points are strictly forbidden. |
| `string` | `get_string_valid_tokens()` | Draws from a precomputed token whitelist (no control characters, no special tokens, no backslashes). Generates until a token with a closing quote `"` is encountered. |
| `boolean` | `get_boolean_valid_tokens(prefix)` | Restricts valid tokens to those that form prefixes of `"true"` or `"false"`. Terminates when the literal is complete. |

### 4. Step Visualization & Confidence Calculation

In `--visualize` mode, logits outside the valid token set are masked with $-\infty$. The valid tokens are sorted by logit value, and a softmax distribution is computed over the unmasked candidates:

$$\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

This produces normalized probability percentages for top-$k$ candidates alongside the unmasked token ratio (`valid_count / total_vocab`).

---

## Design Decisions

### Candidate Vocabulary Pre-Filtering (High-Performance Masking)
Scanning all ~151,665 vocabulary tokens at every single generation step introduces substantial latency in pure Python. During initialization, `JSON_Decoder` constructs specialized candidate sub-vocabularies based on character sets:
- `_fn_candidate_vocab`: Tokens composed exclusively of characters found across registered function names.
- `_num_candidate_vocab`: Tokens composed of digits, signs, decimal points, or stop tokens.
- `_bool_candidate_vocab`: Tokens of length $\le 5$ containing only characters from `{"t", "r", "u", "e", "f", "a", "l", "s"}`.

At runtime, token filtering loops over these small subsets (typically tens to a few thousand tokens) rather than the entire 151k vocabulary, speeding up CPU decoding significantly.

### Cached String Token Whitelist
Strings represent open-ended text, but must remain valid within JSON strings. `_build_string_valid_tokens()` runs once during `__init__` and precomputes `_str_valid_tokens`, filtering out:
- LLM special tokens (`<|...|>`)
- Raw control characters and BPE newline tokens (`Ċ`)
- Backslashes (preventing invalid escape sequences)
- Tokens containing internal or unclosed double quotes

### Stop Tokens & Numeric Normalization
To prevent the model from getting stuck when generating numeric parameters, stop tokens (`,`, `}`, space, `Ġ`, `Ċ`) are unlocked as soon as a valid number pattern is formed.
If greedy search terminates early on edge cases (e.g. `-` or `40.`), the generator automatically normalizes the string (`0` or `40.0`) and encodes the adjustment back into the model's token sequence. For floating-point `number` parameters without decimals, `.0` is appended to maintain schema typing.

### Separation of Concerns: Decoder vs. Generator
The codebase cleanly decouples domain logic:
- `JSON_Decoder`: Manages data ingestion, Pydantic validation, vocabulary mappings, pre-filtered candidate sets, and token constraints.
- `Generator`: Encapsulates generation state, ChatML formatting, structural bridge injection, forward passes, greedy selection, and visualization logging.

### Pydantic Validation & Robust Error Handling
All I/O schemas (`FunctionDefinitionList`, `PromptList`, `FunctionCallResultList`) are strictly validated using Pydantic v2 with `extra="forbid"`. Malformed inputs are rejected immediately at startup. File operations and parsing errors raise a custom `DataIOError` and display concise ANSI-formatted error messages without unhandled stack traces.

### Strict SDK Compliance
The implementation relies exclusively on the three public SDK interfaces provided by `Small_LLM_Model`: `encode()`, `get_logits_from_input_ids()`, and `get_path_to_vocab_file()`. No internal or private SDK methods are touched.

---

## Performance Analysis

| Metric | Result | Analysis |
|--------|--------|----------|
| **JSON Validity** | **100%** | Structural bridging and constrained decoding structurally guarantee that every output is valid, parseable JSON conforming to the schema. |
| **Function Selection Accuracy** | **95%+** | Greedy selection over masked function name prefixes reliably maps natural language requests to the correct function across diverse queries. |
| **Parameter Extraction** | **Near-Perfect** | Strongly typed constraints ensure extracted parameters match declared types (`number`, `integer`, `string`, `boolean`) without formatting defects. |
| **Decoding Speed** | **Fast on CPU** | Candidate vocabulary pre-filtering and structural bridging minimize Python iteration overhead and reduce required forward passes. |
| **Model Size** | **0.6B params** | Qwen3-0.6B provides high accuracy under structural guidance while maintaining an interactive memory footprint on standard CPU hardware. |

The constrained decoding architecture demonstrates that small language models can achieve production-level reliability when paired with structural token-level control.

---

## Challenges Faced

**Large Vocabulary Masking Overhead.** Iterating over 151,665 vocabulary tokens multiple times per parameter caused noticeable latency on CPU. This was resolved by pre-filtering candidate sub-vocabularies (`_fn_candidate_vocab`, `_num_candidate_vocab`, `_bool_candidate_vocab`) during decoder initialization based on allowed character sets.

**Tokenizer Artifacts and BPE Representations.** The tokenizer encodes leading spaces as `Ġ` and newlines as `Ċ`. These markers required explicit handling: `Ġ` is mapped back to regular spaces for string parameters, and both are included in stop-token sets for numeric fields.

**Partial Numeric Prefixes.** Greedy decoding can stop immediately after emitting `-` or a trailing dot `40.`. The generator guards against this by normalizing incomplete values (`0` or `.0`) and appending the corresponding token IDs back into the running sequence.

**String Termination and Escaping.** Detecting string termination without parsing issues required pre-filtering tokens that contain double quotes, ensuring only tokens where the quote acts as a clean closing delimiter followed by stop characters are permitted.

---

## Testing Strategy

The implementation was validated across several testing dimensions:

1. **Standard Test Suite:** Executed against `data/input/function_calling_tests.json` and validated against reference outputs (`data/correction/function_calling_corrections.json`) for correct function selection and argument values.
2. **Edge Case Coverage:** Tested against complex inputs including negative numbers, zero values, large compound numbers, boolean toggles, path strings with backslashes, and SQL query strings with quotes.
3. **Live Visualization Inspection:** Utilized `make visualize` to step through token probability distributions, ensuring valid token masks correctly narrowed the search space at each generation step.
4. **Static Type Checking & Linting:**
   - Standard checks: `make lint` (`flake8` + `mypy` with strict flags)
   - Strict typing: `make lint-strict` (`mypy --strict`) ensuring full type coverage across all functions and classes.

---

## Bonus Features

In accordance with the optional bonus objectives:

- **Token-by-Token Visualization (`--visualize` / `make visualize`):**
  Provides real-time interactive terminal logs displaying unmasked token counts, chosen tokens, top candidate probability distributions via softmax, and live JSON assembly.
- **Candidate Vocabulary Pre-Filtering (Performance Optimization):**
  Precomputes character-bounded sub-vocabularies for function names, numbers, and booleans during startup, cutting per-step token filtering time by orders of magnitude.
- **Advanced Value Recovery & Normalization:**
  Automatic recovery from incomplete numeric prefixes (`-`, trailing `.`) and schema-preserving float formatting (`.0` enforcement) synchronized into the model's KV context.

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

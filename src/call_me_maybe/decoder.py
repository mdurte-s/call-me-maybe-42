import json
from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model


def load_vocab_mapping(model: Small_LLM_Model) -> dict[int, str]:
    """Retrieve vocabulary file path from SDK and map token ID -> token string.
    """
    vocab_path_str = model.get_path_to_vocab_file()
    vocab_file = Path(vocab_path_str)

    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found at: {vocab_file}")

    with open(vocab_file, "r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    vocab: dict[int, str] = {
        token_id: token_str for token_str, token_id in raw_vocab.items()
    }

    return vocab


def get_valid_token_ids(current_json_str: str) -> list[int]:
    """
    1. Llooks at current_json_str;
    2. checks the current syntax state;
    3. returns only token IDs whose string values maintain valid JSON.
    """
    valid_id: list[int] = []
    return valid_id

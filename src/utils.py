"""Utility functions for terminal formatting and decoding visualization."""


def log_decode_step(
    state: str,
    valid_count: int,
    total_vocab: int,
    chosen_token: str,
    top_candidates: list[tuple[str, float]],
    current_json: str,
) -> None:
    """Print a single constrained-decoding step for visualization.

    Args:
        state (str): Current generation state or field name.
        valid_count (int): Number of unmasked tokens allowed at this step.
        total_vocab (int): Total size of the model vocabulary.
        chosen_token (str): String representation of the selected token.
        top_candidates (list[tuple[str, float]]): Top candidate tokens and
            their softmax probabilities (in percent).
        current_json (str): Progressive reconstructed JSON string.
    """
    candidates_str = " | ".join(
        f"{repr(tok)} ({prob:.1f}%)" for tok, prob in top_candidates
    )
    print(
        f"\n  \033[1;36m[{state}]\033[0m "
        f"Unmasked: \033[1m{valid_count}/{total_vocab}\033[0m tokens\n"
        f"    Chosen : \033[1;32m{repr(chosen_token)}\033[0m  "
        f"(Top: {candidates_str})\n"
        f"    JSON   : \033[0;37m{current_json}\033[0m"
    )


def error(msg: str) -> str:
    """Format an error message with red prefix.

    Args:
        msg (str): Error description message.

    Returns:
        str: ANSI formatted error string.
    """
    return f"\033[0;31m[Error]\033[0m {msg}"


def bold_y(msg: str) -> str:
    """Format a string with bold yellow text style.

    Args:
        msg (str): Text to format in bold.

    Returns:
        str: ANSI bold formatted string.
    """
    return f"\033[1;38;5;178m{msg}\033[0m"


def bold_w(msg: str) -> str:
    """Format a string with bold white text style.

    Args:
        msg (str): Text to format in bold.

    Returns:
        str: ANSI bold formatted string.
    """
    return f"\033[1m{msg}\033[0m"

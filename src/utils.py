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

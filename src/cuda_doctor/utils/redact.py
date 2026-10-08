"""Privacy redaction utilities.

Reports may be shared publicly (GitHub issues, support tickets), so paths and
other free-form strings are rewritten before reporting:

- the user's home directory becomes ``~``;
- any leftover occurrence of the username becomes ``<user>``.

Example: ``C:\\Users\\alice\\project`` -> ``~\\project``.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any


def default_home() -> str:
    """Return the current user's home directory as a string."""
    return str(Path.home())


def redact_text(text: str, home: str | None = None) -> str:
    """Redact a single string (home directory, then username)."""
    if not text:
        return text
    home = home if home is not None else default_home()
    out = text.replace(home, "~") if home else text
    user = os.path.basename(home.rstrip("/\\"))
    if user and user != "~" and user in out:
        out = out.replace(user, "<user>")
    return out


def redact_value(value: Any, home: str | None = None) -> Any:
    """Recursively normalize and redact structured data.

    - ``Enum`` members become their values (JSON-ready);
    - strings are redacted via :func:`redact_text`;
    - tuples become lists (JSON-ready);
    - dicts and lists are walked recursively.

    Passing ``home=""`` performs normalization without redaction.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return redact_text(value, home) if home else value
    if isinstance(value, dict):
        return {redact_value(k, home): redact_value(v, home) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item, home) for item in value]
    return value

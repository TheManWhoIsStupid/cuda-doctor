"""Tests for privacy redaction."""

from __future__ import annotations

from cuda_doctor.core.enums import Severity
from cuda_doctor.utils.redact import redact_text, redact_value

HOME = "/home/alice"


class TestRedactText:
    def test_home_replaced(self):
        assert redact_text("/home/alice/project", HOME) == "~/project"

    def test_exact_home(self):
        assert redact_text("/home/alice", HOME) == "~"

    def test_username_leftover_masked(self):
        assert redact_text("/data2/alice/workspace", HOME) == "/data2/<user>/workspace"

    def test_unrelated_text_unchanged(self):
        assert redact_text("/usr/local/cuda-12.4", HOME) == "/usr/local/cuda-12.4"

    def test_empty(self):
        assert redact_text("", HOME) == ""

    def test_windows_style(self):
        assert redact_text(r"C:\Users\alice\project", r"C:\Users\alice") == r"~\project"

    def test_empty_home_is_noop(self):
        assert redact_text("/home/alice/x", "") == "/home/alice/x"


class TestRedactValue:
    def test_nested_structures(self):
        data = {
            "path": "/home/alice/venv",
            "items": ["/home/alice/a", ("plain", 42)],
            "nested": {"deep": "/home/alice/deep"},
        }
        out = redact_value(data, home=HOME)
        assert out["path"] == "~/venv"
        assert out["items"][0] == "~/a"
        assert out["items"][1] == ["plain", 42]
        assert out["nested"]["deep"] == "~/deep"

    def test_enums_become_values(self):
        assert redact_value(Severity.INFO, home=HOME) == "INFO"

    def test_non_strings_untouched(self):
        assert redact_value(7, home=HOME) == 7
        assert redact_value(None, home=HOME) is None
        assert redact_value(True, home=HOME) is True

    def test_empty_home_normalizes_only(self):
        out = redact_value({"s": "/home/alice/x", "sev": Severity.ERROR}, home="")
        assert out == {"s": "/home/alice/x", "sev": "ERROR"}

"""Machine-readable JSON report."""

from __future__ import annotations

import json
from typing import Any

from cuda_doctor.core.models import snapshot_to_dict
from cuda_doctor.reporters.base import Reporter, ReportInputs
from cuda_doctor.version import __version__

SCHEMA_VERSION = 1


class JsonReporter(Reporter):
    """Full redacted snapshot + issues + summary, ready for diffing/archiving."""

    def render(self, inputs: ReportInputs) -> str:
        payload = self.build(inputs)
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)

    @staticmethod
    def build(inputs: ReportInputs) -> dict[str, Any]:
        result = inputs.result
        return {
            "schema_version": SCHEMA_VERSION,
            "tool": {"name": "cuda-doctor", "version": __version__},
            "generated_at": inputs.generated_at,
            "environment": snapshot_to_dict(inputs.snapshot, redact=True),
            "issues": [issue.to_dict(redact=True) for issue in result.issues],
            "summary": result.summary.to_dict(),
            "check_errors": dict(result.check_errors),
        }

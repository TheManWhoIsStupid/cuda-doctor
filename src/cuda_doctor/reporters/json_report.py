"""Machine-readable JSON report."""

from __future__ import annotations

import json
from typing import Any

from cuda_doctor.core.models import snapshot_to_dict
from cuda_doctor.reporters.base import Reporter, ReportInputs
from cuda_doctor.utils.redact import redact_value
from cuda_doctor.version import __version__

SCHEMA_VERSION = 2


class JsonReporter(Reporter):
    """Full redacted snapshot + issues + diagnoses, ready for diffing/archiving."""

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
            # Schema v2 (additive): every finalized diagnosis through the
            # accepted Diagnosis serializer — machine-complete, never
            # filtered by presentation suppression.
            "diagnoses": [
                diagnosis.to_dict(redact=True) for diagnosis in result.diagnoses
            ],
            "summary": result.summary.to_dict(),
            # Exception texts can embed absolute paths — redact like the rest.
            "check_errors": redact_value(dict(result.check_errors)),
            # Separate layer from check_errors (bounded values by contract);
            # redaction here is defense-in-depth only.
            "diagnosis_errors": redact_value(dict(result.diagnosis_errors)),
        }

"""Interpretation of PyTorch's bundled CUDA runtime vs. the local toolkit.

Core product rule: a mismatch between the locally installed CUDA Toolkit and
``torch.version.cuda`` is NOT automatically an error — official PyTorch wheels
bundle their own CUDA runtime libraries. This module centralizes that
knowledge (and the exceptions where a mismatch does matter).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cuda_doctor.utils.versions import CudaVersion, compare_versions


@dataclass(frozen=True)
class TorchCudaInterpretation:
    relation: str  # "match" | "torch-newer" | "torch-older" | "unknown"
    torch_runtime: CudaVersion | None
    local_toolkit: CudaVersion | None
    explanation: str


def interpret_torch_cuda(
    torch_cuda: CudaVersion | None,
    toolkit: CudaVersion | None,
) -> TorchCudaInterpretation:
    """Explain the relationship between torch's CUDA and the local toolkit."""
    if torch_cuda is None or toolkit is None:
        return TorchCudaInterpretation(
            "unknown",
            torch_cuda,
            toolkit,
            "Not enough information to compare PyTorch's CUDA runtime with the local toolkit.",
        )
    order = compare_versions(torch_cuda, toolkit)
    if order == 0:
        return TorchCudaInterpretation(
            "match", torch_cuda, toolkit, "PyTorch's CUDA runtime matches the local toolkit."
        )
    relation = "torch-newer" if order > 0 else "torch-older"
    base = (
        f"PyTorch was built for CUDA {torch_cuda} while the local toolkit is CUDA {toolkit}. "
        "This is usually fine: official PyTorch wheels bundle their own CUDA runtime "
        "libraries, so the versions do not have to match."
    )
    if torch_cuda.major != toolkit.major:
        base += (
            " It mainly matters when compiling custom CUDA extensions — prefer a toolkit "
            "whose major version matches PyTorch's to avoid ABI surprises."
        )
    else:
        base += " Same major version, so even extension builds are typically unproblematic."
    return TorchCudaInterpretation(relation, torch_cuda, toolkit, base)


@dataclass(frozen=True)
class KnownIssue:
    """A known problem signature with advice."""

    id: str
    title: str
    advice: str
    fragments: tuple[str, ...] = ()


class KnownIssues:
    """Substring matcher over the bundled known-issue signatures."""

    def __init__(self, groups: Mapping[str, Sequence[KnownIssue]]) -> None:
        self._groups = {name: tuple(items) for name, items in groups.items()}

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> KnownIssues:
        groups: dict[str, tuple[KnownIssue, ...]] = {}
        if isinstance(data, Mapping):
            for group, entries in data.items():
                if group == "comment" or not isinstance(entries, (list, tuple)):
                    continue
                parsed = tuple(
                    KnownIssue(
                        id=str(entry.get("id", "KI")),
                        title=str(entry.get("title", "")),
                        advice=str(entry.get("advice", "")),
                        fragments=tuple(str(item) for item in entry.get("contains", ())),
                    )
                    for entry in entries
                    if isinstance(entry, Mapping)
                )
                groups[str(group)] = parsed
        return cls(groups)

    def match(self, group: str, text: str | None) -> KnownIssue | None:
        """First known issue whose fragments appear in ``text`` (case-insensitive)."""
        if not text:
            return None
        lowered = text.lower()
        for issue in self._groups.get(group, ()):
            if any(fragment.lower() in lowered for fragment in issue.fragments):
                return issue
        return None

# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class AssociationResult:
    status: str
    error: str
    outprefix: str
    log_file: str
    elapsed_sec: float
    result_files: list[str] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return str(self.status).strip().lower() == "done"

    @classmethod
    def from_gwas_payload(cls, payload: Mapping[str, Any]) -> "AssociationResult":
        return cls(
            status=str(payload.get("status", "")),
            error=str(payload.get("error", "")),
            outprefix=str(payload.get("outprefix", "")),
            log_file=str(payload.get("log_file", "")),
            elapsed_sec=float(payload.get("elapsed_sec", 0.0) or 0.0),
            result_files=[str(x) for x in payload.get("result_files", []) or []],
            summary_rows=[dict(x) for x in payload.get("summary_rows", []) or []],
            traits=[str(x) for x in payload.get("traits", []) or []],
            payload=dict(payload),
        )


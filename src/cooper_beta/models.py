from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResidueRecord:
    """C-alpha residue record used by the public loader API."""

    res_id: int
    coord: Any
    is_sheet: bool
    chain: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedChainPayload:
    """Prepared per-chain payload passed into the chain analyzer."""

    filename: str
    chain: str
    residues_data: list[dict[str, object]]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PreparedChainPayload:
        return cls(
            filename=str(payload.get("filename", "")),
            chain=str(payload.get("chain", "")),
            residues_data=list(payload.get("residues_data", []) or []),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LayerDiagnostic:
    """Per-slice geometric diagnostic returned by the analyzer."""

    z: float
    n_points: int
    valid: bool
    reason: str
    fit: dict[str, float] | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> LayerDiagnostic:
        fit = row.get("fit")
        return cls(
            z=float(row.get("z", 0.0) or 0.0),
            n_points=int(row.get("n_points", 0) or 0),
            valid=bool(row.get("valid", False)),
            reason=str(row.get("reason", "")),
            fit=fit if isinstance(fit, dict) else None,
            raw=dict(row),
        )

    def to_dict(self) -> dict[str, object]:
        data = dict(self.raw)
        data.update({"z": self.z, "n_points": self.n_points, "valid": self.valid, "reason": self.reason})
        if self.fit is not None:
            data["fit"] = self.fit
        return data


@dataclass(frozen=True)
class AnalysisReport:
    """Structured report for one analyzed chain before CSV row serialization."""

    is_barrel: bool | None
    score: float
    score_adjust: float
    valid_layers: int = 0
    total_layers: int = 0
    total_scored_layers: int = 0
    avg_radius: float = 0.0
    message: str = ""
    layer_details: list[LayerDiagnostic] = field(default_factory=list)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, report: Mapping[str, object]) -> AnalysisReport:
        layers = [
            LayerDiagnostic.from_mapping(layer)
            for layer in list(report.get("layer_details", []) or [])
            if isinstance(layer, Mapping)
        ]
        return cls(
            is_barrel=report.get("is_barrel") if isinstance(report.get("is_barrel"), bool) else None,
            score=float(report.get("score", 0.0) or 0.0),
            score_adjust=float(report.get("score_adjust", 0.0) or 0.0),
            valid_layers=int(report.get("valid_layers", 0) or 0),
            total_layers=int(report.get("total_layers", 0) or 0),
            total_scored_layers=int(report.get("total_scored_layers", 0) or 0),
            avg_radius=float(report.get("avg_radius", 0.0) or 0.0),
            message=str(report.get("msg", "")),
            layer_details=layers,
            raw=dict(report),
        )

    def to_dict(self) -> dict[str, object]:
        data = dict(self.raw)
        data.update(
            {
                "is_barrel": self.is_barrel,
                "score": self.score,
                "score_adjust": self.score_adjust,
                "valid_layers": self.valid_layers,
                "total_layers": self.total_layers,
                "total_scored_layers": self.total_scored_layers,
                "avg_radius": self.avg_radius,
                "msg": self.message,
                "layer_details": [layer.to_dict() for layer in self.layer_details],
            }
        )
        return data


@dataclass(frozen=True)
class DetectionResult:
    """Stable public result for one structure chain."""

    filename: str
    chain: str
    result: str
    result_stage: str
    reason: str
    decision_score: float = 0.0
    decision_basis: str = ""
    decision_threshold: float = 0.0
    score_raw: float = 0.0
    score_adjust: float = 0.0
    valid_layers: int = 0
    scored_layers: int = 0
    total_layers: int = 0
    valid_layer_frac: float = 0.0
    scored_layer_frac: float = 0.0
    junk_layers: int = 0
    invalid_layers: int = 0
    avg_radius: float = 0.0
    chain_residues: int = 0
    sheet_residues: int = 0
    informative_slices: int = 0
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> DetectionResult:
        def to_float(key: str) -> float:
            try:
                return float(row.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        def to_int(key: str) -> int:
            try:
                return int(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        return cls(
            filename=str(row.get("filename", "")),
            chain=str(row.get("chain", "")),
            result=str(row.get("result", "")),
            result_stage=str(row.get("result_stage", "")),
            reason=str(row.get("reason", "")),
            decision_score=to_float("decision_score"),
            decision_basis=str(row.get("decision_basis", "")),
            decision_threshold=to_float("decision_threshold"),
            score_raw=to_float("score_raw"),
            score_adjust=to_float("score_adjust"),
            valid_layers=to_int("valid_layers"),
            scored_layers=to_int("scored_layers"),
            total_layers=to_int("total_layers"),
            valid_layer_frac=to_float("valid_layer_frac"),
            scored_layer_frac=to_float("scored_layer_frac"),
            junk_layers=to_int("junk_layers"),
            invalid_layers=to_int("invalid_layers"),
            avg_radius=to_float("avg_radius"),
            chain_residues=to_int("chain_residues"),
            sheet_residues=to_int("sheet_residues"),
            informative_slices=to_int("informative_slices"),
            raw=dict(row),
        )

    def to_dict(self) -> dict[str, object]:
        data = dict(self.raw)
        for key, value in asdict(self).items():
            if key != "raw":
                data[key] = value
        return data


@dataclass(frozen=True)
class PipelineRunResult:
    """Structured result for a complete Cooper-Beta run."""

    rows: list[DetectionResult]
    input_files: list[str] = field(default_factory=list)
    output_path: str | None = None
    config: object | None = None

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        input_files: Iterable[str] | None = None,
        output_path: str | None = None,
        config: object | None = None,
    ) -> PipelineRunResult:
        return cls(
            rows=[DetectionResult.from_row(row) for row in rows],
            input_files=list(input_files or []),
            output_path=output_path,
            config=config,
        )

    @property
    def result_counts(self) -> dict[str, int]:
        return dict(Counter(row.result for row in self.rows))

    def to_rows(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]

    def raw_rows(self) -> list[dict[str, object]]:
        """Return original row mappings for backward-compatible callers."""
        return [dict(row.raw) if row.raw else row.to_dict() for row in self.rows]

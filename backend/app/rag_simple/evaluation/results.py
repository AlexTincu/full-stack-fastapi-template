"""JSON snapshots of aggregated per-stage retrieval metrics, for before/after comparison."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    return slug or "snapshot"


@dataclass
class EvalSnapshot:
    label: str
    timestamp: str
    config: dict
    stage_metrics: dict[str, dict]
    notes: str = ""

    def save(self, results_dir: Path) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.timestamp}_{_slugify(self.label)}.json"
        path = results_dir / filename
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "EvalSnapshot":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def list_snapshots(results_dir: Path) -> list[EvalSnapshot]:
    """All snapshots in results_dir, oldest first (filenames start with an ISO-8601 timestamp)."""
    if not results_dir.exists():
        return []
    return [EvalSnapshot.load(path) for path in sorted(results_dir.glob("*.json"))]


def diff_snapshots(a: EvalSnapshot, b: EvalSnapshot) -> dict[str, dict]:
    """Per-stage, per-metric delta (b - a), for before/after comparison."""
    stages = sorted(set(a.stage_metrics) | set(b.stage_metrics))
    diff: dict[str, dict] = {}
    for stage in stages:
        a_metrics = a.stage_metrics.get(stage, {})
        b_metrics = b.stage_metrics.get(stage, {})
        metric_names = sorted(set(a_metrics) | set(b_metrics))
        diff[stage] = {
            metric: b_metrics.get(metric, 0.0) - a_metrics.get(metric, 0.0) for metric in metric_names
        }
    return diff

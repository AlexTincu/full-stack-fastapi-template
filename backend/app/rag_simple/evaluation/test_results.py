from pathlib import Path

import pytest

from app.rag_simple.evaluation.results import EvalSnapshot, diff_snapshots, list_snapshots


def _snapshot(label: str, timestamp: str, semantic_mrr: float) -> EvalSnapshot:
    return EvalSnapshot(
        label=label,
        timestamp=timestamp,
        config={
            "RETRIEVAL_K": 20,
            "FINAL_K": 20,
            "CHAT_MODEL": "openai:gpt-4.1-mini",
            "EMBEDDING_MODEL": "openai:text-embedding-3-large",
        },
        stage_metrics={"semantic": {"mrr": semantic_mrr, "recall_at_10": 0.5}},
    )


def test_save_writes_json_with_timestamp_and_slugified_label(tmp_path: Path):
    snapshot = _snapshot("baseline run", "20260917T120000Z", 0.5)

    path = snapshot.save(tmp_path)

    assert path.name == "20260917T120000Z_baseline-run.json"
    assert path.exists()


def test_save_creates_results_dir_if_missing(tmp_path: Path):
    results_dir = tmp_path / "results"
    snapshot = _snapshot("baseline", "20260917T120000Z", 0.5)

    snapshot.save(results_dir)

    assert results_dir.exists()


def test_load_round_trips_saved_snapshot(tmp_path: Path):
    snapshot = _snapshot("baseline", "20260917T120000Z", 0.75)
    path = snapshot.save(tmp_path)

    loaded = EvalSnapshot.load(path)

    assert loaded == snapshot


def test_list_snapshots_returns_oldest_first(tmp_path: Path):
    older = _snapshot("baseline", "20260917T090000Z", 0.5)
    newer = _snapshot("contextualize-on", "20260917T150000Z", 0.6)
    newer.save(tmp_path)
    older.save(tmp_path)  # saved out of chronological order

    snapshots = list_snapshots(tmp_path)

    assert [s.label for s in snapshots] == ["baseline", "contextualize-on"]


def test_list_snapshots_missing_dir_returns_empty_list(tmp_path: Path):
    assert list_snapshots(tmp_path / "missing") == []


def test_diff_snapshots_computes_delta_per_stage_and_metric():
    a = _snapshot("baseline", "20260917T090000Z", 0.5)
    b = _snapshot("contextualize-on", "20260917T150000Z", 0.8)

    delta = diff_snapshots(a, b)

    assert delta["semantic"]["mrr"] == pytest.approx(0.3)
    assert delta["semantic"]["recall_at_10"] == pytest.approx(0.0)


def test_diff_snapshots_handles_stage_missing_from_one_side():
    a = EvalSnapshot(label="a", timestamp="t1", config={}, stage_metrics={"semantic": {"mrr": 0.5}})
    b = EvalSnapshot(
        label="b",
        timestamp="t2",
        config={},
        stage_metrics={"semantic": {"mrr": 0.5}, "fused": {"mrr": 0.9}},
    )

    delta = diff_snapshots(a, b)

    assert delta["fused"]["mrr"] == pytest.approx(0.9)

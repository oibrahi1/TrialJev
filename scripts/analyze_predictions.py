#!/usr/bin/env python3
"""Post-hoc analysis for TrialJev ``eval-*-preds.jsonl`` files.

This script does not run or select a model. It reads the row-level output written by
``trialmatch.evaluate`` and reports:

* confusion matrix and per-class precision/recall/F1;
* accuracy, macro-F1, AUROC, average precision, Brier score, and ECE;
* per-patient performance summaries;
* patient-clustered bootstrap confidence intervals; and
* optional paired metric differences between two checkpoints on identical examples.

Example (run from the repository root):

    python scripts/analyze_predictions.py \
      --preds runs/run1-q38-27b-base/step-400/eval-test-preds.jsonl \
      --compare runs/run1-q35-27b-base/step-400/eval-test-preds.jsonl \
      --bootstrap 10000 --seed 20260929 \
      --out runs/comparisons/q38-vs-q35-test-clustered.json

The bootstrap resamples patients, not individual patient-trial pairs. This preserves
within-patient dependence and is the correct unit for the 50-patient TREC-2022 test.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES = {0: "excluded", 1: "eligible"}


def as_binary(value: Any) -> int:
    """Convert evaluator booleans/strings to excluded=0, eligible=1."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "eligible"}:
        return 1
    if text in {"false", "0", "excluded", "ineligible"}:
        return 0
    raise ValueError(f"cannot map value to binary eligibility: {value!r}")


def row_id(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    if meta.get("id"):
        return str(meta["id"])
    patient = meta.get("patient_id")
    trial = meta.get("nct_id")
    if patient is not None and trial is not None:
        return f"{patient}/{trial}"
    raise ValueError("prediction row lacks meta.id and patient_id/nct_id")


def patient_id(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    value = meta.get("patient_id")
    if value is None:
        value = meta.get("group_id")
    if value is None:
        raise ValueError(f"prediction row {row_id(row)!r} lacks patient_id/group_id")
    return str(value)


def probability_eligible(row: dict[str, Any]) -> float:
    probs = row.get("probs")
    if not isinstance(probs, dict):
        raise ValueError(f"prediction row {row_id(row)!r} lacks a probs object")
    for key in ("true", "eligible", "1"):
        if key in probs:
            value = float(probs[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid probability in row {row_id(row)!r}: {value}")
            return value
    raise ValueError(f"prediction row {row_id(row)!r} lacks an eligible/true probability")


def load_predictions(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row_id(row)
            if key in seen:
                raise ValueError(f"duplicate row id {key!r} at {path}:{line_number}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"no prediction rows found in {path}")

    ids = np.asarray([row_id(row) for row in rows], dtype=object)
    patients = np.asarray([patient_id(row) for row in rows], dtype=object)
    gold = np.asarray([as_binary(row["gold"]) for row in rows], dtype=np.int8)
    pred = np.asarray([as_binary(row["pred"]) for row in rows], dtype=np.int8)
    prob = np.asarray([probability_eligible(row) for row in rows], dtype=float)
    return {
        "path": str(path),
        "ids": ids,
        "patients": patients,
        "gold": gold,
        "pred": pred,
        "prob": prob,
    }


def weighted_ece(
    gold: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    weights: np.ndarray | None = None,
    bins: int = 10,
) -> float:
    if weights is None:
        weights = np.ones(len(gold), dtype=float)
    confidence = np.maximum(prob, 1.0 - prob)
    correct = (gold == pred).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    result = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        mass = float(weights[mask].sum())
        if mass:
            accuracy = float(np.average(correct[mask], weights=weights[mask]))
            mean_confidence = float(np.average(confidence[mask], weights=weights[mask]))
            result += (mass / total) * abs(accuracy - mean_confidence)
    return result


def weighted_classification_stats(
    gold: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for label in (0, 1):
        true_positive = float(weights[(gold == label) & (pred == label)].sum())
        false_positive = float(weights[(gold != label) & (pred == label)].sum())
        false_negative = float(weights[(gold == label) & (pred != label)].sum())
        support = float(weights[gold == label].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def weighted_auroc(gold: np.ndarray, prob: np.ndarray, weights: np.ndarray) -> float:
    """Weighted binary AUROC with half credit for tied scores."""
    positive_total = float(weights[gold == 1].sum())
    negative_total = float(weights[gold == 0].sum())
    if positive_total == 0.0 or negative_total == 0.0:
        return math.nan
    order = np.argsort(prob, kind="mergesort")
    score = prob[order]
    ordered_gold = gold[order]
    ordered_weights = weights[order]
    group_starts = np.flatnonzero(np.r_[True, score[1:] != score[:-1]])
    positive_by_group = np.add.reduceat(ordered_weights * (ordered_gold == 1), group_starts)
    negative_by_group = np.add.reduceat(ordered_weights * (ordered_gold == 0), group_starts)
    negative_before = np.cumsum(negative_by_group) - negative_by_group
    concordant = float(
        np.sum(positive_by_group * (negative_before + 0.5 * negative_by_group))
    )
    return concordant / (positive_total * negative_total)


def weighted_average_precision(gold: np.ndarray, prob: np.ndarray, weights: np.ndarray) -> float:
    """Weighted non-interpolated average precision, grouping tied scores."""
    positive_total = float(weights[gold == 1].sum())
    if positive_total == 0.0:
        return math.nan
    order = np.argsort(-prob, kind="mergesort")
    score = prob[order]
    ordered_gold = gold[order]
    ordered_weights = weights[order]
    group_starts = np.flatnonzero(np.r_[True, score[1:] != score[:-1]])
    positive_by_group = np.add.reduceat(ordered_weights * (ordered_gold == 1), group_starts)
    negative_by_group = np.add.reduceat(ordered_weights * (ordered_gold == 0), group_starts)
    true_positive = np.cumsum(positive_by_group)
    predicted_positive = true_positive + np.cumsum(negative_by_group)
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_positive > 0.0,
    )
    return float(np.sum((positive_by_group / positive_total) * precision))


def metric_values(data: dict[str, Any], weights: np.ndarray | None = None) -> dict[str, float]:
    gold = data["gold"]
    pred = data["pred"]
    prob = data["prob"]
    if weights is None:
        weights = np.ones(len(gold), dtype=float)
    total = float(weights.sum())
    accuracy = float(np.dot(weights, gold == pred) / total)
    brier = float(np.dot(weights, (prob - gold) ** 2) / total)
    class_stats = weighted_classification_stats(gold, pred, weights)
    values = {
        "accuracy": accuracy,
        "macro_f1": float(np.mean([class_stats[0]["f1"], class_stats[1]["f1"]])),
        "brier_conventional_binary": brier,
        "brier_evaluator_two_class_sum": 2.0 * brier,
        "ece_10_bin": weighted_ece(gold, pred, prob, weights=weights, bins=10),
        "auroc": weighted_auroc(gold, prob, weights),
        "average_precision_eligible": weighted_average_precision(gold, prob, weights),
    }
    return values


def point_report(data: dict[str, Any]) -> dict[str, Any]:
    gold = data["gold"]
    pred = data["pred"]
    weights = np.ones(len(gold), dtype=float)
    class_stats = weighted_classification_stats(gold, pred, weights)
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (gold, pred), 1)
    patient_rows: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(data["patients"]):
        patient_rows[str(patient)].append(index)

    per_patient: dict[str, Any] = {}
    patient_accuracies: list[float] = []
    for patient in sorted(patient_rows):
        indexes = np.asarray(patient_rows[patient], dtype=int)
        subset = {
            "gold": gold[indexes],
            "pred": pred[indexes],
            "prob": data["prob"][indexes],
        }
        metrics = metric_values(subset)
        patient_accuracies.append(metrics["accuracy"])
        per_patient[patient] = {
            "n": int(len(indexes)),
            "eligible": int(gold[indexes].sum()),
            "excluded": int(len(indexes) - gold[indexes].sum()),
            **metrics,
        }

    return {
        "source": data["path"],
        "pairs": int(len(gold)),
        "patients": int(len(patient_rows)),
        "class_distribution": {
            "excluded": int((gold == 0).sum()),
            "eligible": int((gold == 1).sum()),
        },
        "metrics": metric_values(data),
        "confusion_matrix": {
            "labels": ["excluded", "eligible"],
            "rows_gold_columns_predicted": matrix.astype(int).tolist(),
        },
        "per_class": {
            CLASS_NAMES[label]: {
                "precision": class_stats[label]["precision"],
                "recall": class_stats[label]["recall"],
                "f1": class_stats[label]["f1"],
                "support": int(class_stats[label]["support"]),
            }
            for label in (0, 1)
        },
        "patient_macro_accuracy": float(np.mean(patient_accuracies)),
        "patient_accuracy_median": float(np.median(patient_accuracies)),
        "per_patient": per_patient,
    }


def patient_index(data: dict[str, Any]) -> tuple[list[str], dict[str, np.ndarray]]:
    mapping: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(data["patients"]):
        mapping[str(patient)].append(index)
    patients = sorted(mapping)
    return patients, {patient: np.asarray(mapping[patient], dtype=int) for patient in patients}


def cluster_bootstrap(
    data: dict[str, Any],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    patients, indexes = patient_index(data)
    rng = np.random.default_rng(seed)
    names = list(metric_values(data))
    samples: dict[str, list[float]] = {name: [] for name in names}
    n_rows = len(data["gold"])

    for _ in range(iterations):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        multiplicity = Counter(drawn.tolist())
        weights = np.zeros(n_rows, dtype=float)
        for patient, count in multiplicity.items():
            weights[indexes[patient]] = count
        values = metric_values(data, weights=weights)
        for name, value in values.items():
            if math.isfinite(value):
                samples[name].append(value)

    point = metric_values(data)
    report: dict[str, Any] = {}
    for name in names:
        values = np.asarray(samples[name], dtype=float)
        report[name] = {
            "estimate": point[name],
            "ci95_percentile": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "valid_bootstraps": int(len(values)),
        }
    return {
        "unit": "patient",
        "patients": len(patients),
        "iterations": iterations,
        "seed": seed,
        "metrics": report,
    }


def align_comparison(reference: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    reference_positions = {key: index for index, key in enumerate(reference["ids"].tolist())}
    comparison_positions = {key: index for index, key in enumerate(comparison["ids"].tolist())}
    reference_ids = set(reference_positions)
    comparison_ids = set(comparison_positions)
    if reference_ids != comparison_ids:
        missing = sorted(reference_ids - comparison_ids)[:10]
        extra = sorted(comparison_ids - reference_ids)[:10]
        raise ValueError(
            "paired files do not contain identical row ids; "
            f"missing_from_compare={missing}, extra_in_compare={extra}"
        )
    order = np.asarray([comparison_positions[key] for key in reference["ids"]], dtype=int)
    aligned = {
        "path": comparison["path"],
        "ids": comparison["ids"][order],
        "patients": comparison["patients"][order],
        "gold": comparison["gold"][order],
        "pred": comparison["pred"][order],
        "prob": comparison["prob"][order],
    }
    if not np.array_equal(reference["gold"], aligned["gold"]):
        raise ValueError("paired files disagree on gold labels")
    if not np.array_equal(reference["patients"], aligned["patients"]):
        raise ValueError("paired files disagree on patient ids")
    return aligned


def paired_cluster_bootstrap(
    reference: dict[str, Any],
    comparison: dict[str, Any],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    comparison = align_comparison(reference, comparison)
    patients, indexes = patient_index(reference)
    rng = np.random.default_rng(seed)
    reference_point = metric_values(reference)
    comparison_point = metric_values(comparison)
    names = list(reference_point)
    deltas: dict[str, list[float]] = {name: [] for name in names}
    n_rows = len(reference["gold"])

    for _ in range(iterations):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        multiplicity = Counter(drawn.tolist())
        weights = np.zeros(n_rows, dtype=float)
        for patient, count in multiplicity.items():
            weights[indexes[patient]] = count
        left = metric_values(reference, weights=weights)
        right = metric_values(comparison, weights=weights)
        for name in names:
            delta = left[name] - right[name]
            if math.isfinite(delta):
                deltas[name].append(delta)

    report: dict[str, Any] = {}
    for name in names:
        values = np.asarray(deltas[name], dtype=float)
        low = float(np.quantile(values, 0.025))
        high = float(np.quantile(values, 0.975))
        report[name] = {
            "reference": reference_point[name],
            "comparison": comparison_point[name],
            "delta_reference_minus_comparison": reference_point[name] - comparison_point[name],
            "delta_ci95_percentile": [low, high],
            "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
            "valid_bootstraps": int(len(values)),
        }
    return {
        "reference_source": reference["path"],
        "comparison_source": comparison["path"],
        "unit": "patient",
        "patients": len(patients),
        "iterations": iterations,
        "seed": seed,
        "metrics": report,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, required=True,
                        help="reference eval-*-preds.jsonl (for example Qwen3.8-27B)")
    parser.add_argument("--compare", type=Path,
                        help="optional paired eval-*-preds.jsonl (for example Qwen3.5-27B)")
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="patient-clustered bootstrap iterations (default: 10000)")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise SystemExit("--bootstrap must be at least 1000 for stable percentile intervals")

    reference = load_predictions(args.preds)
    result: dict[str, Any] = {
        "schema_version": 1,
        "reference": point_report(reference),
        "reference_patient_clustered_bootstrap": cluster_bootstrap(
            reference, iterations=args.bootstrap, seed=args.seed
        ),
    }
    if args.compare:
        comparison = align_comparison(reference, load_predictions(args.compare))
        result["comparison"] = point_report(comparison)
        result["comparison_patient_clustered_bootstrap"] = cluster_bootstrap(
            comparison, iterations=args.bootstrap, seed=args.seed
        )
        result["paired_patient_clustered_bootstrap"] = paired_cluster_bootstrap(
            reference, comparison, iterations=args.bootstrap, seed=args.seed
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    concise = {
        "out": str(args.out),
        "reference": result["reference"]["metrics"],
        "reference_ci": result["reference_patient_clustered_bootstrap"]["metrics"],
    }
    if args.compare:
        concise["paired_differences"] = result["paired_patient_clustered_bootstrap"]["metrics"]
    print(json.dumps(json_safe(concise), indent=2))


if __name__ == "__main__":
    main()

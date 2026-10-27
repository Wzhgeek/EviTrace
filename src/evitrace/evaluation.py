"""Complete-coverage classification, ranking, token and component comparisons."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics
from .io import PROJECTS, read_jsonl, write_json
from .metrics import classify, ranked_metrics, classification_uncertainty, ranking_uncertainty


def evaluate(pairs_path, predictions_path, output, scores_path=None, bootstrap=2000):
    pairs = read_jsonl(pairs_path)
    ids = {r["pair_id"] for r in pairs}
    if len(ids) != len(pairs):
        raise ValueError("Duplicate candidate identities")
    predictions, usage = defaultdict(dict), defaultdict(dict)
    for row in read_jsonl(predictions_path):
        if row.get("status", "ok") != "ok":
            continue
        pair_id, arm = row["pair_id"], row["arm"]
        if pair_id not in ids or type(row["prediction"]) is not int or row["prediction"] not in (0, 1):
            raise ValueError("Invalid candidate identity or prediction")
        if pair_id in predictions[arm]:
            raise ValueError("Duplicate completed prediction")
        predictions[arm][pair_id] = row["prediction"]
        usage[arm][pair_id] = {"input_tokens": row.get("usage", {}).get("prompt_tokens"), "cases": row.get("case_count")}
    if not predictions or "EviTrace" not in predictions:
        raise ValueError("Complete EviTrace predictions are required")
    for arm, current in predictions.items():
        if set(current) != ids:
            raise ValueError(f"Incomplete prediction coverage for {arm}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    classification = {}
    for arm, current in predictions.items():
        classification[arm] = {}
        for project in PROJECTS:
            selected = [r for r in pairs if r["dataset"] == project]
            if not selected:
                raise ValueError(f"Missing project: {project}")
            classification[arm][project] = classify([r["label"] for r in selected], [current[r["pair_id"]] for r in selected])
        classification[arm]["Macro"] = {k: statistics.fmean(classification[arm][p][k] for p in PROJECTS) for k in ("precision", "recall", "f1")}
    write_json(output / "classification.json", classification)
    with (output / "classification.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Method", "Project", "P (%)", "R (%)", "F1 (%)"])
        for arm, projects in classification.items():
            for project, values in projects.items():
                writer.writerow([arm, project, *(f"{100 * values[k]:.2f}" for k in ("precision", "recall", "f1"))])
    if bootstrap:
        write_json(output / "paired_classification.json", classification_uncertainty(pairs, predictions, list(predictions), bootstrap))
        if "FTLR" in predictions:
            from .metrics import ftlr_uncertainty
            write_json(output / "paired_ftlr.json", ftlr_uncertainty(pairs, predictions, bootstrap))
    accounting = {}
    for arm, rows in usage.items():
        tokens = [r["input_tokens"] for r in rows.values()]
        cases = [r["cases"] for r in rows.values()]
        accounting[arm] = {"pairs": len(rows), "input_tokens": sum(tokens) if all(v is not None for v in tokens) else None,
                           "case_occurrences": sum(cases) if all(v is not None for v in cases) else None,
                           "case_bearing_pairs": sum(v > 0 for v in cases) if all(v is not None for v in cases) else None}
    write_json(output / "input_accounting.json", accounting)
    if scores_path:
        scores = {r["pair_id"]: r["scores"] for r in read_jsonl(scores_path)}
        if set(scores) != ids:
            raise ValueError("Ranking scores must cover exactly the classification candidates")
        groups = defaultdict(list)
        for pair in pairs:
            groups[pair["dataset"], pair["req_name"]].append(pair)
        requirements = []
        for (project, requirement), rows in groups.items():
            for method in ("VSM", "BM25", "LSI", "BM25 + LiSSA reranking", "BM25 + EviTrace reranking"):
                if method.startswith("BM25 + "):
                    arm = "LiSSA" if "LiSSA" in method else "EviTrace"
                    if arm not in predictions:
                        raise ValueError(f"Missing predictions for ranking: {arm}")
                    ordered = sorted(rows, key=lambda r: (-predictions[arm][r["pair_id"]], -scores[r["pair_id"]]["BM25"], r["pair_id"]))
                else:
                    ordered = sorted(rows, key=lambda r: (-scores[r["pair_id"]][method], r["pair_id"]))
                requirements.append({"dataset": project, "req_name": requirement, "method": method,
                                     "ranking_eligible": any(r["label"] for r in rows), **ranked_metrics([r["label"] for r in ordered])})
        ranking, zero = {}, {}
        for method in sorted({r["method"] for r in requirements}):
            ranking[method], zero[method] = {}, {}
            for project in PROJECTS:
                rows = [r for r in requirements if r["method"] == method and r["dataset"] == project]
                for table, selected in [(ranking, [r for r in rows if r["ranking_eligible"]]), (zero, rows)]:
                    table[method][project] = {m: statistics.fmean(r[m] for r in selected) for m in ("MRR", "Hit@5", "MAP", "nDCG@10", "R-precision")}
            for table in (ranking, zero):
                table[method]["Macro"] = {m: statistics.fmean(table[method][p][m] for p in PROJECTS) for m in table[method][PROJECTS[0]]}
        write_json(output / "ranking.json", ranking)
        write_json(output / "ranking_zero_inclusive.json", zero)
        if bootstrap:
            write_json(output / "paired_ranking.json", ranking_uncertainty(requirements, bootstrap))
    return {"pairs": len(pairs), "requirements": len({(r["dataset"], r["req_name"]) for r in pairs}),
            "methods": list(predictions), "macro_f1": {m: v["Macro"]["f1"] for m, v in classification.items()}}

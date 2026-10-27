"""Classification, candidate ranking and paired requirement bootstrap."""
import collections
import math
import random
import statistics
from .io import PROJECTS as DATASETS
RANKING_METRICS = ("MRR", "Hit@5", "MAP", "nDCG@10", "R-precision")

def classify(gold: list[int], predicted: list[int]) -> dict:
    tp = sum(y == p == 1 for y, p in zip(gold, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(gold, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(gold, predicted))
    tn = sum(y == p == 0 for y, p in zip(gold, predicted))
    precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
    return {"n": len(gold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "accuracy": (tp + tn) / len(gold) if gold else 0.}

def ranked_metrics(labels: list[int]) -> dict[str, float]:
    """Binary-relevance ranking; R is the number of positives in this list."""
    if not labels or any(x not in (0, 1) for x in labels):
        raise ValueError("A nonempty binary-relevance list is required")
    relevant = sum(labels)
    if not relevant:
        return dict.fromkeys(RANKING_METRICS, 0.)
    positive_ranks = [i for i, value in enumerate(labels, 1) if value]
    ap = math.fsum(j / rank for j, rank in enumerate(positive_ranks, 1)) / relevant
    dcg = math.fsum(value / math.log2(i + 1) for i, value in enumerate(labels[:10], 1))
    idcg = math.fsum(1 / math.log2(i + 1) for i in range(1, min(relevant, 10) + 1))
    return {"MRR": 1 / positive_ranks[0], "Hit@5": float(any(labels[:5])), "MAP": ap,
            "nDCG@10": dcg / idcg, "R-precision": sum(labels[:relevant]) / relevant}

def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] * (upper - position) + values[upper] * (position - lower) if upper != lower else values[lower]

def ranking_uncertainty(requirements, repetitions=2000):
    targets = ("BM25 + EviTrace reranking", "BM25 + LiSSA reranking")
    grouped = collections.defaultdict(dict)
    for row in requirements:
        if not row["ranking_eligible"]:
            continue
        grouped[row["dataset"], row["req_name"]][row["method"]] = row
    by_project = {}
    for dataset in DATASETS:
        values = [v for (d, _), v in grouped.items() if d == dataset]
        by_project[dataset] = {m: [v[targets[0]][m] - v[targets[1]][m] for v in values] for m in RANKING_METRICS}
    result = {}
    rng = random.Random(20261023)
    for metric in RANKING_METRICS:
        project_draws = {}
        for dataset in DATASETS:
            values = by_project[dataset][metric]
            draws = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repetitions)]
            project_draws[dataset] = draws
            result.setdefault(dataset, {})[metric] = {"difference": statistics.fmean(values),
                "paired_requirement_bootstrap_95ci": [quantile(draws, .025), quantile(draws, .975)]}
        macro = [statistics.fmean(project_draws[d][i] for d in DATASETS) for i in range(repetitions)]
        result.setdefault("Macro", {})[metric] = {"difference": statistics.fmean(statistics.fmean(by_project[d][metric]) for d in DATASETS),
            "paired_requirement_bootstrap_95ci": [quantile(macro, .025), quantile(macro, .975)]}
    return {"comparison": list(targets), "repetitions": repetitions, "seed": 20261023,
        "scope": "Conditional on the eight measured projects; bootstrap requirements within each project, equal project macro.", "results": result}

def classification_uncertainty(pairs, predictions, methods, repetitions=2000):
    """Sample whole requirement clusters, paired for every compared method."""
    rng = random.Random(20261023)
    project_draws, result = {}, {}
    baselines = [m for m in methods if m != "EviTrace"]
    for dataset in DATASETS:
        grouped = collections.defaultdict(list)
        for row in pairs:
            if row["dataset"] == dataset:
                grouped[row["req_name"]].append(row)
        groups = list(grouped.values())
        positives = [sum(r["label"] for r in group) for group in groups]
        counts = {}
        point = {}
        for method in methods:
            counts[method] = [(sum(r["label"] == predictions[method][r["pair_id"]] == 1 for r in group),
                               sum(r["label"] == 0 and predictions[method][r["pair_id"]] == 1 for r in group)) for group in groups]
            tp, fp = map(sum, zip(*counts[method]))
            point[method] = 2 * tp / (tp + fp + sum(positives)) if tp + fp + sum(positives) else 0.
        draws = {m: [] for m in methods}
        for _ in range(repetitions):
            indexes = rng.choices(range(len(groups)), k=len(groups))
            positive = sum(positives[i] for i in indexes)
            for method in methods:
                tp = sum(counts[method][i][0] for i in indexes)
                fp = sum(counts[method][i][1] for i in indexes)
                draws[method].append(2 * tp / (tp + fp + positive) if tp + fp + positive else 0.)
        project_draws[dataset] = draws
        result[dataset] = {method: {"EviTrace_f1": point["EviTrace"], "baseline_f1": point[method],
            "difference_f1": point["EviTrace"] - point[method],
            "paired_requirement_cluster_95ci": [quantile([a-b for a,b in zip(draws["EviTrace"], draws[method])], .025),
                                                quantile([a-b for a,b in zip(draws["EviTrace"], draws[method])], .975)]}
            for method in baselines}
    result["Macro"] = {}
    for method in baselines:
        deltas = [statistics.fmean(project_draws[d]["EviTrace"][i] - project_draws[d][method][i] for d in DATASETS)
                  for i in range(repetitions)]
        result["Macro"][method] = {"EviTrace_f1": statistics.fmean(result[d][method]["EviTrace_f1"] for d in DATASETS),
            "baseline_f1": statistics.fmean(result[d][method]["baseline_f1"] for d in DATASETS),
            "difference_f1": statistics.fmean(result[d][method]["difference_f1"] for d in DATASETS),
            "paired_requirement_cluster_95ci": [quantile(deltas, .025), quantile(deltas, .975)]}
    return {"name": "Final-configuration paired classification uncertainty", "full_method": "EviTrace",
        "baselines": baselines, "seed": 20261023, "repetitions": repetitions,
        "test_pairs": len(pairs), "requirements": sum(len({r["req_name"] for r in pairs if r["dataset"] == d}) for d in DATASETS),
        "resampling": "Whole requirement clusters resampled within each project; identical sampled clusters for every method; equal project macro.",
        "scope_note": "Conditional on these eight measured projects; does not correct pilot-driven project selection. Single-run predictions remain fixed; no independent model-run variance is estimated. Percentile intervals are not multiple-comparison-adjusted hypothesis tests.",
        "results": result}


def ftlr_uncertainty(pairs, predictions, repetitions=2000):
    """Paired FTLR reference interval with seed 42 and retained multiplicities."""
    import numpy as np
    grouped = {p: {} for p in DATASETS}
    for row in pairs:
        block = grouped[row['dataset']].setdefault(row['req_name'], np.zeros((2,4),dtype=np.int64))
        for j,method in enumerate(('EviTrace','FTLR')):
            y, predicted = row['label'], predictions[method][row['pair_id']]
            block[j,0 if predicted and y else 1 if predicted else 2 if y else 3] += 1
    arrays = {p:np.stack(list(groups.values())) for p,groups in grouped.items()}
    def f1(counts):
        tp,fp,fn = counts[...,0],counts[...,1],counts[...,2]
        denominator = 2*tp+fp+fn
        return np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator!=0)
    rng, deltas = random.Random(42), []
    for _ in range(repetitions):
        scores = [f1(array[[rng.randrange(len(array)) for __ in range(len(array))]].sum(axis=0)) for array in arrays.values()]
        score = np.mean(scores,axis=0)
        deltas.append(float(score[0]-score[1]))
    observed = np.mean([f1(array.sum(axis=0)) for array in arrays.values()],axis=0)
    return {'seed':42,'repetitions':repetitions,'EviTrace_f1':float(observed[0]),'FTLR_f1':float(observed[1]),
            'difference_f1':float(observed[0]-observed[1]),'paired_95ci':np.quantile(deltas,[.025,.975]).tolist()}

"""FTLR aggregation adapter for the maintained author implementation."""
from pathlib import Path
import json
import sys
from .io import PROJECTS, read_jsonl, write_json
from .metrics import classify


def evaluate_ftlr(checkout, manifest_path, pairs_path, output):
    """Use the released voting and call-dependency code with supplied distances."""
    sys.path.insert(0, str(Path(checkout).resolve()))
    from precalculating.TraceLinkDataStructure import ElementLevelTraceLinkDataStructure
    from traceLinkProcessing.TraceLinkProcessor import MajProcessor
    from traceLinkProcessing.SimilarityFilter import SimilarityFilter
    from traceLinkProcessing.TraceLinkCreator import CallGraphTraceLinkAggregator
    from traceLinkProcessing.NeighborHandler import NeighborStrategy
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    pairs = read_jsonl(pairs_path)
    grid = [round(.10 + .02 * i, 2) for i in range(41)]
    records, settings = [], {}
    for project in PROJECTS:
        config = manifest[project]
        names = json.loads((root / config["name_map"]).read_text())
        matrix = ElementLevelTraceLinkDataStructure.load_data_from(root / config["matrix"], root / config["artifact_map"])
        aggregator = None
        if config.get("callgraph"):
            graph = json.loads((root / config["callgraph"]).read_text())
            aggregator = CallGraphTraceLinkAggregator(.9, NeighborStrategy.both, graph)
        processor = MajProcessor(matrix, SimilarityFilter(False), min, min, grid, [.59], aggregator)
        variants = processor.run()[.59]
        selected_rows = [r for r in pairs if r["dataset"] == project]
        candidates = []
        for threshold, links in variants.items():
            if isinstance(links, str):
                continue
            selected = set()
            for link in links:
                requirement, tail = str(link).split("<->", 1)
                source_name = tail.rsplit(":", 1)[0].strip()
                selected.add((requirement.strip(), names[source_name]))
            values = [int((r["req_name"], r["code_name"]) in selected) for r in selected_rows]
            metrics = classify([r["label"] for r in selected_rows], values)
            candidates.append((metrics["f1"], -threshold, threshold, values, metrics))
        _, _, threshold, values, metrics = max(candidates)
        settings[project] = {"threshold": threshold, **metrics}
        records.extend({"pair_id": r["pair_id"], "arm": "FTLR", "prediction": p, "status": "ok"}
                       for r, p in zip(selected_rows, values))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ftlr_thresholds.json", {"majority_threshold": .59,
               "selection": "Maximum project F1 over the fixed grid using evaluation labels", "projects": settings})
    (output / "ftlr_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return settings

"""Import fixed candidate identities and original artifacts into portable manifests."""
from pathlib import Path
import json
from .io import PROJECTS, SOURCES, canonical, normalize, read_jsonl, sha, write_json


def prepare(evaluation_pairs, case_pairs, artifact_root, output):
    root, output = Path(artifact_root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = {}
    directories = {}

    def register(relative, project, side):
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Missing artifact or unsafe relative path: {relative}")
        identity = f"{project}/{side}/{path.name}"
        raw = path.read_bytes()
        record = {"artifact_id": identity, "project": project, "side": side,
                  "name": path.name, "relative_path": path.relative_to(root).as_posix(),
                  "sha256": sha(raw), "text_sha256": sha(normalize(raw).encode())}
        if identity in entries and entries[identity] != record:
            raise ValueError(f"Conflicting artifact identity: {identity}")
        entries[identity] = record
        directories[(project, side, path.parent)] = True
        return identity

    pairs, cases = [], []
    for raw in read_jsonl(evaluation_pairs):
        project = "Maven" if raw["dataset"] == "maven" else raw["dataset"]
        if project not in PROJECTS or type(raw["label"]) is not int or raw["label"] not in (0, 1):
            raise ValueError("Invalid evaluation project or label")
        r = register(raw["requirement_path"], project, "requirement")
        c = register(raw["code_path"], project, "code")
        pair_id = raw.get("pair_id") or sha(canonical([project, r, c]))
        pairs.append({"pair_id": pair_id, "dataset": project, "requirement_id": r,
                      "code_id": c, "req_name": entries[r]["name"], "code_name": entries[c]["name"], "label": raw["label"]})
    for raw in read_jsonl(case_pairs):
        project = raw.get("source_project", raw.get("project"))
        if project not in SOURCES or type(raw["label"]) is not int or raw["label"] not in (0, 1):
            raise ValueError("Cases must come from the four external projects and have binary labels")
        r = register(raw["requirement_path"], project, "requirement")
        c = register(raw["code_path"], project, "code")
        cases.append({"case_id": raw.get("case_id") or sha(canonical([project, r, c, raw["label"]])),
                      "project": project, "requirement_id": r, "code_id": c, "label": raw["label"]})
    # Unlabeled project artifacts are needed for corpus fitting and calibration.
    for project, side, directory in list(directories):
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                register(path.relative_to(root), project, side)
    target_hashes = {v["text_sha256"] for v in entries.values() if v["project"] in PROJECTS}
    if any(v["project"] in SOURCES and v["text_sha256"] in target_hashes for v in entries.values()):
        raise ValueError("Source and evaluation artifact text overlap; resolve the corpus before calibration")
    if len({p["pair_id"] for p in pairs}) != len(pairs) or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate candidate or case identity")
    for name, rows in [("pairs", pairs), ("cases", cases), ("artifacts", sorted(entries.values(), key=lambda r: r["artifact_id"]))]:
        (output / (name + ".jsonl")).write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    report = {"candidate_pairs": len(pairs), "case_pairs": len(cases), "artifacts": len(entries),
              "requirements": len({p["requirement_id"] for p in pairs}),
              "positive_pairs": sum(p["label"] for p in pairs), "source_projects": list(SOURCES)}
    write_json(output / "manifest.json", report)
    return report

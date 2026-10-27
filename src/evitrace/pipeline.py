"""Final method, same-model baselines and matched component controls."""
from pathlib import Path
import json
from .io import SOURCES, Artifacts, read_jsonl, sha, write_json
from .gate import EvidenceGate
from .representation import augmented_code
from .prompts import (SER_SYSTEM, SER_TASK, SER_TEMPLATE, DIRECT_TASK, SUMMARY_SYSTEM,
                      SUMMARY_TEMPLATE, LISSA_SYSTEM, LISSA_USER_TEMPLATE, parse_prediction)

ARMS = ("EviTrace", "without_cases", "without_SER", "raw_source", "ungated", "Direct_LLM", "LiSSA")


def render_cases(cases):
    return "\n".join(f"Case {i}: " + json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                     for i, row in enumerate(cases, 1)) if cases else "No cases provided."


def messages_for(arm, requirement, raw_code, represented_code, cases):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if arm == "LiSSA":
        text = (LISSA_USER_TEMPLATE.replace("{source_type}", "requirement")
                .replace("{source_content}", requirement).replace("{target_type}", "source code")
                .replace("{target_content}", raw_code))
        return [{"role": "system", "content": LISSA_SYSTEM}, {"role": "user", "content": text}], 1536, "LiSSA"
    raw = arm in ("raw_source", "Direct_LLM")
    direct = arm in ("without_SER", "Direct_LLM")
    template = SER_TEMPLATE.replace("summary plus verifiable original source evidence", "complete original source code") if raw else SER_TEMPLATE
    text = template.format(code=raw_code if raw else represented_code, requirement=requirement,
                           cases=render_cases(cases), instruction=DIRECT_TASK if direct else SER_TASK)
    return [{"role": "system", "content": SER_SYSTEM}, {"role": "user", "content": text}], 16 if direct else 768, "C" if direct else "F"


class Experiment:
    def __init__(self, repository, artifact_dir, work, client, tokenizer=None):
        self.root = Path(repository)
        self.work = Path(work)
        self.client = client
        self.artifacts = Artifacts(repository, artifact_dir)
        self.gate = EvidenceGate.load(self.root / "gate.json")
        self.cases = {r["case_id"]: r for r in read_jsonl(self.root / "cases.jsonl")}
        if any(r["project"] not in SOURCES for r in self.cases.values()):
            raise ValueError("Cases must belong to the four external projects")
        retrieval = self.root / "retrieval.jsonl"
        self.hits = {r["pair_id"]: r["cases"] for r in read_jsonl(retrieval)} if retrieval.exists() else {}
        summaries = self.root / "summaries.jsonl"
        self.summaries = {r["source_sha256"]: r["summary"] for r in read_jsonl(summaries)} if summaries.exists() else {}
        self.tokenizer = tokenizer

    def dense_hits(self, pair, requirement, code):
        if pair["pair_id"] not in self.hits:
            if self.tokenizer is None:
                raise ValueError("Supply saved retrieval hits or --tokenizer and a case index")
            import numpy as np
            from .retrieval import retrieve
            index = json.loads((self.root / "case_index.json").read_text())
            matrix = np.load(self.root / "case_vectors.npy", mmap_mode="r")
            hits = retrieve(requirement, code, index["case_ids"], matrix, self.client, self.tokenizer)
            self.hits[pair["pair_id"]] = hits
        return self.hits[pair["pair_id"]]

    def summary(self, artifact_id):
        text = self.artifacts.text(artifact_id)
        digest = self.artifacts.entries[artifact_id]["sha256"]
        if digest in self.summaries:
            return self.summaries[digest]
        path = self.work / "summaries" / (digest + ".json")
        if path.exists():
            data = json.loads(path.read_text())
            if data["source_sha256"] != digest:
                raise ValueError("Summary source checksum differs")
            summary = data["summary"]
        else:
            messages = [{"role": "system", "content": SUMMARY_SYSTEM},
                        {"role": "user", "content": SUMMARY_TEMPLATE.format(code=text)}]
            summary, usage = self.client.chat(messages, 1024)
            write_json(path, {"source_sha256": digest, "summary": summary, "usage": usage})
        self.summaries[digest] = summary
        return summary

    def run_pair(self, pair, arm):
        requirement = self.artifacts.text(pair["requirement_id"])
        code = self.artifacts.text(pair["code_id"])
        cases, audits = [], []
        if arm not in ("without_cases", "Direct_LLM", "LiSSA"):
            for hit in self.dense_hits(pair, requirement, code):
                case = self.cases[hit["case_id"]]
                if case["project"] == pair["dataset"] or hit["score"] <= .5:
                    raise ValueError("Retrieved case violates source separation or score cutoff")
                r = self.artifacts.text(case["requirement_id"])
                c = self.artifacts.text(case["code_id"])
                gate = self.gate.evaluate(requirement, code, r, c)
                audits.append({**hit, **gate})
                if arm == "ungated" or gate["accepted"]:
                    cases.append({"case_id": case["case_id"], "requirement": r,
                                  "code": self.summary(case["code_id"]), "label": case["label"]})
        represented = ""
        if arm not in ("raw_source", "Direct_LLM", "LiSSA"):
            represented, _ = augmented_code(requirement, code, self.summary(pair["code_id"]))
        messages, limit, parser = messages_for(arm, requirement, code, represented, cases)
        content, usage = self.client.chat(messages, limit)
        prediction = parse_prediction(parser, content)
        return {"pair_id": pair["pair_id"], "arm": arm, "prediction": prediction,
                "response": content, "usage": usage, "case_count": len(cases), "gate": audits,
                "requirement_sha256": sha(requirement.encode()), "code_sha256": sha(code.encode())}

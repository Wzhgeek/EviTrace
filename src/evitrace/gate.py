"""Source-only calibration and bilateral admission of retrieved cases."""
from collections import Counter
import json
import math
import random
from pathlib import Path
from .io import SOURCES
from .tokenization import _terms, _vector, _cosine, _quantile


class EvidenceGate:
    def __init__(self, sides):
        self.sides = sides
        self.cache = {}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["sides"])

    def vector(self, text, side):
        key = (side, text)
        if key not in self.cache:
            if len(self.cache) >= 4096:
                self.cache.pop(next(iter(self.cache)))
            config = self.sides[side]
            self.cache[key] = _vector(_terms(text, side), config["idf"], config["unseen_idf"])
        return self.cache[key]

    def evaluate(self, requirement, code, case_requirement, case_code):
        vectors = [self.vector(requirement, "requirement"), self.vector(code, "code"),
                   self.vector(case_requirement, "requirement"), self.vector(case_code, "code")]
        r = _cosine(vectors[0], vectors[2])
        c = _cosine(vectors[1], vectors[3])
        accepted = all(v[1] > 0 for v in vectors) and r >= self.sides["requirement"]["threshold"] and c >= self.sides["code"]["threshold"]
        return {"accepted": bool(accepted), "requirement_score": r, "code_score": c}


def calibrate(artifacts, seed=42, samples=2000, quantile=.75):
    """Read each distinct external artifact once; link labels are not inputs."""
    rng = random.Random(seed)
    sides = {}
    for side in ("requirement", "code"):
        grouped = {p: [] for p in SOURCES}
        entries = [r for r in artifacts.entries.values() if r["project"] in SOURCES and r["side"] == side]
        for row in sorted(entries, key=lambda r: (r["project"], r["name"])):
            terms = _terms(artifacts.text(row["artifact_id"]), side)
            if terms:
                grouped[row["project"]].append(terms)
        if any(not grouped[p] for p in SOURCES):
            raise ValueError("Calibration requires nonempty artifacts from all four source projects")
        documents = [terms for p in SOURCES for terms in grouped[p]]
        frequencies = Counter(t for terms in documents for t in terms)
        count = len(documents)
        idf = {t: 1 + math.log((count + 1) / (df + 1)) for t, df in sorted(frequencies.items())}
        unseen = 1 + math.log(count + 1)
        vectors = {p: [_vector(t, idf, unseen) for t in grouped[p]] for p in SOURCES}
        scores = []
        for _ in range(samples):
            a, b = rng.sample(list(SOURCES), 2)
            scores.append(_cosine(rng.choice(vectors[a]), rng.choice(vectors[b])))
        sides[side] = {"idf": idf, "unseen_idf": unseen, "nonempty_documents": count,
                       "threshold": max(1e-6, _quantile(scores, quantile)), "reference_scores": scores}
    return {"calibration": {"seed": seed, "samples_per_side": samples, "quantile": quantile}, "sides": sides}

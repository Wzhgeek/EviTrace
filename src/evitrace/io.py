"""Portable experimental inputs and verified artifact access."""
from pathlib import Path
import hashlib
import json

PROJECTS = ("eTour", "eAnci", "iTrust", "SMOS", "LibEST", "Albergate", "Dronology", "Maven")
SOURCES = ("Derby", "Drools", "Pig", "Seam2")
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path.cwd()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize(raw):
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError:
        value = raw.decode("iso-8859-1")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


class Artifacts:
    def __init__(self, repository, directory):
        self.directory = Path(directory).resolve()
        rows = read_jsonl(Path(repository) / "artifacts.jsonl")
        self.entries = {row["artifact_id"]: row for row in rows}
        self.cache = {}

    def text(self, artifact_id):
        if artifact_id not in self.cache:
            row = self.entries[artifact_id]
            path = (self.directory / row["relative_path"]).resolve()
            if self.directory not in path.parents:
                raise ValueError("Artifact path escapes the input directory")
            raw = path.read_bytes()
            if sha(raw) != row["sha256"]:
                raise ValueError(f"Artifact checksum differs: {artifact_id}")
            self.cache[artifact_id] = normalize(raw)
        return self.cache[artifact_id]

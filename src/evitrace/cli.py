"""Command-line entry points for the final experimental protocol."""
import argparse
import json
import os
from pathlib import Path


def main():
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    if hasattr(os, "nice"):
        os.nice(10)
    from .io import Artifacts, read_jsonl, write_json
    parser = argparse.ArgumentParser(description="EviTrace final experimental pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--evaluation-pairs", type=Path, required=True)
    prepare.add_argument("--case-pairs", type=Path, required=True)
    prepare.add_argument("--artifact-root", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    gate = commands.add_parser("calibrate")
    gate.add_argument("--inputs", type=Path, required=True)
    gate.add_argument("--artifact-root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--work", type=Path, default=Path("work"))
    run.add_argument("--config", type=Path, default=Path("config/final.json"))
    run.add_argument("--arms", nargs="+", default=["EviTrace"])
    run.add_argument("--project", nargs="+")
    run.add_argument("--limit", type=int)
    run.add_argument("--tokenizer", type=Path)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--budget-cny", type=float)
    index = commands.add_parser("index")
    index.add_argument("--inputs", type=Path, required=True)
    index.add_argument("--artifact-root", type=Path, required=True)
    index.add_argument("--tokenizer", type=Path, required=True)
    index.add_argument("--work", type=Path, default=Path("work"))
    index.add_argument("--config", type=Path, default=Path("config/final.json"))
    index.add_argument("--execute", action="store_true")
    index.add_argument("--budget-cny", type=float)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--pairs", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--scores", type=Path)
    evaluate.add_argument("--bootstrap", type=int, default=2000)
    evaluate.add_argument("--out", type=Path, required=True)
    ir = commands.add_parser("ir")
    ir.add_argument("--inputs", type=Path, required=True)
    ir.add_argument("--artifact-root", type=Path, required=True)
    ir.add_argument("--out", type=Path, required=True)
    ir.add_argument("--bm25-reference-pairs", type=Path)
    ftlr = commands.add_parser("ftlr")
    ftlr.add_argument("--checkout", type=Path, required=True)
    ftlr.add_argument("--manifest", type=Path, required=True)
    ftlr.add_argument("--pairs", type=Path, required=True)
    ftlr.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        print(json.dumps(prepare(args.evaluation_pairs, args.case_pairs, args.artifact_root, args.out)))
    elif args.command == "calibrate":
        from .gate import calibrate
        value = calibrate(Artifacts(args.inputs, args.artifact_root))
        path = args.inputs / "gate.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("Existing calibration differs; use a separate input directory")
        write_json(path, value)
        print(json.dumps({k: v["threshold"] for k, v in value["sides"].items()}))
    elif args.command == "evaluate":
        from .evaluation import evaluate
        print(json.dumps(evaluate(args.pairs, args.predictions, args.out, args.scores, args.bootstrap)))
    elif args.command == "ftlr":
        from .ftlr import evaluate_ftlr
        print(json.dumps(evaluate_ftlr(args.checkout, args.manifest, args.pairs, args.out)))
    elif args.command == "ir":
        from .ir import run_ir
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            print(json.dumps(run_ir(args.inputs, args.artifact_root, args.out, args.bm25_reference_pairs)))
    else:
        if args.command == "run":
            from .pipeline import ARMS
            if any(arm not in ARMS for arm in args.arms):
                raise ValueError(f"Available arms: {ARMS}")
            pairs = read_jsonl(args.inputs / "pairs.jsonl")
            if args.project:
                pairs = [p for p in pairs if p["dataset"] in args.project]
            if args.limit is not None:
                pairs = pairs[:args.limit]
            if not args.execute:
                print(json.dumps({"candidate_pairs": len(pairs), "arms": args.arms, "network_enabled": False}))
                return
        elif not args.execute:
            print(json.dumps({"cases": len(read_jsonl(args.inputs / "cases.jsonl")), "network_enabled": False}))
            return
        from .client import APIClient, BudgetExceeded
        client = APIClient(args.work / "requests.sqlite3", args.budget_cny or 0, json.loads(args.config.read_text()))
        tokenizer = None
        if args.tokenizer:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(args.tokenizer))
        if args.command == "index":
            import numpy as np
            from .retrieval import embed_query
            artifacts = Artifacts(args.inputs, args.artifact_root)
            cases = sorted(read_jsonl(args.inputs / "cases.jsonl"), key=lambda r: r["case_id"])
            vectors = [embed_query((artifacts.text(r["requirement_id"]) + "\n" + artifacts.text(r["code_id"])).strip(), client, tokenizer) for r in cases]
            np.save(args.inputs / "case_vectors.npy", np.asarray(vectors, dtype=np.float64))
            write_json(args.inputs / "case_index.json", {"case_ids": [r["case_id"] for r in cases], "dimensions": 1024})
        else:
            from .pipeline import Experiment
            experiment = Experiment(args.inputs, args.artifact_root, args.work, client, tokenizer)
            args.work.mkdir(parents=True, exist_ok=True)
            output = args.work / "predictions.jsonl"
            previous = read_jsonl(output) if output.exists() else []
            done = {(r["pair_id"], r["arm"]) for r in previous if r.get("status") == "ok"}
            failed = {(r["pair_id"], r["arm"]) for r in previous if r.get("status") != "ok"}
            with output.open("a", encoding="utf-8") as stream:
                for pair in pairs:
                    for arm in args.arms:
                        identity = (pair["pair_id"], arm)
                        if identity in done:
                            continue
                        if identity in failed:
                            raise ValueError("A prior failed prediction requires inspection in a separate run directory")
                        try:
                            record = {**experiment.run_pair(pair, arm), "status": "ok"}
                        except BudgetExceeded:
                            print("Budget limit reached; completed predictions were retained.")
                            return
                        except Exception as exc:
                            record = {"pair_id": pair["pair_id"], "arm": arm, "status": "error", "error_type": type(exc).__name__}
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        if record["status"] != "ok":
                            raise RuntimeError("Prediction failed; inspect the local record before continuing")


if __name__ == "__main__":
    main()

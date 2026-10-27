# EviTrace

Code for **EviTrace: Evidence-Guided Requirements-to-Code Link Recovery for Collaborative Software Maintenance**.

[Supplementary material (PDF)](EviTrace_Supplementary_Material.pdf)

EviTrace combines reusable code summaries, original source evidence, bilateral admission of external cases, and structured evidential reasoning. The final evaluation covers eTour, eAnci, iTrust, SMOS, LibEST, Albergate, Dronology, and Maven.

## Installation

Python 3.12 is recommended. CPU execution uses one numerical-library thread. Model inference is performed through the configured API.

```bash
python -m pip install -e '.[retrieval,ftlr]'
python -m nltk.downloader punkt_tab stopwords
```

## Inputs

Original benchmark artifacts are obtained from their public replication packages. They are supplied locally and are not included in this repository.

Prepare two JSONL lists. Evaluation rows contain `dataset`, `pair_id`, `requirement_path`, `code_path`, and binary `label`. External-case rows contain `source_project`, `case_id`, the two artifact paths, and binary `label`. Paths are relative to a local artifact directory. Candidate identities and the four development-requirement exclusions must be retained when reproducing the paper. The evaluated set contains 5,782 pairs from 448 requirements. The case library contains 7,980 pairs from Derby, Drools, Pig, and Seam2.

```bash
evitrace prepare --evaluation-pairs evaluation.jsonl --case-pairs cases.jsonl \
  --artifact-root artifacts --out work/inputs
evitrace calibrate --inputs work/inputs --artifact-root artifacts
```

Calibration reads only external artifact text. The fixed quantile is 0.75, with 2,000 reference pairs per side and seed 42. The paper thresholds are recorded in `config/final.json`.

## Run

API credentials are read from `DEEPSEEK_API_KEY` and, when embedding is needed, `DASHSCOPE_API_KEY`. No credentials are bundled. Set environment variables in the shell. Check provider prices and the service model before enabling paid execution.

Saved retrieval hits and summaries may be supplied as `retrieval.jsonl` and `summaries.jsonl` inside the prepared input directory. Retrieval rows contain `pair_id` and `cases`, a list of `case_id`/`score` objects. Summary rows contain the original artifact `source_sha256` and `summary`. This preserves the paper's archived retrieval inputs. The supplementary material specifies their embedding coverage.

To construct a new full-text case index, supply the Qwen3-Embedding-0.6B tokenizer and explicitly authorize a request budget:

```bash
evitrace index --inputs work/inputs --artifact-root artifacts \
  --tokenizer tokenizer.json --execute --budget-cny 5
```

The default run only reports its scope. Network requests require both `--execute` and a positive `--budget-cny`. Completed payloads are cached. Failed or incomplete responses are retained as failures.

```bash
evitrace run --inputs work/inputs --artifact-root artifacts --arms EviTrace
evitrace run --inputs work/inputs --artifact-root artifacts \
  --arms EviTrace without_cases without_SER raw_source ungated Direct_LLM LiSSA \
  --execute --budget-cny 5
```

Supply `--tokenizer tokenizer.json` when retrieval hits need to be computed. All seven configurations use the final shared model settings. `without_SER` changes only the requested evidence output. `raw_source` replaces the combined summary and source-evidence representation.

## Evaluation and baselines

```bash
evitrace ir --inputs work/inputs --artifact-root artifacts --out work/ir \
  --bm25-reference-pairs original_candidates.jsonl
evitrace evaluate --pairs work/inputs/pairs.jsonl \
  --predictions work/predictions.jsonl --scores work/ir/ranking_scores.jsonl \
  --out work/evaluation
```

For the archived BM25 classification scope, `original_candidates.jsonl` contains the original 5,950 candidate rows before development exclusions, in the same format as the evaluation input. VSM, LSI, LDA, fixed-candidate ranking, and the candidate classifiers retain the evaluated scope.

Classification is averaged equally across projects. Ranking is evaluated on requirements with positive candidates, with an additional zero-inclusive result. LLM reranking retains BM25 order within its yes/no groups. Paired intervals resample complete requirement groups.

The FTLR adapter invokes the maintained author implementation on supplied element-distance matrices. Its input manifest maps each project to `matrix`, `artifact_map`, `name_map`, and optional `callgraph` paths relative to the manifest. `name_map` maps implementation filenames to the benchmark filenames. Mixed Java/JSP and filename/classname coverage must be retained in these inputs. No neural baseline or GPU is loaded by this adapter.

```bash
evitrace ftlr --checkout external/finegrained-traceability \
  --manifest ftlr/manifest.json --pairs work/inputs/pairs.jsonl --out work/ftlr
```

FTLR uses majority threshold 0.59 and selects the best project F1 on the final-threshold grid described in the supplement. The released author code is available at [finegrained-traceability](https://github.com/tobhey/finegrained-traceability), commit `197cf2f395e9e90636436f30d6383465cb9e9f63`. Its upstream license applies to that code.

LiSSA's reasoning prompt follows its [official replication package](https://github.com/ArDoCo/Replication-Package-ICSE25_LiSSA-Toward-Generic-Traceability-Link-Recovery-through-RAG), commit `a8e652f29bbdcc4bdcd3d98cd85aa4a37cb65480`. The project-wide VSM, LSI, and LDA settings follow Wang et al., *Journal of King Saud University—Computer and Information Sciences*, 36(6), 102118, 2024. See the PDF for the final settings, prompts, results, and statistical procedure.

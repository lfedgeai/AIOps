# 🔬 Project Title: AutoRAG Evaluation for AIOps Troubleshooting Retrieval

> **Executive Summary:** An AutoML-driven evaluation harness that automatically compares retrieval and generation strategies (embedding models, top_k, retrieval methods, LLMs) to find the best-performing RAG configuration for answering AIOps troubleshooting questions.

---
For all your approved research tasks, please use the below template in the top-level directory of your research project.


## 📌 Research Overview
* **Status:** `🏗️ In-Progress`
* **Domain:** Retrieval-Augmented Generation (RAG) / AIOps
* **Core Question:** What retrieval configuration (method, top_k, embedding model) and generator setup produce the most accurate answers to AIOps troubleshooting questions?

---

## 🎯 Desired Outcomes
* **Primary Objective:** to prove out that the winning retrieval + generation configuration is best suited (highest retrieval F1 / semantic answer quality) for answering AIOps troubleshooting questions against our operational knowledge base
* **Key Deliverables:**
    - [ ] **Trained Weights:** N/A — this project evaluates retrieval configuration and off-the-shelf generator models rather than training new weights.
    - [X] **Cleaned Dataset:** QA pairs with ground-truth references (`data/aiops_qa_fixed.parquet`) and chunked corpus (`data/aiops_corpus.parquet`).
    - [ ] **Blog Post:** Formal documentation of findings.
    - [X] **Visual Dashboard:** AutoRAG-generated comparison summaries (`summary.csv` per node) and a demo handout.

---

## 🛠 Methodology & Framework
*Describe the technical approach and the "How" behind the research.*

I am using [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) (v0.3.25) to automatically build, run, and score candidate RAG pipeline configurations against a labeled AIOps QA dataset, comparing retrieval methods and parameters on standard IR metrics, and comparing generated answers on lexical and semantic similarity metrics.

* **Approach:** collect/curate a QA dataset with known-correct source chunks, define candidate pipeline configurations in a single YAML config, let AutoRAG run every candidate and score it against the same metrics, and use the resulting `summary.csv` files to select the best-performing configuration per pipeline stage.
* **Core Logic:** `config/aiops.yaml` (pipeline + candidate definitions), executed via the `autorag` CLI (`validate` for dry-run config checks, `evaluate` for full runs).
* **Tech Stack:** AutoRAG 0.3.25, ChromaDB (persistent vector store), HuggingFace embeddings (`BAAI/bge-small-en-v1.5`), Qwen2.5-0.5B-Instruct (local, CPU, via `llama_index`), Python (pandas), Parquet.

---

## 📊 Data Management & Transparency
1.  **Source:** AIOps operational knowledge (troubleshooting content) chunked into `data/aiops_corpus.parquet`, paired with question/answer/ground-truth-chunk records in `data/aiops_qa_fixed.parquet`.
2.  **Processing:** `create_dataset.py` builds the initial corpus and QA pairs; `fix_qa.py` repairs/cleans the QA dataset schema (source of the `_fixed` suffix); `ingest_chroma.py` embeds corpus chunks into the persistent Chroma vector store (`resources/chroma/`).
3.  **Ethical Considerations:** dataset is derived from internal/synthetic AIOps operational content for evaluation purposes only; no external personal or copyrighted data is used.

---

## 📂 Repository Structure
```text
├── config/                 # AutoRAG pipeline configs (config/aiops.yaml)
├── data/                   # QA dataset + corpus (Parquet)
├── resources/chroma/       # Persistent Chroma vector store (embedded corpus)
├── evaluator/              # Evaluation helper scripts
├── results/                # Saved comparison outputs
├── project/, project2/     # AutoRAG run outputs (per-trial summaries, node outputs)
├── create_dataset.py       # Builds corpus + QA pairs
├── fix_qa.py                # Cleans/repairs the QA dataset
├── ingest_chroma.py         # Embeds corpus into Chroma
└── README.md                # This file
```


# Project Details
## AutoRAG AIOps Retrieval Evaluation

A configuration-driven harness to compare retrieval strategies and generation setups for an AIOps troubleshooting assistant, using AutoRAG's automatic pipeline search and scoring.

- Compare retrieval strategies (semantic / lexical / hybrid) and parameters (e.g. `top_k`)
- Compare prompt formatting and generator (LLM) configurations
- Score every candidate on standard IR and text-generation metrics
- Automatically select and carry forward the best-performing candidate at each pipeline stage

### Repository layout

- `config/aiops.yaml` — defines the vector store, pipeline stages (nodes), and every candidate module/parameter AutoRAG should try
- `create_dataset.py` — builds the QA dataset and/or corpus chunks
- `fix_qa.py` — repairs/cleans QA dataset schema issues
- `ingest_chroma.py` — embeds corpus chunks into the persistent Chroma vector store
- `data/` — curated dataset files:
  - `aiops_corpus.parquet` — chunked knowledge base (one row per chunk)
  - `aiops_qa_fixed.parquet` — question/answer pairs with ground-truth reference chunks
- `resources/chroma/` — persistent embedded vector store used by the `vectordb` retrieval module
- `project/`, `project2/` — AutoRAG output directories (one subfolder per trial run)

## Data model and how it fits together

There are two data assets that AutoRAG requires:

- **Corpus** (`aiops_corpus.parquet`): the knowledge base, already split into chunks. Each row is one retrievable unit of text embedded into the vector store.
- **QA dataset** (`aiops_qa_fixed.parquet`): the ground truth used for scoring. Each row has:
  - `qid` / question text
  - reference answer(s)
  - `retrieval_gt` — the chunk(s) from the corpus that should be retrieved to correctly answer the question

Retrieval and generation are scored differently:
- **Retrieval** is scored against `retrieval_gt` using metrics like F1, recall, NDCG, and MRR — did the pipeline find (and rank highly) the correct chunk(s)?
- **Generation** is scored against the reference answer using metrics like METEOR, ROUGE (lexical overlap), and `sem_score` (embedding-based semantic similarity) — did the generated answer match in wording and/or meaning?

## Prerequisites

- Python 3.12
- AutoRAG 0.3.25 (`pip install AutoRAG`)
- ChromaDB, `llama-index`, `llama-index-embeddings-huggingface`, `llama-index-llms-huggingface`


Install Python deps:

```bash
pip install -U pip setuptools wheel
pip install autorag chromadb llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface
```

## Quickstart

From this directory:

```bash
# 1) Build the corpus + initial QA data
python3  create_dataset.py

# 2) Clean/repair the QA schema
python3 fix_qa.py

# 3) Ingest the corpus into the Chroma vector store (only needed once, or after corpus changes)
python3 ingest_chroma.py

# 4) Validate the config on a small automatic sample (fast sanity check)
autorag validate \
  --qa_data_path data/aiops_qa_fixed.parquet \
  --corpus_data_path data/aiops_corpus.parquet \
  --config config/aiops.yaml

# 5) Run the full evaluation
autorag evaluate \
  --qa_data_path data/aiops_qa_fixed.parquet \
  --corpus_data_path data/aiops_corpus.parquet \
  --config config/aiops.yaml \
  --project_dir project
```

Outputs (per trial, numbered from `0`) go to `<project_dir>/0/`:
- `<node_line>/<node_type>/summary.csv` — every candidate tried at that stage, its scores, and `is_best`
- `<node_line>/<node_type>/*.parquet` — full retrieved/generated output per candidate

## From raw source content to QA + corpus (curation guide)

1) Author or collect source AIOps content (runbooks, troubleshooting notes, etc.).

2) Run `create_dataset.py` to chunk the source content into `aiops_corpus.parquet` and generate an initial QA set with reference answers and ground-truth chunk links.

3) Run `fix_qa.py` to repair/normalize the generated QA schema (e.g. malformed `retrieval_gt` fields), producing `aiops_qa_fixed.parquet`.

4) Run `ingest_chroma.py` to embed every corpus chunk and populate `resources/chroma/` — this is what the `vectordb` module in `config/aiops.yaml` actually searches at query time.

5) Verify:
```bash
autorag validate \
  --qa_data_path data/aiops_qa_fixed.parquet \
  --corpus_data_path data/aiops_corpus.parquet \
  --config config/aiops.yaml
```

### ASCII diagrams

End-to-end data flow:

```
+----------------------------+
|  Source AIOps content      |
|  (runbooks, docs, etc.)    |
+-------------+--------------+
              |
              | create_dataset.py (chunk + generate QA)
              v
   data/aiops_corpus.parquet        data/aiops_qa.parquet (raw)
              |                                |
              |                                | fix_qa.py (repair/clean)
              |                                v
              |                     data/aiops_qa_fixed.parquet
              |
              | ingest_chroma.py (embed chunks)
              v
   resources/chroma/  (persistent vector store)

                    (both feed into)
                              |
                              v
                    config/aiops.yaml
              (pipeline stages + candidates)
                              |
                              v
                  autorag validate / evaluate
                              |
                              v
              project/<trial>/<node_line>/<node>/summary.csv
```

How pipeline stages are evaluated (node-by-node "bake-off"):

```
                         config/aiops.yaml node_lines
                                    |
        +---------------------------------------------------+
        |  For each node: try every module/param combination |
        |  (e.g. top_k=[3,5], modules=[vectordb, bm25, ...]) |
        +---------------------------------------------------+
                 |                                  |
                 v                                  v
      Score on listed metrics              Score on listed metrics
      (retrieval_f1, recall,                (meteor, rouge, sem_score)
       ndcg, mrr)                                    |
                 |                                  |
                 v                                  v
        Pick is_best=True candidate        Pick is_best=True candidate
        (primary metric = first listed)     (primary metric = first listed)
                 |                                  |
                 +----------- winner's output ------+
                              flows forward to
                              the next node
```

## Reports and how they're generated

There is one primary set of outputs, generated per pipeline run:

- **Retrieval node summaries** (`semantic_retrieval_node_line/semantic_retrieval/summary.csv`, `lexical_retrieval_node_line/lexical_retrieval/summary.csv`):
  - Inputs: corpus embedded in Chroma (`vectordb` module) and/or BM25 index built from the corpus (`bm25` module).
  - Steps:
    1) For each candidate module/parameter combination (e.g. `top_k=3` vs `top_k=5`), retrieve chunks for every QA question.
    2) Compare retrieved chunks against `retrieval_gt`.
    3) Compute retrieval_f1, retrieval_recall, retrieval_ndcg, retrieval_mrr.
  - Outputs: `summary.csv` (per-candidate scores + `is_best`), per-candidate `.parquet` files with full retrieved output.

- **Generation node summary** (`post_retrieve_node_line/generator/summary.csv`):
  - Inputs: prompts built by `prompt_maker` from the winning retrieval candidate's output.
  - Steps:
    1) Generate an answer per QA question using the configured LLM (Qwen2.5-0.5B-Instruct).
    2) Compare generated answers to reference answers.
    3) Compute meteor, rouge, and sem_score (embedding-based semantic similarity).
  - Output: `summary.csv` (per-candidate scores + `is_best`), plus generated answer text per row.

### Feature set (high level)
Chunk-level content used for retrieval (examples):
- Chunk text (source paragraph/section from AIOps troubleshooting content)
- Embedding vector (via `BAAI/bge-small-en-v1.5`, stored in Chroma)
- Metadata: source document, chunk index

## Retrieval & generation options included

- **Semantic retrieval — `vectordb` module (sklearn/Chroma-backed)**
  - Embeds the query and searches the Chroma vector store for nearest-neighbor chunks.
  - Configurable via `top_k` (number of chunks retrieved).
  - Strengths: captures semantic/paraphrastic similarity, not just keyword overlap.
  - Limitations: quality depends heavily on embedding model fit to domain vocabulary (e.g. AIOps-specific terminology).

- **Lexical retrieval — `bm25` module**
  - Classic term-frequency-based keyword search; no embeddings involved.
  - Strengths: fast, robust to rare/exact terms (error codes, service names) that embeddings may under-weight.
  - Limitations: cannot match paraphrased or semantically-equivalent but lexically different queries.

- **Hybrid retrieval — `hybrid_rrf` module** *(planned/optional)*
  - Fuses semantic and lexical rankings via Reciprocal Rank Fusion.
  - Consider when: neither semantic nor lexical retrieval alone is consistently best across all query types.

- **Generator — `llama_index_llm` module (Qwen2.5-0.5B-Instruct, local CPU inference)**
  - Produces the final natural-language answer from the retrieved context and formatted prompt.
  - Strengths: fully local, no external API dependency, fast enough for CPU-only iteration.
  - Limitations: small model size may under-perform larger LLMs on complex, multi-step troubleshooting reasoning — a candidate for future comparison.

All candidates:
- Are scored on the same held-out QA sample.
- Report metrics via AutoRAG's automatic `summary.csv` output.
- Have exactly one `is_best: True` row per node once compared.

### Configuration knobs
- `top_k` (per retrieval node): number of chunks retrieved per query — can be a list (e.g. `[3, 5]`) to trigger a comparison sweep.
- `embedding_model` (vectordb / sem_score metric): which registered AutoRAG embedding preset to use (e.g. `huggingface_baai_bge_small`).
- `strategy.metrics` (per node): which metrics to score candidates on; the **first metric listed** is the primary tiebreaker for `is_best`.

## Tuning and troubleshooting

- **Single-candidate nodes aren't real comparisons:** a node with only one module/parameter combination will always report `is_best: True` trivially — add multiple candidates (lists in YAML) to get a genuine comparison.
- **Small QA sample size:** results from a small QA dataset (e.g. 5 rows) should be treated as an early signal, not a production decision — grow the QA dataset before trusting a comparison result.
- **Node type mismatches:** `node_type` values must match this AutoRAG install's registered node names exactly (`semantic_retrieval`, `lexical_retrieval`, `hybrid_retrieval`, `prompt_maker`, `generator`, etc.) — check with:
  ```bash
  python3 -c "import inspect, autorag.support as s; print(inspect.getsource(s.get_support_nodes))"
  ```
- **Module keyword mismatches:** module/metric parameter names (e.g. `sem_score`'s `embedding_model`) must match the installed function signature exactly — check with:
  ```bash
  python3 -c "import inspect, autorag.evaluation.metric as m; print(inspect.signature(m.sem_score))"
  ```
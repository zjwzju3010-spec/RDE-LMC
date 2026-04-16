# RDE-LMC: Retrieval, Discrimination, and Execution for Legal Mathematical Calculation

A system for answering Chinese legal numerical calculation questions. Given a case description, the system retrieves relevant statutory articles, judges their applicability, symbolically executes the legal formula, and returns both the numerical answer and the cited articles.

## Pipeline Overview

```
Query
  ↓
[1] EntityBR  — BM25 + Dense (Qwen3-Embedding) → RRF fusion → Top-20 candidates
               + Law-name keyword direct lookup (bypasses tokenization for specialized laws)
  ↓
[2] LogicAD   — LLM applicability judgment (parallel) → conflict resolution
  ↓
[3] SymbolCE  — LLM generates IR → AST-safe expression execution
  ↓
[4] DynamicCM — Deficiency detection → query rewrite → up to 3 iterations
  ↓
[Fallback]    — Emergency BM25 top-50 + relaxed judgment → direct LLM answer
  ↓
[Citation]    — Post-execution co-citation completion (companion articles)
  ↓
{numerical_answer: [float], article_answer: [str]}
```

## Project Structure

```
ICAIL2026/
├── main.py                          # Entry point
├── config/
│   └── config.yaml                  # Model, paths, retrieval parameters
├── data/
│   ├── raw_laws/                    # 308 statutory .txt files (~21,813 articles)
│   ├── dataset/
│   │   └── dataset.jsonl            # 700 evaluation samples
│   ├── processed_kb/                # Built KB artifacts (corpus, embeddings, BM25 cache)
│   └── results.jsonl                # Evaluation output (JSONL, resume-safe)
├── src/
│   ├── kb/
│   │   ├── law_parser.py            # Parse .txt → article dicts (Format A/B auto-detect)
│   │   ├── llm_extractor.py         # LLM structure extraction with MD5 file cache
│   │   ├── build_kb.py              # Build corpus + FAISS + BM25 index
│   │   └── schema.py                # LegalArticle / LegalElement dataclasses
│   ├── retrieval/
│   │   ├── hybrid_retriever.py      # BM25 + Dense → RRF → law-name supplement
│   │   ├── law_name_retriever.py    # Keyword-to-article_id direct matching
│   │   ├── bm25_index.py            # BM25Okapi with jieba tokenization
│   │   ├── dense_index.py           # FAISS IndexFlatIP + Qwen3-Embedding
│   │   ├── query_normalizer.py      # Colloquial → formal legal vocabulary
│   │   └── keyword_extractor.py     # jieba TF-IDF keyword extraction
│   ├── discriminator/
│   │   ├── applicability_judge.py   # Parallel LLM judgment (standard + relaxed)
│   │   ├── conflict_resolver.py     # LLM-based conflict resolution (complementary set aware)
│   │   └── logic_prompt.py          # Prompt templates for judgment and resolution
│   ├── execution/
│   │   ├── ir_generator.py          # LLM → variable bindings + expression
│   │   ├── expression_tree.py       # AST-safe expression evaluator (whitelist, no eval())
│   │   ├── executor.py              # Execute IR list, deduplicate by value
│   │   └── citation_completer.py    # Post-execution co-citation via LLM
│   ├── dynamic/
│   │   ├── controller.py            # Main RDE loop + emergency/LLM fallbacks
│   │   ├── deficiency_detector.py   # Detect missing variables in IR
│   │   └── query_rewriter.py        # LLM query rewrite for next iteration
│   ├── evaluation/
│   │   ├── evaluator.py             # Run evaluation with resume support
│   │   └── metrics.py               # Numerical accuracy + article F1
│   └── pipeline/
│       └── rde_pipeline.py          # RDEPipeline wrapper class
└── API.py                           # LLM API client (OpenAI-compatible)
```

## Requirements

```bash
conda create -n env_zjw python=3.10
conda activate env_zjw
pip install faiss-cpu rank-bm25 jieba pyyaml tqdm transformers torch numpy
```

## Configuration

Edit `config/config.yaml`:

```yaml
model:
  llm: "DeepSeek-V3.2"           # LLM for judgment/execution (via API.py)
  embedding: "/path/to/Qwen3-Embedding-0.6B"
  embedding_dim: 1024

retrieval:
  bm25_top_k: 30
  dense_top_k: 30
  hybrid_top_k: 20               # Articles passed to discriminator

discriminator:
  max_applicable: 5              # Max articles sent to executor

execution:
  max_loop_iterations: 3
```

The LLM endpoint is configured in `API.py` (base URL + API key).

## Usage

### Step 1 — Build Knowledge Base

Parses all `.txt` law files and builds BM25 + FAISS indexes. Run once; takes ~10–30 min depending on hardware.

```bash
conda run -n env_zjw python main.py --mode build_kb
```

Outputs to `data/processed_kb/`: `corpus.json`, `embeddings.npy`, `article_ids.json`, `bm25_cache.pkl`.

> LLM structure extraction is **lazy**: articles are extracted on first retrieval and cached in `data/processed_kb/llm_cache/`. Only articles actually used in the pipeline are processed.

### Step 2 — Single Query

```bash
conda run -n env_zjw python main.py --mode single \
  --query "某企业进行营利性治沙活动，造成1公顷土地沙化加重，该企业最高应支付多少罚款？"
```

Expected output:
```json
{
  "numerical_answer": [50000.0],
  "article_answer": ["《中华人民共和国防沙治沙法》第四十条"]
}
```

### Step 3 — Evaluate

```bash
# Full evaluation (700 samples), resumes from existing results
conda run -n env_zjw python main.py --mode evaluate

# First N samples only
conda run -n env_zjw python main.py --mode evaluate --limit 100

# Specific sample IDs
conda run -n env_zjw python main.py --mode evaluate --ids 1,5,42,100

# Force restart (ignore existing results)
conda run -n env_zjw python main.py --mode evaluate --no-resume

# Background with log
nohup conda run -n env_zjw python main.py --mode evaluate > data/eval.log 2>&1 &
```

Results are written incrementally to `data/results.jsonl` (one JSON record per line). Evaluation can be safely interrupted and resumed.

### Output Format

Each record in `results.jsonl`:
```json
{
  "id": 0,
  "query": "...",
  "predicted_numerical": [50000.0],
  "predicted_articles": ["《中华人民共和国防沙治沙法》第四十条"],
  "true_numerical": [50000.0],
  "true_articles": ["《中华人民共和国防沙治沙法》第四十条"],
  "success": true
}
```

## Dataset Format

`data/dataset/dataset.jsonl` — 700 samples:

```json
{"id": 0, "query": "...", "numerical_answer": [50000.0], "article_answer": ["《...》第X条"]}
```

Article answer distribution: 68% single article, 20% two articles, 11% three articles, 1% four articles.

## Evaluation Metrics

| Metric | Description |
|---|---|
| `numerical_accuracy` | Prediction within 1% tolerance of ground truth |
| `article_f1` | Token-level F1 between predicted and true article sets |
| `success_rate` | Fraction of queries returning a non-empty numerical answer |

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Lazy LLM extraction with MD5 cache | 21,813 articles total; only ~500 are ever retrieved across 700 queries |
| AST-safe expression evaluator | Whitelist-only AST node traversal; never uses `eval()` |
| Law-name direct lookup | Specialized tax laws (e.g., 烟叶税法) have poor BM25 recall due to rare terminology |
| LLM-only conflict resolution | Rule-based specificity ranking caused wrong primary selections |
| Citation completion (post-execution) | Definition/scope articles filtered by "calculable" criterion; added back via conservative LLM check |
| Arabic→Chinese numeral normalization | LLM fallback generates `第4条`; normalized to `第四条` to match KB article IDs |
| Emergency fallback chain | Guarantees ~100% success rate: regular loop → BM25-50 + relaxed judge → direct LLM |

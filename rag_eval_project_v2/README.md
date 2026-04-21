# RAG Evaluation Framework v2 (Codex Build)

This project implements the PDF blueprint with:

- Stratified `35/15` train/test split
- Four pipelines:
  - `rag`
  - `no_rag`
  - `rag_pretrained`
  - `rag_pretrained_web`
- HyDE query expansion
- Web crawling via Tavily + Firecrawl
- Document validation (3-signal scoring + proof logging)
- Metric-set-aware evaluation
- Local validation cross-check for judge scores
- Per-run Excel reports + global comparison report
- Reviewer-friendly Streamlit UI (`ui/reviewer_app.py`)

## Setup (one-time)

```powershell
cd rag_eval_project_v2
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -r requirements.txt
ollama pull qwen2.5:14b
ollama pull kimi-k2.5:cloud
ollama pull nomic-embed-text
.\.venv\Scripts\python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', local_files_only=False); print('CE_OK')"
```

Create/update local env file (already scaffolded):

```powershell
notepad .env
```

Minimum `.env` values for full 4-pipeline run:

```env
TAVILY_API_KEY=YOUR_TAVILY_KEY
FIRECRAWL_API_KEY=YOUR_FIRECRAWL_KEY
# Optional unless using OpenRouter provider directly:
# OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
```

Start Ollama server before running:

```powershell
$env:OLLAMA_HOST="127.0.0.1:11435"
ollama serve
```

## Run project (common commands)

Show all CLI options:

```powershell
.\.venv\Scripts\python main.py --help
```

Quick smoke run (first test question, all 4 pipelines with Qwen):

```powershell
.\.venv\Scripts\python main.py --limit 1 --models qwen --pipelines rag no_rag rag_pretrained rag_pretrained_web --retrieval-backend llamaindex
```

Standard run (full test split, all 4 pipelines with Qwen):

```powershell
.\.venv\Scripts\python main.py --models qwen --pipelines rag no_rag rag_pretrained rag_pretrained_web --retrieval-backend llamaindex
```

Run train split, test split, or both:

```powershell
.\.venv\Scripts\python main.py --eval-split train --models qwen --pipelines rag_pretrained_web --retrieval-backend llamaindex
.\.venv\Scripts\python main.py --eval-split test --models qwen --pipelines rag_pretrained_web --retrieval-backend llamaindex
.\.venv\Scripts\python main.py --eval-split both --models qwen --pipelines rag_pretrained_web --retrieval-backend llamaindex
```

Use Kimi Cloud judge via Ollama cloud model:

```powershell
.\.venv\Scripts\python main.py --use-kimi-cloud-judge --models qwen --pipelines rag_pretrained_web --retrieval-backend llamaindex --limit 1
```

Override judges explicitly (evaluation + web validator):

```powershell
.\.venv\Scripts\python main.py --judge-provider openrouter --judge-model kimi-k2.5:cloud --web-judge-provider openrouter --web-judge-model kimi-k2.5:cloud --models qwen --pipelines rag_pretrained_web --retrieval-backend llamaindex --limit 1
```

Force-enable cross-encoder reranker (already enabled by default in `config.yaml`):

```powershell
.\.venv\Scripts\python main.py --models qwen --pipelines rag rag_pretrained --retrieval-backend llamaindex --enable-reranker --limit 1
```

Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Run train tuning:

```powershell
.\.venv\Scripts\python scripts\tune_on_train.py
```

Run train tuning with explicit profiles:

```powershell
.\.venv\Scripts\python scripts\tune_on_train.py --profile quick
.\.venv\Scripts\python scripts\tune_on_train.py --profile full
.\.venv\Scripts\python scripts\tune_on_train.py --profile quick --max-train-questions 20
```

Tuning artifacts:

- `outputs/progress/tuning_results.json` (candidate ranking + best objective)
- `config/best_config.yaml` (auto-merged override loaded on next runs)

Manual two-step targeted retrieval fix pass:

```powershell
.\.venv\Scripts\python scripts\tune_on_train.py --profile quick
.\.venv\Scripts\python main.py --eval-split both --models qwen --pipelines rag no_rag rag_pretrained rag_pretrained_web --retrieval-backend llamaindex --enable-reranker --enable-web-reranker
```

## Optional environment variables

- `OPENROUTER_API_KEY` (only if you explicitly choose OpenRouter provider)
- `GEMINI_API_KEY` (for Gemini generation pipeline)
- `TAVILY_API_KEY` (required for `rag_pretrained_web`)
- `FIRECRAWL_API_KEY` (required for `rag_pretrained_web`)
- `OLLAMA_BASE_URL` (preferred full URL, e.g. `http://127.0.0.1:11435`)
- `OLLAMA_HOST` (host:port form also supported, e.g. `127.0.0.1:11435`)

`runtime.strict_mode: true` is enabled by default. Missing models/services now fail fast.
Default generation model is local Ollama `qwen2.5:14b`.
Default evaluator/web judge is local Ollama `kimi-k2.5:cloud`.
`main.py` and `scripts/run_everything.ps1` both auto-load keys from `.env`.
You can switch judges at runtime with flags:
- `--use-kimi-cloud-judge`
- `--judge-provider`, `--judge-model`
- `--web-judge-provider`, `--web-judge-model`

## Embedding model profiles

`config/config.yaml` includes:

- `embeddings.model` (active)
- `embeddings.provider` (`ollama` or `sentence_transformers`)
- `embeddings.recommended_models.quality` (`nomic-embed-text`)
- `embeddings.recommended_models.balanced` (`sentence-transformers/all-mpnet-base-v2`)
- `embeddings.recommended_models.fast` (`sentence-transformers/all-MiniLM-L6-v2`)

Current default is `ollama + nomic-embed-text`.
Nomic task prefixes are enabled by default in config:
- document chunks use `search_document:`
- user queries use `search_query:`

## Installed package check

```powershell
.\.venv\Scripts\python -m pip show chromadb pyyaml rank-bm25 duckduckgo-search openpyxl
```

Check UI dependency:

```powershell
.\.venv\Scripts\python -m pip show streamlit
.\.venv\Scripts\python -m pip show dash dash-bootstrap-components plotly
```

## Run everything

One-command run with strict preflight (checks deps, server, models, keys; stops on failure):

```powershell
.\scripts\run_everything.ps1 -RetrievalBackend llamaindex
```

Cross-encoder cache warmup (one-time, required because KB/web rerankers are enabled by default):

```powershell
.\.venv\Scripts\python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', local_files_only=False); print('CE_OK')"
```

Run everything + tuning:

```powershell
.\scripts\run_everything.ps1 -Tune -RetrievalBackend llamaindex
```

Targeted retrieval fix pass (implemented workflow):

1) Tune retrieval + reranker parameters on train split  
2) Re-evaluate on full holdout coverage (train + test)

```powershell
.\scripts\run_everything.ps1 -Tune -EvalSplit both -RetrievalBackend llamaindex -EnableReranker -EnableWebReranker
```

Run everything with all configured models (`qwen gemma gemini`):

```powershell
.\scripts\run_everything.ps1 -AllModels -RetrievalBackend llamaindex
```

Run everything with a question limit (fast validation):

```powershell
.\scripts\run_everything.ps1 -Limit 1 -RetrievalBackend llamaindex
```

Run everything on train/test/both split:

```powershell
.\scripts\run_everything.ps1 -EvalSplit test -RetrievalBackend llamaindex
.\scripts\run_everything.ps1 -EvalSplit train -RetrievalBackend llamaindex
.\scripts\run_everything.ps1 -EvalSplit both -RetrievalBackend llamaindex
```

Run everything with Kimi Cloud judge:

```powershell
.\scripts\run_everything.ps1 -UseKimiCloudJudge -Limit 1 -RetrievalBackend llamaindex
```

Cloud login before Kimi cloud judge runs:

```powershell
ollama signin
ollama run kimi-k2.5:cloud "Reply only: KIMI_OK"
```

Set OpenRouter key only if you explicitly use OpenRouter provider:

```powershell
$env:OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

Run explicitly with cross-encoder rerankers enabled:

```powershell
.\scripts\run_everything.ps1 -RetrievalBackend llamaindex -EnableReranker -EnableWebReranker -Limit 1
```
Disable rerankers explicitly if you want pure retrieval scoring:

```powershell
.\scripts\run_everything.ps1 -RetrievalBackend llamaindex -DisableReranker -DisableWebReranker -Limit 1
```

Reset tuning overrides and go back to base `config.yaml` only:

```powershell
Remove-Item .\config\best_config.yaml -Force
```

## Reviewer UI

Run the default Dash reviewer UI:

```powershell
.\scripts\run_ui.ps1
```

Open UI for a specific progress file:

```powershell
.\scripts\run_ui.ps1 -ProgressFile outputs\progress\run_progress_YYYYMMDD_HHMMSS.json
```

Run on custom port:

```powershell
.\scripts\run_ui.ps1 -Port 8060
```

Optional: run the legacy Streamlit UI:

```powershell
.\scripts\run_ui.ps1 -Framework streamlit
```

Open the URL printed by Dash (usually `http://127.0.0.1:8050`).

Architecture/code-flow HTML (for business + reviewer walkthrough):

```powershell
start .\ui\code_flow_frameworks.html
```

## Validation visibility additions

Each report now includes:

- `Validation_Flags`
- `Validation_Reason`
- `Validation_Audit` sheet with flagged metrics
- `Web_Validation` sheet for per-document web validation signals
- `S3_Partial_Relevance` for edge-case web relevance visibility
- adjusted score tracking when a metric is flagged
- Interactive reviewer UI for run cards + question outcomes + web validation + proof table

Web proof log (append-only JSONL):

- `outputs/logs/web_validation_proof.jsonl`

## Tavily metadata policy filter

Registry-driven Tavily candidate filtering is enabled in `config/config.yaml` under:

- `crawler.metadata_filtering.enabled`
- `crawler.metadata_filtering.registry_path`
- `crawler.metadata_filtering.min_metadata_score`
- `crawler.metadata_filtering.min_credibility_score`
- `crawler.metadata_filtering.allow_decisions`
- `crawler.metadata_filtering.block_decisions`

Behavior:

- Filters Tavily URLs before Firecrawl scrape using `rag_data_management_registry.xlsx`
- Applies domain blocklist/allowlist and score thresholds
- Logs policy rejections into web validation proof flow

## How Train/Test Split Works

- Master file: `data/questions/questions_master.xlsx`
- Split config (current):
  - `train: 35`
  - `test: 15`
  - `strategy: stratified`
  - `stratify_column: Use Case Category`
- First split generation writes:
  - `data/questions/train_questions.xlsx`
  - `data/questions/test_questions.xlsx`
- Subsequent runs reuse those files for stable comparison.
- `--eval-split` controls what you evaluate:
  - `train`: tuning/diagnostics split
  - `test`: holdout split
  - `both`: runs both (recommended after tuning)
- `--limit` truncates selected split for quick checks only; it does not rewrite split files.

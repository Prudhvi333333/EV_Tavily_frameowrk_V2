# RAG Evaluation Framework v2 (Codex Build)

This project implements the PDF blueprint with:

- Stratified `35/15` train/test split
- Four pipelines:
  - `rag`
  - `no_rag`
  - `rag_pretrained`
  - `rag_pretrained_web`
- HyDE query expansion
- Web crawling with cache
- Metric-set-aware evaluation
- Local validation cross-check for judge scores
- Per-run Excel reports + global comparison report

## Setup (one-time)

```powershell
cd rag_eval_project_v2
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -r requirements.txt
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

## Run project (common commands)

Quick smoke run (first test question, all 4 pipelines with Qwen):

```powershell
.\.venv\Scripts\python main.py --limit 1 --models qwen --pipelines rag no_rag rag_pretrained rag_pretrained_web
```

Standard run (full test split, all 4 pipelines with Qwen):

```powershell
.\.venv\Scripts\python main.py --models qwen --pipelines rag no_rag rag_pretrained rag_pretrained_web
```

Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Run train tuning:

```powershell
.\.venv\Scripts\python scripts\tune_on_train.py
```

## Optional environment variables

- `OPENROUTER_API_KEY` (for Kimi K2 judge via OpenRouter)
- `GEMINI_API_KEY` (for Gemini generation pipeline)
- `TAVILY_API_KEY` (optional alternative search backend)
- `OLLAMA_BASE_URL` (defaults to `http://localhost:11434`)

`runtime.strict_mode: true` is enabled by default. Missing models/services now fail fast.
Default judge is local Ollama `qwen2.5:14b` (`evaluation.judge.provider: ollama`).

## Embedding model profiles

`config/config.yaml` includes:

- `embeddings.model` (active)
- `embeddings.provider` (`ollama` or `sentence_transformers`)
- `embeddings.recommended_models.quality` (`BAAI/bge-m3`)
- `embeddings.recommended_models.balanced` (`all-mpnet-base-v2`)
- `embeddings.recommended_models.fast` (`all-MiniLM-L6-v2`)

Current default is `ollama + nomic-embed-text`.

## Installed package check

```powershell
.\.venv\Scripts\python -m pip show chromadb pyyaml rank-bm25 duckduckgo-search openpyxl
```

## Run everything

One-command run (installs deps, ensures models, runs tests, runs pipelines):

```powershell
.\scripts\run_everything.ps1
```

Run everything + tuning:

```powershell
.\scripts\run_everything.ps1 -Tune
```

Run everything with all configured models (`qwen gemma gemini`):

```powershell
.\scripts\run_everything.ps1 -AllModels
```

Run everything with a question limit (fast validation):

```powershell
.\scripts\run_everything.ps1 -Limit 1
```

## Validation visibility additions

Each report now includes:

- `Validation_Flags`
- `Validation_Reason`
- `Validation_Audit` sheet with flagged metrics
- adjusted score tracking when a metric is flagged

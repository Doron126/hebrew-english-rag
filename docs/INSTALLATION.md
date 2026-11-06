# Installation and first use

This guide describes the available local setup on **Ubuntu 26.04 x86-64 in WSL**, using the Linux shell. The dependency lock targets CPython 3.14.4. It does not establish support for native Windows, other Linux releases, or macOS. The commands below are source-checked. A publication walkthrough indexed one synthetic text file (one chunk, no failures or skips) in isolated storage using an existing Python environment and a locally cached embedding model. An in-process API test returned 200 for liveness, the manually configured synthetic tenant catalog and its indexed document list; its source endpoint returned the exact synthetic original. This was a direct source-opening check, not a citation returned by a query. A separate isolated Uvicorn startup on an ephemeral loopback port returned 200 for `/livez` and `/api/tenants`; `/readyz` returned 503 with `model_unavailable` for the synthetic indexed tenant and absent isolated model socket. The process was then stopped. These checks reused an existing Python environment and cached embedding model; they did not verify a fresh dependency installation, successful model readiness, or an end-to-end model answer.

**Important limitation:** a fresh installation cannot currently start its generation model using a portable configuration. The included supervisor requires a specifically validated WSL/systemd, llama.cpp and Gemma profile, and its machine-specific paths have been replaced with invalid placeholders in this publication. Setting a URL or placing a model in a directory does not make the supervisor accept it. Operating it on a new machine requires substantial adaptation of the supervisor and its deployment, outside this reference publication. Do not disable its admission or authentication checks. Indexing and API liveness can be attempted independently; a first model-backed answer and full readiness are not established.

No original tenants, customer documents, or prebuilt indexes are included. Create your own tenants and ingest your own documents. The local interface does not create users, tenants, or uploads. Tenant creation for this local mode is an operator configuring a tenant ID and creating its own document directory; no account bootstrap is implemented in this mode.

## Requirements

- Ubuntu 26.04 x86-64 under WSL, Python 3.14.4, `python3.14-venv`, `pip`, Git, curl, enough disk space for the dependencies, downloaded models and your documents, and loopback networking. The locked Python dependencies include CPU PyTorch, FastAPI, Chroma, and sentence-transformers.
- Indexing and retrieval use the separately downloaded `intfloat/multilingual-e5-large-instruct` embedding model. An optional reranker is controlled by `USE_RERANKER`; this guide disables it. These models use system memory even if generation uses a GPU.
- Generation requires a compatible llama.cpp `llama-server` and Gemma 4 31B Q4_0 GGUF, separately obtained and installed. They are not included. The code selects the fixed model identifier `gemma-4-31b`; it does not connect directly to a user-supplied HTTP model server.
- PDF and image OCR optionally require Tesseract and the Hebrew and English language data. OCR and table extraction can fail or produce imperfect text. Install `tesseract-ocr`, `tesseract-ocr-heb`, and `tesseract-ocr-eng` if needed. Searchable PDFs do not necessarily need OCR.
- The historical WSL supervisor profile required 12 GiB of Windows physical memory available, 12 GiB of Windows commit headroom, 19 GiB of guest available memory, and 8 GiB of free GPU memory at admission. These are configurable profile checks, not universal hardware minimums; GPU memory and system RAM are different resources. The publication does not establish a new hardware recommendation.

For context, the developer's observed WSL guest exposed 12 logical CPUs on an Intel Core i5-11600, approximately 25.8 GiB of guest RAM, and an NVIDIA GeForce RTX 3060 with 12 GiB of GPU memory. This describes the observed machine, not a tested minimum or a recommendation.

The private GitHub repository requires an account with authorized access. Package installation, cloning, and the first model download need network access and may require upstream access or license acceptance. No credentials belong in this repository or these commands.

## Obtain code and Python dependencies

Run in the WSL Ubuntu shell, from a directory where you keep source checkouts:

```bash
git clone https://github.com/Doron126/hebrew-english-rag.git
cd hebrew-english-rag
sudo apt update
sudo apt install python3.14 python3.14-venv git curl
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

For OCR, additionally run `sudo apt install tesseract-ocr tesseract-ocr-heb tesseract-ocr-eng`. The repository lock has an extra index for the CPU PyTorch wheel. It does not install the separate native llama.cpp server. Installation from the lock requires access to package indexes. The commands are intended for this target, not a general cross-platform guarantee.

## Configure an isolated installation

Run from the checkout in the same WSL shell. Pick a data directory that belongs to you, and use it consistently for indexing and serving. The example tenant ID is an illustration; replace `my-tenant` with your own 1–64 ASCII letters, digits, hyphens or underscores, starting and ending with a letter or digit. This creates no tenant automatically on checkout.

```bash
export RAG_DATA="$HOME/hebrew-english-rag-data"
export TENANT_ID="my-tenant"
mkdir -p "$RAG_DATA/documents/$TENANT_ID" "$RAG_DATA/config/tenants" "$RAG_DATA/cache"
export APP_ENV=wsl
export APP_BIND_HOST=127.0.0.1
export TENANT_AUTH_MODE=client
export LOCAL_ALLOWED_TENANTS="$TENANT_ID"
export WARMUP_TENANTS=""
export PROJECTX_GEMMA_TENANTS="$TENANT_ID"
export DOCS_BASE_DIR="$RAG_DATA/documents"
export TENANT_CONFIG_DIR="$RAG_DATA/config/tenants"
export CHROMA_PATH="$RAG_DATA/chroma"
export INGEST_EXTRACTION_AUDIT_DIR="$RAG_DATA/extraction-audit"
export HF_HOME="$RAG_DATA/cache/huggingface"
export GEMMA_SUPERVISOR_SOCKET="$RAG_DATA/supervisor.sock"
export USE_RERANKER=0
export CHAT_USE_RERANKER=0
```

These process environment variables override the repository's optional `.env` and legacy `env` files (`config_env.py`). Repeat them in each shell used for the API or ingestion. `PROJECTX_GEMMA_TENANTS` is read by the separate supervisor process, so changing it in the API shell does not configure a different running supervisor. Do not point this example at another installation's storage or model socket.

| Setting | Purpose | Status and code default | Example / where set |
|---|---|---|---|
| `APP_ENV` | Select local or other authorization behavior | Optional; `dev` | `wsl`, shell |
| `APP_BIND_HOST` | Local-mode binding check | Optional; `127.0.0.1` | `127.0.0.1`, shell and matching Uvicorn argument |
| `TENANT_AUTH_MODE` | Tenant selection mode | Optional; `client` locally, `trusted_header` otherwise | `client`, shell |
| `LOCAL_ALLOWED_TENANTS` | Operator's local WSL tenant allow-list | Required for WSL local tenant selection; empty | `my-tenant`, shell |
| `WARMUP_TENANTS` | Tenants warmed at application startup | Optional; empty | Empty initially; shell |
| `PROJECTX_GEMMA_TENANTS` | Supervisor's generation tenant allow-list | Required by supervisor; empty fails authorization | `my-tenant`, supervisor environment |
| `TENANT_CONFIG_DIR` | Optional tenant profile directory | Optional; `config/tenants` | `$RAG_DATA/config/tenants`, shell |
| `DOCS_BASE_DIR` | Root of original documents, with one directory per tenant | Optional; `data` | `$RAG_DATA/documents`, shell |
| `CHROMA_PATH` | Persistent Chroma index | Optional; `storage/chroma` | `$RAG_DATA/chroma`, shell |
| `RAG_COLLECTION` | Base collection name | Optional; `company-knowledge` | `my-knowledge`, shell |
| `TENANT_STRATEGY` | Per-tenant collections or shared metadata | Optional; `collection` | `collection`, shell |
| `EMBED_MODEL` | Sentence-transformers embedding model | Optional; `intfloat/multilingual-e5-large-instruct` | Same name, shell |
| `USE_RERANKER` | Enable retrieval reranker | Optional; `1` in code | `0`, shell |
| `CHAT_USE_RERANKER` | Enable answer-stage reranking | Optional; `1` | `0`, shell |
| `INGEST_EXTRACTION_AUDIT_DIR` | Extraction audit output | Optional; `extraction-audit` beside the Chroma directory | `$RAG_DATA/extraction-audit`, shell |
| `PROJECT_X_CONVERSATION_STATE_PATH` | Persist backend conversation state as SQLite | Optional; unset means in-memory | `$RAG_DATA/conversations.sqlite3`, shell |
| `LLM_BACKEND` | Supported generation backend | Optional; `llama_cpp` | `llama_cpp`, shell |
| `LLAMA_MODEL_ID` | Expected generation model identifier | Optional; `gemma-4-31b` | `gemma-4-31b`, shell |
| `GEMMA_SUPERVISOR_SOCKET` | API client Unix socket for the supervisor | Optional; `run/hebrew-english-rag/supervisor.sock` relative to the process working directory | `$RAG_DATA/supervisor.sock`, shell |
| `HF_HOME` | Hugging Face downloaded-model cache | Optional; Hugging Face default | `$RAG_DATA/cache/huggingface`, shell |

The API listens on the host and port passed to Uvicorn below, not through an undocumented port variable. Its client uses `GEMMA_SUPERVISOR_SOCKET` to locate the supervisor Unix socket; the isolated path above prevents it from contacting another installation. The supervisor receives its listening socket from systemd through file descriptor 3. A matching systemd `.socket` unit must create that socket, but this publication does not include the required unit or deployment. Choosing the client socket therefore does not configure or start a listening supervisor. `LLAMA_BASE_URL` does not override the client socket. Client-selected local tenancy is restricted to loopback and does not provide account authentication. Trusted-header mode requires a separately configured trusted identity proxy; this guide does not supply one.

## Prepare models

The first real ingest downloads the [multilingual E5 embedding model](https://huggingface.co/intfloat/multilingual-e5-large-instruct) through sentence-transformers into the configured Hugging Face cache. Review its model card and license, and ensure network and any required access are available. The same model must be selected for ingestion and querying. Optional reranking may require its own model download.

The embedding download is selected by model name, without an immutable upstream revision in this publication. The Python dependency lock therefore does not establish exact model-weight reproducibility. The documented browser interface does not require Streamlit; the separately retained `webui_streamlit.py` is a legacy reference and its dependencies and operation are outside this installation path.

The generation implementation expects a separately run llama.cpp server owned by its supervisor and a Gemma 4 31B Q4_0 GGUF file. Consult [llama.cpp](https://github.com/ggml-org/llama.cpp) and the [Gemma model information](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf), including upstream access and license terms. The application does not distribute these assets. The model page does not establish that any given download matches the supervisor's exact expected bytes. The supervisor cannot accept a new model location or an already-running server through the configuration above; this is an unresolved setup limitation, not a complete model-start command.

## Create a tenant and index your documents

The directory and allow-list configured above define a user-chosen local tenant namespace. A profile file is optional; absent a display name the API displays the ID. There is no built-in account creation or UI file-upload flow. Copy your own small supported files to `"$RAG_DATA/documents/$TENANT_ID"` using your preferred file manager or shell, then run, from the checkout and activated virtual environment:

```bash
python ingest.py --tenant "$TENANT_ID" \
  --docs-dir "$RAG_DATA/documents/$TENANT_ID" \
  --continue-on-error --report-json "$RAG_DATA/ingest-report.json"
```

The ingestion command creates its Chroma collection on first use; no separate database migration command is required. It prints per-file processing and a final summary. Inspect the report for each document's `status`, errors, unchanged files and totals:

```bash
python -m json.tool "$RAG_DATA/ingest-report.json"
```

A partial import exits with code 2 when `--continue-on-error` encounters failures. Correct the reported file or dependency issue and rerun ingestion. By default unchanged indexed files can be skipped. `--sync` and `--clear` can remove obsolete index records after a successful scan and are not needed for the first import. Keep the originals in the tenant directory: source links validate and open those originals.

The default include list covers text and Markdown, PDF, DOCX, XLSX, PPTX, HTML, CSV/TSV, JSON and several source/log formats, plus PNG/JPEG/WebP/SVG and PostScript. Extraction and OCR are best-effort and bounded; scans, images, complex tables and formatting are not guaranteed to be understood. An unextractable file is reported rather than assumed to be indexed.

## Start and inspect the API

With the same environment exports and activated virtual environment, from the checkout:

```bash
python -m uvicorn chat_api:app --host 127.0.0.1 --port 18080
```

In another WSL shell, check liveness and visit the interface:

```bash
curl --fail http://127.0.0.1:18080/livez
```

Open `http://127.0.0.1:18080/` in a browser. In Settings, select the configured knowledge base; the local tenant catalog reads `LOCAL_ALLOWED_TENANTS`. `/livez` confirms only API liveness. `/readyz` also checks tenant index, source metadata, startup state and model readiness, so it should not be expected to pass without the separate working supervisor. A stopped or unavailable model is not repaired by restarting the API.

When a compatible supervisor is available, a user can enter an English question such as “What does the document say about the project schedule?” or a Hebrew question such as “מה כתוב במסמך על לוח הזמנים?” and use a returned source link to open the original. These are illustrative questions, not measured results. Model-backed querying and opening a citation returned by a query have not been established for a fresh install of this publication.

## Stop, restart and troubleshoot

Press Ctrl+C in the terminal running Uvicorn. Stop a separately managed supervisor through its own service manager; the API does not own it. Repeat the environment exports and command to restart the API. Chroma, originals, the model cache and any configured conversation SQLite file persist in the selected data directory; browser conversation history is kept in the browser's local storage. The backend conversation store is otherwise in memory. Extraction audit files and the report may contain document-derived information; protect them like your documents. Terminal output and service-manager logs are also local and may contain operational details.

- **Missing dependency:** confirm `.venv` is active and `python -m pip check` succeeds; reinstall from the lock if installation failed.
- **Tenant catalog or index unavailable:** check `APP_ENV`, `LOCAL_ALLOWED_TENANTS`, matching tenant ID, `CHROMA_PATH` and ingestion report. A document directory alone is not an index.
- **Model unavailable:** `/api/model` and `/readyz` report model state for a selected tenant. Confirm the separate supervisor is configured and running; the portable startup limitation above remains. An HTTP endpoint alone is not supported by the generation client.
- **Missing model or resources:** check the upstream model/cache and the supervisor's service logs and admission state. Do not weaken checks or equate available RAM with GPU VRAM.
- **Permission or port errors:** verify ownership and parent-directory traversal for your data directory; use another unused loopback port if `18080` is occupied. Check listeners with `ss -ltnp`.
- **Ingestion failure:** inspect the per-document report and terminal error, check file readability, supported type, OCR dependencies and available disk. Do not delete user data as a generic repair step.

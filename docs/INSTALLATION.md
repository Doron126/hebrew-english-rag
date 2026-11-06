# Installation and first use

This guide sets up the application on Linux x86-64, including Ubuntu under WSL2. You will install the Python application and a local model server, create your own knowledge base, and index your own documents. The repository starts empty; the setup commands create the configuration and data for your installation.

## Requirements

| Component | What to prepare |
|---|---|
| Python | Python 3.12 with a virtual environment. The dependency profile uses CPU PyTorch. |
| Local model server | A `llama-server` executable from [llama.cpp b12110](https://github.com/ggml-org/llama.cpp/releases/tag/b12110). Build instructions are below. |
| Generation model | An instruction-tuned GGUF model with a chat template. The walkthrough uses Qwen3-4B-Instruct-2507 in the Q4_K_M GGUF published by Unsloth. |
| Embedding model | Downloaded on first ingestion. The walkthrough selects `intfloat/multilingual-e5-small`; initialization otherwise defaults to `intfloat/multilingual-e5-large-instruct`. |
| Hardware | Enough system RAM for the generation model, embeddings, application, and context buffers. GPU acceleration is optional and uses a compatible llama.cpp build. |
| Storage and network | Space for Python dependencies, model files, indexes, and your documents. Installation and the first model downloads need internet access. |
| Optional OCR | Tesseract with Hebrew and English language data for scanned PDFs and images. |

Model file size is only part of the memory requirement. Embeddings and retrieval use system RAM even when generation is offloaded to a GPU. Choose model size and GPU layer settings for the memory available on your machine.

The examples below use Ubuntu package commands. Run them in the Linux shell. Access to the private GitHub repository is required while its visibility remains private.

## 1. Install the application

Install Git and curl if needed:

```bash
sudo apt update
sudo apt install git curl
```

Clone the repository and create a Python 3.12 environment:

```bash
git clone https://github.com/Doron126/hebrew-english-rag.git
cd hebrew-english-rag
python3.12 -m venv .venv
. .venv/bin/activate
```

<details>
<summary>If Python 3.12 or venv support is missing</summary>

[uv](https://docs.astral.sh/uv/getting-started/installation/) can download Python and create the environment. Using the official installer and its default location, run this from the checkout in place of the virtual-environment command above:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" venv --python 3.12 --seed .venv
. .venv/bin/activate
```

</details>

Install CPU PyTorch first, followed by the application dependencies:

```bash
python -m pip install --no-deps -r requirements-cpu.txt
python -m pip install -r requirements.lock.txt
python -m pip check
```

`requirements-cpu.txt` installs the CPU PyTorch wheel from its official package index. The next command installs all remaining dependencies together using the pinned versions in `requirements.lock.txt` from PyPI. Keep this order. These packages provide the FastAPI browser application, ingestion, embeddings, and retrieval; the native model server is installed separately in the next step.

## 2. Obtain llama-server

You can use the Ubuntu x64 CPU archive from the [official b12110 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12110), keeping its executable and accompanying libraries together. Alternatively, build the pinned version below. These commands keep the model runtime outside the application checkout and leave your current directory unchanged.

```bash
sudo apt install build-essential cmake libssl-dev

RAG_LLAMA_DIR="$HOME/.local/share/hebrew-english-rag-tools/llama.cpp-b12110"
mkdir -p "$HOME/.local/share/hebrew-english-rag-tools"
git clone --depth 1 --branch b12110 \
  https://github.com/ggml-org/llama.cpp.git "$RAG_LLAMA_DIR"
cmake -S "$RAG_LLAMA_DIR" -B "$RAG_LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$RAG_LLAMA_DIR/build" --config Release \
  --target llama-server -j 2
"$RAG_LLAMA_DIR/build/bin/llama-server" --version
```

The resulting executable is `$HOME/.local/share/hebrew-english-rag-tools/llama.cpp-b12110/build/bin/llama-server`. The commands follow the [upstream CMake build instructions](https://github.com/ggml-org/llama.cpp/blob/b12110/docs/build.md).

For NVIDIA GPU generation, follow the upstream CUDA prerequisites and configure the build with `-DGGML_CUDA=ON`. Select the number of layers to offload with `--gpu-layers` during application initialization. The walkthrough below uses CPU generation.

## 3. Download a generation model

Download a GGUF file from its publisher and review the model card and license. This example uses **Qwen3-4B-Instruct-2507**, with the Q4_K_M quantization from [Unsloth's GGUF publication](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF). The file is approximately 2.50 GB. The download below pins the publisher revision and checks the file's SHA-256.

```bash
RAG_MODEL_DIR="$HOME/models/Qwen3-4B-Instruct-2507"
mkdir -p "$RAG_MODEL_DIR"
curl --fail --location \
  --output "$RAG_MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
  "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/18727206c51467496bfba014368bd0a30e97f411/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
printf '%s  %s\n' \
  "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597" \
  "$RAG_MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" | sha256sum --check
```

The checksum should report `OK` before you continue. You can choose another instruction-tuned GGUF supported by the pinned llama.cpp build. Answer quality depends on the model, extraction results, and available evidence; choose a model suited to the languages and documents you intend to query. Larger models also require more memory.

## 4. Initialize your installation

From the application checkout, with `.venv` active:

```bash
python manage.py init \
  --llama-server "$HOME/.local/share/hebrew-english-rag-tools/llama.cpp-b12110/build/bin/llama-server" \
  --model "$HOME/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
  --model-id "Qwen3-4B-Instruct-2507" \
  --data-dir "$HOME/hebrew-english-rag-data" \
  --embedding-model intfloat/multilingual-e5-small
```

If you downloaded a prebuilt llama.cpp archive, replace `--llama-server` with the absolute path to its executable.

Initialization writes the checkout's `.env` and creates the external data directories and a private local model API key. **It creates no tenants and indexes no documents.** The generated configuration is read automatically by `manage.py`; no repeated environment exports are needed.

The default application port is `18080`, and the model server uses `18081`. To choose other ports at initialization, add `--port 18080 --model-port 18081` with your preferred values. Both services bind to `127.0.0.1`. A GPU-enabled installation can add a positive `--gpu-layers` value; the default is `0`.

If a configuration already exists, initialization leaves it in place. Edit that `.env` to change settings, then restart the application. Use absolute paths in it; values are read as configuration, not as shell expressions.

## 5. Create a knowledge base and ingest your files

Choose a tenant ID for your documents. The example uses `my-documents`; replace it consistently if you prefer another ID. IDs contain 1–64 ASCII letters, digits, hyphens, or underscores, starting and ending with a letter or digit. The display name can be Hebrew or English.

```bash
python manage.py tenant add my-documents --name "My documents"
python manage.py tenant list
```

The command prints the new document directory. With the paths used above, it is:

```text
~/hebrew-english-rag-data/documents/my-documents/
```

Copy your own files into that directory using a file manager or shell. Start with a few readable documents, then index them:

```bash
python manage.py ingest --tenant my-documents
```

The first ingestion downloads the selected embedding model. Subsequent runs reuse the model cache and can skip unchanged documents. Keep the same embedding model for ingestion and querying.

The command prints a summary and the report path. For this example:

```bash
python -m json.tool "$HOME/hebrew-english-rag-data/reports/my-documents-ingest.json"
```

Review failed or empty files before moving on. A partial import exits with code `2` while retaining successful work. Correct the reported file or dependency issue and rerun ingestion.

Keep the originals in the tenant's document directory: citations open those files. Document updates should be followed by another ingestion run to refresh the index and source metadata.

### File formats and OCR

| Files | Processing |
|---|---|
| Text, Markdown, HTML, CSV/TSV, JSON | Text and structure extraction |
| PDF | Native text extraction, with optional OCR for scanned content |
| DOCX, XLSX, PPTX | Extraction from Office document structures |
| PNG, JPEG, WebP | OCR through Tesseract when available |

For Hebrew and English OCR on Ubuntu:

```bash
sudo apt install tesseract-ocr tesseract-ocr-heb tesseract-ocr-eng
tesseract --list-langs
```

Confirm that `heb` and `eng` are listed. OCR results depend on scan quality and layout. Complex tables and images may need review; the ingestion report helps identify files that did not produce usable text. The UI reads the knowledge bases you configure; files are added through their directories and the ingestion command.

## 6. Start the application

Check the dependencies and configuration, then launch:

```bash
python manage.py doctor
python manage.py serve
```

The launcher starts `llama-server`, waits for it to load the model, and then starts the API. When the API reports that startup is complete, open **[http://127.0.0.1:18080](http://127.0.0.1:18080)** in your browser.

In **Settings**, choose **My documents**. Use the language button in the top bar to switch between Hebrew and English. For your first question, request a summary using the **exact filename you indexed, including its extension**. If your file is named `report.pdf`, enter `Summarize report.pdf.` or `סכם את report.pdf.`. Replace `report.pdf` with your own filename. Then open a returned source citation to review the original context.

In a second terminal, you can check the running application:

```bash
curl --fail http://127.0.0.1:18080/livez
curl --fail -H "X-Tenant-ID: my-documents" \
  http://127.0.0.1:18080/readyz
```

`/livez` checks the API process. `/readyz` additionally checks the selected tenant's index, source metadata, and model runtime. `doctor` checks local setup before launch; it does not load the model or answer a question.

## 7. Stop, restart, and update documents

Press **Ctrl+C** in the terminal running `manage.py serve`. It stops both processes it launched. Your documents, index, model cache, and conversation database remain in the data directory.

To start again, open a terminal in the checkout:

```bash
. .venv/bin/activate
python manage.py serve
```

To update a knowledge base, stop the application, add or edit its files, and rerun `python manage.py ingest --tenant my-documents`. Start the application again when ingestion finishes. Use `--force` with the ingestion command when you explicitly want to re-extract and re-embed unchanged files.

For another knowledge base, run `python manage.py tenant add` with a new ID, copy files into its printed directory, and ingest that tenant. Restart the application to load the updated tenant list.

## Configuration and data locations

The generated `.env` is local to this checkout. `manage.py` passes its values to the processes it starts. To use another configuration file, place the global option before the command, for example `python manage.py --config /absolute/path/workspace.env doctor`.

| Setting | Purpose |
|---|---|
| `RAG_MODEL_RUNTIME=llama_cpp` | Select the portable local model connection |
| `LLAMA_SERVER_BIN`, `LLAMA_MODEL_PATH` | Absolute paths to the server executable and GGUF model |
| `LLAMA_MODEL_ID`, `BASE_MODEL` | Model alias; keep these two values in agreement |
| `LLAMA_BASE_URL` | Model origin, such as `http://127.0.0.1:18081` |
| `LLAMA_API_KEY_FILE` | Private key generated for the local model connection |
| `APP_PORT` | Browser/API port; default `18080` |
| `LLAMA_N_GPU_LAYERS` | Generation layers to offload; default `0` |
| `LLAMA_CONTEXT_SIZE_TOKENS` | Model context; initialized to `4096` |
| `LLAMA_THREADS` | CPU threads used by the model server |
| `LOCAL_ALLOWED_TENANTS` | Tenant IDs created by the setup commands |
| `DOCS_BASE_DIR`, `TENANT_CONFIG_DIR`, `CHROMA_PATH` | Original documents, tenant profiles, and persistent index |
| `EMBED_MODEL`, `EMBED_DEVICE` | Embedding model and device; the local setup uses CPU |
| `USE_RERANKER`, `CHAT_USE_RERANKER` | Optional rerankers; both disabled in the initial setup |
| `HF_HOME` | Download cache for embedding and reranking models |
| `PROJECT_X_CONVERSATION_STATE_PATH` | SQLite conversation database enabled by local initialization |
| `INGEST_EXTRACTION_AUDIT_DIR` | Document extraction audit output |

With the example data directory, originals live in `documents/<tenant-id>`, indexes in `chroma`, tenant profiles in `config/tenants`, and embedding-model downloads in `cache/huggingface`. The GGUF file remains at the model path you supplied during initialization. Ingestion reports are written to `reports`, extraction audits to `extraction-audit`, and model output to `logs/model.log`. Backend conversation state is stored in `conversations.sqlite3`; browser history and preferences also use browser local storage.

The index, reports, audits, and conversations contain user data. Keep the data directory and local configuration private. Model downloads have their own upstream licenses. To retain an exact setup, record the model revision and file checksum alongside the pinned Python packages and llama.cpp version.

## Troubleshooting

| Symptom | What to check |
|---|---|
| A Python module is missing or incompatible | Activate `.venv`, install the CPU manifest and main lock in that order, then run `python -m pip check` and `python manage.py doctor`. |
| The model does not load | Read the printed model log path. Check the GGUF download, server executable, available memory, and GPU build/settings. |
| A port is already in use | Choose unused application and model ports in `.env`. The launcher leaves existing listeners alone. |
| The knowledge base is absent from Settings | Check `python manage.py tenant list` and restart the application after adding a tenant. |
| Readiness reports an empty or missing index | Run ingestion for the selected tenant and inspect its report. A document directory becomes searchable after ingestion. |
| A source link is unavailable | Check that the original still exists in its tenant directory. Rerun ingestion after changing the file. |
| A scan produces little or no text | Confirm Tesseract and the required language data are installed, inspect the scan, and rerun ingestion with `--force`. |
| Model requests are slow | Check whether generation uses CPU or GPU, the selected model size, and the model log. The setup's request and startup timeouts can be adjusted in `.env`. |

The [architecture guide](ARCHITECTURE.md) explains how the components connect, how source access works, and where tenant boundaries are applied.

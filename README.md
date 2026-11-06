# Hebrew-English RAG Application

Hebrew-English RAG Application is a local document search and question-answering application developed by **Doron Shamo**. It indexes users' documents and can return grounded responses with links to their sources. Its interface and text processing support Hebrew and English.

Active development has stopped. The application is retained as an **experimental reference implementation**. Parsing coverage, retrieval, and answers have limitations; verify answers against the original documents.

**No original tenants, customer documents, or prebuilt indexes are included. Create your own tenants and ingest your own documents.**

The code provides a browser interface, a Python/FastAPI backend, document extraction and indexing into Chroma, multilingual embeddings, and a separate local Gemma model supervisor. The installation is based on Ubuntu 26.04 under WSL and Python 3.14.4. Other environments are not established by this publication.

**Current installation limitation:** the model supervisor is bound to a specific historical WSL runtime profile and cannot be configured for arbitrary model paths or a separately running endpoint. A fresh checkout can prepare data and use the ingestion pipeline, but this publication does not provide a verified portable model-startup path or an end-to-end first answer. See the [installation guide](docs/INSTALLATION.md) before starting.

## Getting started

With authorized access to this private repository, clone it and follow [installation and first use](docs/INSTALLATION.md). That guide covers prerequisites, an isolated data directory, user-defined tenant configuration, document ingestion and the model limitation. See [architecture](docs/ARCHITECTURE.md) for components, data flow and tenant boundaries.

The documented interface is the FastAPI-served browser application. `webui_streamlit.py` is retained as a legacy reference only; Streamlit is not included in the supported dependency lock, and that alternative interface is not part of the installation walkthrough.

Source revision used for this publication: `c015ec4c62b68ba051e7eaa421c01d80935a9469`.

## Author

Doron Shamo

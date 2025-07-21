# Brainstorming Solutions for PDF-based RAG System Assignment

## Objective Overview
The goal is to build a Retrieval-Augmented Generation (RAG) system that ingests PDF documents from a specified Google Drive folder, extracts text (with OCR if necessary), converts content into vector embeddings, stores them in a searchable index, and provides a chat interface for natural language queries. Answers should include references to source documents and page numbers, with clickable links to the relevant sections. Priorities emphasize breadth in search results (covering as many relevant references as possible) over speed.

Key components:
- **Document Ingestion**: Download and extract text from PDFs in the folder (https://drive.google.com/drive/folders/1lo09v9yZthWMDvyCaCXKeT1ZcxR-DEvB?usp=drive_link).
- **Indexing and Search**: Chunk text, embed into vectors, store in a vector database, enable semantic search.
- **Question Interface**: Basic chat UI similar to NotebookLM, with references and navigation to sources.
- **Priorities**: Maximize breadth (high top-k retrieval, inclusive relevance) rather than speed.

Below, I brainstorm all possible solutions, categorized by key technologies and approaches. I'll include pros, cons, and how they align with priorities. This draws from web search results like SemanticPDF (https://semanticpdf.com/), SK-DocumentSearch (https://github.com/leungkimming/SK-DocumentSearch), and Elastic's RAG on PDFs (https://www.elastic.co/search-labs/blog/rag-with-pdfs-genai-search).

## 1. Framework-Based Solutions
Use high-level frameworks to handle ingestion, embedding, retrieval, and UI.

### 1.1 LangChain or LlamaIndex
- **Description**: Use LangChain for chain-based RAG pipelines or LlamaIndex for index management. Ingest PDFs with loaders (e.g., PyPDFLoader with OCR via Tesseract), chunk text, embed using models like OpenAI's text-embedding-ada-002 or HuggingFace's sentence-transformers, store in a vector store, and query with a retriever. Build UI with Streamlit or Gradio.
- **Embedding and Storage Options**: Integrate with vector DBs like FAISS (local), Pinecone (cloud), or Chroma.
- **Breadth Enhancement**: Set high top-k (e.g., 50+), use MMR (Maximum Marginal Relevance) for diversity, or hybrid search (keyword + semantic).
- **UI**: Streamlit chat interface with markdown rendering for references (e.g., hyperlinks to PDF pages).
- **Pros**: Rapid prototyping, modular, handles metadata for references (doc name, page). Aligns with breadth via configurable retrieval.
- **Cons**: Dependency-heavy; may require API keys for embeddings/LLMs. OCR integration needs extra setup.
- **Alignment**: Excellent for breadth; can rerank results to include more references.

### 1.2 Haystack
- **Description**: Open-source framework for building search systems. Use DocumentStore with Elasticsearch or FAISS, PDF converters with OCR, embed with HuggingFace models, and build a pipeline for retrieval + generation.
- **Breadth Enhancement**: Use ExtractiveReader for multiple snippets, or high top-k with diversity ranking.
- **UI**: Integrate with Streamlit or build a custom web app.
- **Pros**: Flexible pipelines, good for custom chunking to maximize breadth.
- **Cons**: Steeper learning curve than LangChain.
- **Alignment**: Strong for semantic search breadth; supports multi-document retrieval.

### 1.3 Semantic Kernel (as in SK-DocumentSearch)
- **Description**: Based on https://github.com/leungkimming/SK-DocumentSearch. Use Microsoft's Semantic Kernel to ingest PDFs, embed with HuggingFace (e.g., intfloat/e5-large-v2), store in Redis, search semantically, and summarize with models like bart_lfqa.
- **Breadth Enhancement**: Redis vector search with high similarity thresholds; combine with keyword search for broader coverage.
- **UI**: Extend with a simple Flask app for chat, displaying references from metadata.
- **Pros**: Integrates well with .NET/C# ecosystems; uses Redis for fast, scalable storage (though speed isn't priority).
- **Cons**: Requires Docker for Redis/HuggingFace; focused on summarization, may need extension for full references.
- **Alignment**: Good for breadth via semantic search on embeddings; can pull multiple chunks.

## 2. Vector Database-Centric Solutions
Focus on a specific DB for storage and search.

### 2.1 Elasticsearch with semantic_text (as in Elastic's Blog)
- **Description**: Based on https://www.elastic.co/search-labs/blog/rag-with-pdfs-genai-search. Use Elastic's semantic_text field for automatic chunking and embedding (e.g., with multilingual-e5-small model). Ingest PDFs via attachment processor (Apache Tika for extraction/OCR). Query via Playground UI or API.
- **Breadth Enhancement**: semantic_text handles chunking; use high result limits and hybrid search for broad coverage across languages.
- **UI**: Elastic's Playground for prototyping chat; extend to Kibana dashboard or custom app.
- **Pros**: End-to-end (ingestion to UI), handles multilingual PDFs well, transparent relevance scores.
- **Cons**: Requires Elastic setup (cloud or on-prem); not free for large-scale.
- **Alignment**: Prioritizes breadth with configurable chunking and broad retrieval; references via metadata.

### 2.2 Pinecone or Weaviate
- **Description**: Cloud vector DBs. Extract text with PDFPlumber + Tesseract OCR, chunk, embed with Sentence Transformers, upsert to Pinecone/Weaviate with metadata (doc, page).
- **Breadth Enhancement**: Query with high top-k, filter by namespace for document grouping; Weaviate's generative module for RAG.
- **UI**: Gradio for chat, with PDF.js for rendering clickable pages.
- **Pros**: Scalable, managed service; Weaviate supports hybrid search.
- **Cons**: Costs for cloud; API rate limits.
- **Alignment**: High top-k supports breadth; metadata for exact references.

### 2.3 Local Options: FAISS or Chroma
- **Description**: For non-cloud setups. Use FAISS for in-memory index or Chroma for persistent storage. Embed with local HuggingFace models.
- **Breadth Enhancement**: Custom retrieval with large k; add rerankers like Cohere.
- **UI**: Streamlit with session state for chat history.
- **Pros**: Free, offline; full control.
- **Cons**: Limited scalability; FAISS is memory-intensive for large collections.
- **Alignment**: Easy to tweak for maximum breadth (e.g., retrieve all matches above threshold).

## 3. Custom or Minimalist Solutions
Build from scratch without heavy frameworks.

### 3.1 Python Script with HuggingFace and FAISS
- **Description**: Script to download PDFs via gdown, extract with PyMuPDF + Pytesseract OCR, chunk by paragraphs, embed with all-MiniLM-L6-v2, index in FAISS. Use OpenAI API for generation.
- **Breadth Enhancement**: Retrieve top-100, filter/rerank minimally to include more.
- **UI**: Simple CLI or Flask app with HTML for references (embed PDF viewer).
- **Pros**: Lightweight, customizable.
- **Cons**: More coding effort; no built-in UI.
- **Alignment**: Full control over retrieval parameters for breadth.

### 3.2 Using SemanticPDF Tool
- **Description**: Based on https://semanticpdf.com/. Upload PDFs to their service for automatic scanning, embedding, and semantic search. Extend with custom script for batch ingestion and API queries.
- **Breadth Enhancement**: Their highlighting of relevant results; query multiple times for broader context.
- **UI**: Their web interface, or integrate via API.
- **Pros**: No-code ingestion/search; handles PDFs directly.
- **Cons**: Limited to 10MB/file, web-only; privacy concerns for sensitive docs.
- **Alignment**: Semantic search focuses on relevance, but can be queried broadly.

## 4. UI and Reference Handling Approaches
Across solutions:
- **Chat Interface**: Streamlit/Gradio for quick prototypes; React/Next.js for production with PDF.js for rendering pages on click.
- **References**: Store metadata (file, page) with each chunk. In responses, list as hyperlinks (e.g., [Doc.pdf - Page 5](link-to-viewer)).
- **Breadth**: Implement multi-query retrieval or query expansion to cover variations; use LLMs to generate sub-queries for broader search.

## 5. General Considerations
- **OCR**: Essential for scanned PDFs; integrate Tesseract via pytesseract.
- **Chunking**: Paragraph-based for breadth; overlap chunks to avoid missing context.
- **Embeddings**: Multilingual models (e.g., E5) if PDFs are multi-lang.
- **LLM for Generation**: OpenAI GPT, Llama2 (local), or Bedrock for summarizing broad retrievals.
- **Evaluation**: Test breadth by checking if all known references are included.
- **Challenges**: Handling large PDFs (split processing); ensuring clickable links (host PDFs locally or use Google Drive embeds).
- **Next Steps**: Prototype with LangChain + Chroma for quick validation, then scale based on needs.

This brainstorming covers a wide range of solutions, from no-code to custom builds, ensuring we can select based on resources and expertise. 
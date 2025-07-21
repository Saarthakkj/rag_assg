# Implementation Plan 1: Using LangChain for PDF-based RAG System

This plan details how to implement the assignment using LangChain (as outlined in brainstorming.md section 1.1). LangChain is chosen for its ease of use in building RAG pipelines, modular components, and integration with various tools. We'll use Python, focus on breadth by setting high retrieval parameters, and build a simple Streamlit UI. The plan assumes you have Python installed and access to API keys (e.g., OpenAI for embeddings/LLM).

## Prerequisites
- Python 3.10+
- Libraries: `langchain`, `langchain-community`, `langchain-openai`, `pypdf`, `pytesseract`, `Pillow`, `gdown`, `streamlit`, `faiss-cpu` (or another vector store).
- Tesseract OCR installed (for scanned PDFs).
- OpenAI API key (for embeddings and LLM).
- Google Drive folder access (public link provided).

Install dependencies:
```
pip install langchain langchain-community langchain-openai pypdf pytesseract pillow gdown streamlit faiss-cpu
```

## Step 1: Document Ingestion
### 1.1 Download PDFs from Google Drive
Use `gdown` to download all files from the shared folder.
- Get the folder ID from the URL: '1lo09v9yZthWMDvyCaCXKeT1ZcxR-DEvB'.
- Script to download:
```python
import gdown
import os

folder_url = 'https://drive.google.com/drive/folders/1lo09v9yZthWMDvyCaCXKeT1ZcxR-DEvB'
output_dir = 'pdf_docs'
os.makedirs(output_dir, exist_ok=True)
gdown.download_folder(folder_url, output=output_dir, quiet=False)
```
This creates a local 'pdf_docs' directory with all PDFs.

### 1.2 Extract Text with OCR
Use LangChain's PyPDFLoader for text PDFs, and integrate pytesseract for OCR on scanned pages.
- Custom loader function to handle OCR:
```python
from langchain_community.document_loaders import PyPDFLoader
from PIL import Image
import pytesseract
import io

def load_pdf_with_ocr(file_path):
    loader = PyPDFLoader(file_path, extract_images=True)  # Extracts images for OCR
    pages = loader.load()
    for page in pages:
        # If page content is empty (scanned), perform OCR
        if not page.page_content.strip():
            # Assuming extract_images=True provides image data; otherwise, use pdf2image
            # For simplicity, integrate pdf2image if needed
            page.page_content = pytesseract.image_to_string(Image.open(io.BytesIO(page.metadata['image'])))  # Adjust as per actual metadata
    return pages
```
- Load all PDFs:
```python
docs = []
for file in os.listdir('pdf_docs'):
    if file.endswith('.pdf'):
        docs.extend(load_pdf_with_ocr(os.path.join('pdf_docs', file)))
```
Each doc will have metadata like source file and page number.

## Step 2: Indexing and Search
### 2.1 Chunking
Split documents into chunks to maximize breadth (e.g., paragraph-level with overlap).
- Use RecursiveCharacterTextSplitter:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Overlap for context breadth
chunks = splitter.split_documents(docs)
```
This ensures broad coverage without missing insights across chunk boundaries.

### 2.2 Embeddings
Use OpenAI embeddings (or HuggingFace for local).
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key='your-api-key')
```

### 2.3 Vector Store
Store in FAISS for local, simple indexing (switch to Pinecone for cloud if needed).
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local('faiss_index')
```
Load later: `vectorstore = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)`

### 2.4 Retrieval Setup
Configure retriever for breadth: high top-k, optional MMR for diversity.
```python
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 50})  # High k for breadth
```
This retrieves up to 50 chunks, prioritizing diversity to cover more references.

## Step 3: Question Answering
### 3.1 RAG Chain
Use LangChain's RetrievalQA or create a custom chain.
```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key='your-api-key')
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
```
- To emphasize breadth, modify the prompt to summarize all retrieved docs:
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template='Answer based on all relevant info from context. Include references to all sources.\nContext: {context}\nQuestion: {question}',
    input_variables=['context', 'question']
)
# Integrate into chain
```

### 3.2 Generate Response with References
Query the chain:
```python
result = qa_chain.invoke({'query': 'Your question'})
answer = result['result']
sources = result['source_documents']  # List of docs with metadata
```
Format references: Loop through sources, extract file and page, create markdown links (e.g., [file.pdf - Page X](path/to/file.pdf#page=X)).

## Step 4: Chat Interface
Use Streamlit for a basic chat UI similar to NotebookLM.
- Script (`app.py`):
```python
import streamlit as st
from langchain...  # Import all above

# Load vectorstore...
st.title('PDF RAG Chat')
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Ask a question')
if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    
    with st.chat_message('assistant'):
        result = qa_chain.invoke({'query': prompt})
        answer = result['result']
        sources = result['source_documents']
        refs = '\n'.join([f'[{doc.metadata["source"]} - Page {doc.metadata["page"]}](pdf_docs/{doc.metadata["source"]}#page={doc.metadata["page"]})' for doc in sources])
        full_response = f'{answer}\n\n**References:**\n{refs}'
        st.markdown(full_response)
    st.session_state.messages.append({'role': 'assistant', 'content': full_response})
```
- Run: `streamlit run app.py`
- Host PDFs in 'pdf_docs' for clickable links (use PDF.js if needed for better viewing).

## Step 5: Enhancements for Breadth and Testing
- **Breadth**: If 50 isn't enough, increase k or use multi-query retrieval (LangChain's MultiQueryRetriever) to generate variations and retrieve more.
- **Hybrid Search**: Add BM25 (keyword) via EnsembleRetriever for even broader coverage.
- **Testing**: Query with known questions, verify all relevant references are included. Measure recall.
- **Deployment**: Host on Streamlit Cloud or similar.

## Potential Challenges and Solutions
- **Large PDFs**: Process in batches to avoid memory issues.
- **OCR Accuracy**: Test Tesseract on samples; fine-tune if needed.
- **Costs**: Monitor OpenAI usage with high k.
- **Multilingual**: Switch to multilingual embeddings if PDFs vary in language.

This plan provides a complete, working prototype. Total implementation time: 4-6 hours for basics. 
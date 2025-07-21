'''
TODO : 

    1. shocase sources (in large / multiple files aslo)
    2. evaluate on bleu eta
    3. add more features- 
        - multi-modal?
        - highlight the source?
        - knowledge-graphs?????

'''
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from main import enhanced_qa_chain
import base64
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

st.title('PDF RAG CHAT')
 # Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'active_pdf_viewers' not in st.session_state:  # New: Track which PDFs are open (keyed by message index and button index)
    st.session_state.active_pdf_viewers = {}       


def display_reference(source, page, message_index, i):
    # print(f"inside display_refrence")
    button_label = f"{source} - Page {page}"
    button_key = f"ref_button_{message_index}_{i}"  # Unique key including message index and i
    
    if st.button(button_label, key=button_key):
        # Toggle the viewer state
        viewer_key = f"viewer_{message_index}_{i}"
        if viewer_key in st.session_state.active_pdf_viewers:
            del st.session_state.active_pdf_viewers[viewer_key]  # Close if already open
        else:
            st.session_state.active_pdf_viewers[viewer_key] = (source, page)  # Open: store source and page
    
    # If this viewer is active, render the PDF
    viewer_key = f"viewer_{message_index}_{i}"
    if viewer_key in st.session_state.active_pdf_viewers:
        source, page = st.session_state.active_pdf_viewers[viewer_key]

        print(f" source : {source}")
        pdf_path = os.path.abspath(os.path.join('pdf_docs' , source))  # Fixed: include 'pdf_docs'
        print(f"source: {source} | Attempting to load from: {pdf_path}")
        try:
            if not isinstance(page, int):
                raise ValueError(f"Invalid page number: {page}")
            pdf_viewer(
                pdf_path,
                width=700,
                height=1000,
                zoom_level=1.0,
                viewer_align="right",
                show_page_separator=True,
                scroll_to_page=page
            )
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}. The PDF might be too large, or check the path/params.")

def parse_sources(answer):
    citations = re.findall(r'\[source-page: (.*?)\]', answer)
    unique_pairs = set()
    parsed = []
    for cit in citations:
        print(f" cit in citations : {cit}")
        # Split by comma to handle multiple filename-pageno pairs
        pairs = [pair.strip() for pair in cit.split(',')]
        for pair in pairs:
            if '-' in pair:
                parts = pair.rsplit('-', 1)  # Split on last '-' to get source and page
                source = parts[0].strip()
                try:
                    page = int(parts[1].strip()) + 1  # +1 to make 1-indexed for pdf_viewer
                    unique_pairs.add((source, page))
                except ValueError:
                    continue  # Skip invalid page numbers
    for source, page in sorted(unique_pairs):
        parsed.append({'source': source, 'page': page})
    return parsed

def render_answer_with_buttons(answer: str, message_index: int):
    """Render answer text and insert reference buttons inline where citations appear."""
    # Split the answer on citation patterns while keeping the citation tokens.
    parts = re.split(r"(\[source-page: [^\]]+\])", answer)
    btn_counter = 0  # Ensure unique keys across a single message

    for part in parts:
        # If the part is a citation token, create buttons for each citation
        if part.startswith("[source-page:"):
            # Extract citation content without the surrounding brackets and prefix
            citation_body = part[len("[source-page:"): -1]  # removes prefix and trailing ']'
            # A single token might contain multiple citations separated by commas
            citations = [c.strip() for c in citation_body.split(',') if '-' in c]

            for cit in citations:
                source, page_str = cit.rsplit('-', 1)
                try:
                    page = int(page_str.strip()) + 1  # pdf_viewer is 1-indexed
                except ValueError:
                    continue  # skip malformed citation
                display_reference(source.strip(), page, message_index, btn_counter)
                btn_counter += 1
        else:
            # Regular text segment — render as markdown
            if part.strip():  # Avoid rendering empty strings
                st.markdown(part)

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            # Render answer text interleaving citation buttons
            render_answer_with_buttons(message['content'], idx)
        else:
            # For user messages, simple markdown rendering
            st.markdown(message['content'])
    
prompt = st.chat_input("Ask a question")
if prompt: 
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    
    with st.chat_message('assistant'):
        result = enhanced_qa_chain(prompt)
        answer = result['result']
        parsed_sources = parse_sources(answer)  # Parse sources from answer text
        # Render answer with inline buttons
        render_answer_with_buttons(answer, len(st.session_state.messages))
        
        # Store the full assistant message with parsed_sources
        st.session_state.messages.append({
            'role': 'assistant',
            'content': answer,
            'parsed_sources': parsed_sources  # Store parsed sources for regeneration
        }) 

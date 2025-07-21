# TODO : 
# - how to make sure the correct chunk_size and chunk_overlap is passed into RecursiveCharacterTextSplitter ? (how to test it?)
# - optimize for storage (FAISS index)
# - optimize for timing of embedding
import os
from PIL import Image
import pytesseract
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
import pickle
import random
from dotenv import load_dotenv

load_dotenv()

DOCS_PICKLE_PATH = 'docs.pkl'


def load_pdf_with_ocr(file_path):

    print(f'filepath {file_path}')
    loader = PyPDFLoader(file_path, extract_images=True)  # Extracts images for OCR
    pages = loader.load()
    for page in pages:
        if page: 
            # print(f"page has text")
            continue
        # If page content is empty (scanned), perform OCR
        if not page.page_content.strip():
            print("page content is empty")
            # Assuming extract_images=True provides image data; otherwise, use pdf2image
            # For simplicity, integrate pdf2image if needed
            # This line performs OCR (Optical Character Recognition) on an image extracted from the PDF page.
            # It loads the image bytes from the page's metadata, converts it to a PIL Image, 
            # and then uses pytesseract to extract any text from the image, assigning the result to page.page_content.
            page.page_content = pytesseract.image_to_string(Image.open(io.BytesIO(page.metadata['image'])))
    return pages

docs = []
# for file in os.listdir('pdf_docs'):
#     if file.endswith('.pdf'):
#         docs.extend(load_pdf_with_ocr(os.path.join('pdf_docs', file)))
        
        
#print(f"before if-else")
if os.path.exists(DOCS_PICKLE_PATH):
    with open(DOCS_PICKLE_PATH, 'rb') as f:
        docs = pickle.load(f)
else:
    docs = []
    #print(f"else-block")
    for file in os.listdir('pdf_docs'):
        if file.endswith('.pdf'):
            #print(f" file name : {file}")
            docs.extend(load_pdf_with_ocr(os.path.join('pdf_docs', file)))
    with open(DOCS_PICKLE_PATH, 'wb') as f:
        pickle.dump(docs, f)



splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Overlap for context breadth
chunks = splitter.split_documents(docs)


# # You can choose a local model, e.g., 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if os.path.exists('faiss_index'):
    vectorstore = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local('faiss_index')


bm25_retriever = BM25Retriever.from_documents(chunks, k=50)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 50})  
ensemble_retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.5, 0.5])
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", google_api_key=os.getenv('gemini'))
multi_retriever = MultiQueryRetriever.from_llm(retriever=ensemble_retriever, llm=llm)


cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
def rerank_docs(query, docs): 
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return sorted_docs[:30] # top 30 after re-ranking

# Enhanced general prompt for flexibility, depth, and grounding (applies to any query/topic)
prompt = PromptTemplate(
    template='''
    Answer the question comprehensively based ONLY on the provided context. 
    - Use all relevant information from the context to ensure breadth and depth.
    - If the question involves discussing foundational roles, concepts, branches, or applications (e.g., in a field like AI or others), systematically:
      1. List key concepts or elements (aim for completeness, e.g., 10+ if applicable).
      2. Explain each one's role (e.g., in processes like data handling, modeling, or decision-making).
      3. Link to specific applications or examples from the context (e.g., real-world systems, domains like language processing, vision, or generation).
      4. Synthesize connections between concepts for a holistic view.
    - For simpler questions, provide direct, detailed answers without unnecessary lists.
    - Include inline citations [source-page: <filename>-<page-no>] for every key point or fact to ground your response, using the "Source" and "Page" metadata provided in the context for each section. STRICTLY follow the format [source-page: filename-pageno] - do NOT include chapter names, section titles, book name or any other text. Only use the exact filename and page number.
    - If context is insufficient, state what is known and suggest refinements, but do not hallucinate.

    Context: {context}
    Question: {question}''',
    input_variables=['context', 'question']
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",  # Use 'stuff' for simple concatenation, or 'map_reduce' for large contexts
    retriever=multi_retriever, 
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}  # Integrate the general prompt here
)

def extract_context_from_docs (docs): 
    context_parts = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown Source"))
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"Source: {source} - Page: {page}\n{doc.page_content}")
        
    return "\n\n".join(context_parts)


# Update QA chain with custom logic (includes grounding and evaluation for deeper exploration)
def enhanced_qa_chain_intermediate(query):
    # Initial retrieval and reranking
    docs = multi_retriever.get_relevant_documents(query)
    reranked_docs = rerank_docs(query, docs)  # Assuming rerank_docs is defined as before
    
    # Inject metadata into context for proper citations
    
    context = extract_context_from_docs(reranked_docs)
    # context_parts = []
    # for doc in reranked_docs:
    #     source = os.path.basename(doc.metadata.get("source", "Unknown Source"))
    #     page = doc.metadata.get("page", "N/A")
    #     context_parts.append(f"Source: {source} - Page: {page}\n{doc.page_content}")
    #     # print(f" [main.py] source inside reranked_docs : {source} , doc-metadatA : {doc.metadata} ")
    # context = "\n\n".join(context_parts)

    #print(f"\n\n context : {context}  \n\n")
    
    # Grounding: Optional summarization for large contexts (helps with depth)
    summary_prompt = PromptTemplate(
        template=
        '''Condense key info from this context for a detailed answer, preserving "Source: filename - Page: x" prefixes ONLY for sections you actually use in the summary: {text}. 
        At the end of your condensed output, add a section "## Used Sources" with citations [source-page: filename-pageNo] ONLY for sources referenced in the summary. 
        Do not include metadata or citations for unused sources. 
        STRICTLY follow the citation format [source-page: filename-pageno] - do NOT include chapter names, section titles, or any other text. Only use the exact filename and page number in the given format.
        For example, if you use a section with "Source: source_1.pdf - Page: 59", include [source-page: source_1.pdf-59] in the Used Sources section.''',
        input_variables=["text"]
    )
    summary_chain = summary_prompt | llm
    grounded_context = summary_chain.invoke({"text": context}).content   
    # print(f"\n\n  grounded_context_llm repsonse: {grounded_context} \n\n")
    
    # Genyeration using the general prompt
    formatted_prompt = prompt.format(context=grounded_context, question=query)
    llm_response = llm.invoke(formatted_prompt)
    result = {'result': llm_response.content, 'source_documents': reranked_docs}

    if '## Used Sources' not in result['result']: 
        # if used sources not present -> append top 5 docs from reranked_docs
        result['result'] += "\n\n##Used Sources\n" + "\n".join([f"[source-page: {os.path.basename(doc.metadata.get('source' , 'Uknonwn'))}--{doc.metadata.get('page' , 'N/A')}]" for doc in reranked_docs[:5]])
    

    # print(f"\n\n result is : {result}   \n\n")
    # Evaluation: Self-assess for depth/breadth and refine if needed (triggers deeper dig)
    eval_prompt = PromptTemplate(
        template="Evaluate this answer for breadth (covers multiple aspects?), depth (detailed explanations?), and grounding (citations?). Rate 1-10. If any <8 or if more exploration is needed (e.g., for conceptual questions), suggest a refined query to dig deeper. Only include a 'Refined query:' if you actually think more exploration is needed. Preserve any embedded metadata or Used Sources sections in your evaluation. {answer}",
        input_variables=["answer"]
    )
    eval_result = (eval_prompt | llm).invoke({"answer": result['result']}).content
    low_score = any(f"Rate: {score}" in eval_result or f"Rating: {score}" in eval_result or f"{score}/10" in eval_result for score in ['1', '2', '3', '4', '5', '6', '7'])
    refined_query_present = "refined query:" in eval_result.lower()
    print(f"Refined query : {refined_query_present}")
    # print(f"low_score : {low_score}")
    if low_score:
        refined_query = eval_result.split("refined query:")[-1].strip() if "refined query" in eval_result.lower() else query + " (expand on concepts and applications)"
        refined_result = enhanced_qa_chain_intermediate(refined_query)  # Recursive refinement (limit depth to 2 in production to avoid loops)
        
        #print(f"\n\n this is source_documents : {refined_result['source_documents']}\n\n")
        
        
        return {'result': refined_result['result'] + "\n\n(Original Response:\n" + result['result'] + ")", 'source_documents': refined_result['source_documents']}
    
    return result

def enhanced_qa_chain(query : str) : 
    disjoint_results = enhanced_qa_chain_intermediate(query)

    print(f" disjoint_results {disjoint_results}")
    
    # Synthesis step (few-shot prompting for cohesive, expert-style response with TL;DR and sections)
    synthesis_prompt = PromptTemplate(
        template='''
        You are an expert scientific and technical writer. The user query is: "{query}"

        Synthesize the following intermediate answer into a cohesive, expert-style response. 
        Use the few-shot examples below to guide your style: write narratively with smooth transitions, inline citations [source-page: filename-pageNo], and synthesis of key points without disjointed notes.

        Important: Do NOT shorten or summarize the intermediate answer—reformulate ALL its content into a longer, flowing narrative if needed. The output should be at least as long as the intermediate, combining a TL;DR summary upfront with the full details restructured for better flow. Preserve embedded metadata (e.g., Used Sources sections) from the intermediate, but ONLY include citations and metadata for sources actually referenced in your synthesized content—remove any unused ones. When encountering "Source: filename - Page: x" in the intermediate text, convert them to inline citations in the format [source-page: filename-pageno] while preserving the exact filename. If a "## Used Sources" section exists, use it to validate and include only referenced sources in your output. NEVER omit or replace the filename; if uncertain, retain the original format from the intermediate.

        CRITICAL CITATION FORMAT: STRICTLY follow the format [source-page: filename-pageno] where filename MUST be the actual PDF filename (containing .pdf extension). Do NOT use book names, chapter names, section titles, or any other text. Only use the exact PDF filename with .pdf extension and page number. If the filename doesn't contain .pdf or you're uncertain about the exact PDF filename, omit that citation entirely.

        Few-Shot Example 1:
        Query: Compare the main characters in two novels.
        Intermediate: Character A is ambitious Source: novel1.pdf - Page: 3. Character B is more reserved Source: novel2.pdf - Page: 5. Both face challenges in their journeys.
        ## Used Sources
        [source-page: novel1.pdf-3]
        [source-page: novel2.pdf-5]
        Synthesized: TL;DR: Character A stands out for ambition, while Character B is defined by a reserved nature; both encounter significant challenges that shape their development [source-page: novel1.pdf-3][source-page: novel2.pdf-5].

        Detailed Comparison: Character A's ambition drives much of the plot, leading to both opportunities and conflicts [source-page: novel1.pdf-3]. In contrast, Character B's reserved demeanor results in a more introspective journey, with growth emerging from overcoming personal obstacles [source-page: novel2.pdf-5]. Despite their differences, both characters are shaped by the challenges they face, highlighting the diverse ways individuals respond to adversity.

        Few-Shot Example 2:
        Query: Summarize the key points of a historical event.
        Intermediate: Event started in 1914 [source-page: historybook.pdf-1]. Major battles occurred in Europe [source-page: historybook.pdf-2]. Ended with treaty in 1918 [source-page: historybook.pdf-4].
        Synthesized: TL;DR: The event began in 1914, was marked by significant battles in Europe, and concluded with a treaty in 1918 [source-page: historybook.pdf-1][source-page: historybook.pdf-2][source-page: historybook.pdf-4].

        Overview: The event's origins trace back to 1914, setting off a series of major battles primarily across Europe [source-page: historybook.pdf-1][source-page: historybook.pdf-2]. The conflict involved multiple nations and resulted in widespread change. The resolution came in 1918 with the signing of a treaty, which established new political boundaries and had lasting global effects [source-page: historybook.pdf-4].

        Few-Shot Example 3 (Reference Elaboration):
        Query: Explain the process of photosynthesis.
        Intermediate: Plants use sunlight to make food [source-page: biologytext.pdf-7]. Chlorophyll captures light [source-page: biologytext.pdf-8]. Oxygen is released [source-page: biologytext.pdf-9].
        Synthesized: TL;DR: Photosynthesis enables plants to convert sunlight into food, with chlorophyll capturing light and oxygen released as a byproduct [source-page: biologytext.pdf-7][source-page: biologytext.pdf-8][source-page: biologytext.pdf-9].
        Process Details: During photosynthesis, plants absorb sunlight using chlorophyll, a pigment found in their leaves [source-page: biologytext.pdf-8]. This energy is used to convert carbon dioxide and water into glucose, providing essential nourishment for the plant [source-page: biologytext.pdf-7]. As a result of this process, oxygen is released into the atmosphere, supporting life on Earth [source-page: biologytext.pdf-9].

        Few-Shot Example 4 (Citation Extraction):
        Query: Describe a scientific concept.
        Intermediate: The concept involves energy conversion Source: sci_book.pdf - Page: 10. It produces byproducts Source: sci_book.pdf - Page: 12.
        ## Used Sources
        [source-page: sci_book.pdf-10]
        [source-page: sci_book.pdf-12]
        Synthesized: TL;DR: The concept centers on energy conversion and produces specific byproducts [source-page: sci_book.pdf-10][source-page: sci_book.pdf-12].

        Detailed Explanation: This scientific concept primarily involves the conversion of energy from one form to another, as detailed in the source material [source-page: sci_book.pdf-10]. As part of the process, certain byproducts are generated, which have implications for the environment [source-page: sci_book.pdf-12].

        Now, synthesize this intermediate answer for the query above:
        {answer}

        Structure your output as:
        - Remove any irrelevant details not related to query
        - **Introduction** : Use this markdown heading for introducing what the whole response will contain , giving a 3-4 line skimming over all topics.
        - **Detailed Sections**: Use markdown headings for expandable parts (e.g., ## Key Points, ## Process, ## Comparison, ## Applications), ensuring flow and expert tone. Include ALL details from the intermediate, reformulated for better narrative without shortening.
        -*TL;DR**: A conclusion of 3-4 sentence, concise summary of key takeaways.
        - **CRITICAL:** Preserve embedded metadata but ONLY for sources actually referenced in your output—remove or omit unused citations and metadata. Citations must be in the format [source-page: filename.pdf-pageNo] where filename.pdf is the actual PDF file with .pdf extension. If the source doesn't contain .pdf or you're uncertain about the exact PDF filename, omit that citation entirely.

        Synthesized Response: 
        ''',
        input_variables=['answer', 'query']
    )  

    synthesis_chain = synthesis_prompt | llm  # Chain the prompt with your LLM
    synthesized_text = synthesis_chain.invoke({"answer": disjoint_results['result'], "query": query}).content
    print(f"\n\nsynthesised text \n\n: {synthesized_text}\n\n")
    # Combine with sources for final return (app.py can render markdown with st.expander for sections)
    return {'result': synthesized_text}


'''
def generate_qa_pairs(sample_size=5):
    qa_pairs = []
    selected_chunks = random.sample(chunks, min(sample_size, len(chunks)))
    for chunk in selected_chunks:
        gen_prompt = PromptTemplate(
            template="Based on the following text, generate 1 high-quality question-answer pair. The answer should be detailed and directly from the text.\nText: {text}\nOutput format:\nQuestion: <question>\nAnswer: <answer>",
            input_variables=["text"]
        )
        response = llm.invoke(gen_prompt.format(text=chunk.page_content))
        lines = response.content.split('\n')
        question_line = [line for line in lines if line.startswith("Question: ")]
        answer_line = [line for line in lines if line.startswith("Answer: ")]
        if question_line and answer_line:
            question = question_line[0].replace("Question: ", "").strip()
            answer = answer_line[0].replace("Answer: ", "").strip()
        else:
            continue  # Skip if parsing fails
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    print(f" qa_pairs : {qa_pairs") 
    return qa_pairs
'''

__all__ = ["enhanced_qa_chain"]



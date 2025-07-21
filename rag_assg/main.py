from langchain.chains import RetrievalQAWithSourcesChain

# --- Simple QA With Sources chain (LangChain blog example) ------------------
qa_with_sources_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=multi_retriever,
    chain_type="stuff"  # 'stuff' concatenates docs; adjust if documents are large
)

def qa_with_sources(query: str):
    """
    Simple wrapper around LangChain RetrievalQAWithSourcesChain.
    Returns dict with 'answer' and 'sources' keys where sources is a comma-separated
    list of document identifiers (metadata 'source') used by the chain.
    """
    return qa_with_sources_chain({"question": query})

# ---------------------------------------------------------------------------
__all__ = ["enhanced_qa_chain", "qa_with_sources"] 
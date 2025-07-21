
import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from main import enhanced_qa_chain, generate_qa_pairs, llm
from langchain_core.prompts import PromptTemplate

# Generate ground truth
ground_truth = generate_qa_pairs(sample_size=5)

# Faithfulness check using LLM
def check_faithfulness(answer, sources):
    context = "\n\n".join([doc.page_content for doc in sources])
    faith_prompt = PromptTemplate(
        template="Does the following answer faithfully represent the context without adding unsupported information? Answer yes or no, and explain.\nAnswer: {answer}\nContext: {context}",
        input_variables=["answer", "context"]
    )
    response = llm.invoke(faith_prompt.format(answer=answer, context=context))
    return "yes" in response.content.lower()

# Hallucination check (simple: if faithfulness is no, then hallucinated)
# Could be more sophisticated

results = []
for gt in ground_truth:
    start_time = time.time()
    result = enhanced_qa_chain(gt["question"])
    latency = time.time() - start_time
    generated_answer = result["result"]
    sources = result["source_documents"]
    
    # Quantitative
    reference = gt["answer"].split()
    candidate = generated_answer.split()
    bleu = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gt["answer"], generated_answer)
    
    # Recall@k (check if original source is in retrieved)
    relevant_source = gt["relevant_sources"][0]["source"]
    recall = 1 if any(doc.metadata["source"] == relevant_source for doc in sources) else 0
    
    # Faithfulness
    faithful = check_faithfulness(generated_answer, sources)
    
    # Coverage
    coverage = len(set(doc.metadata.get("source") for doc in sources))
    
    # Hallucination (inverse of faithfulness for simplicity)
    hallucination = not faithful
    
    results.append({
        "question": gt["question"],
        "bleu": bleu,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "recall": recall,
        "latency": latency,
        "coverage": coverage,
        "faithful": faithful,
        "hallucination": hallucination
    })

# Compute averages
avg_metrics = {
    "avg_bleu": np.mean([r["bleu"] for r in results]),
    "avg_rouge1": np.mean([r["rouge1"] for r in results]),
    "avg_rougeL": np.mean([r["rougeL"] for r in results]),
    "avg_recall": np.mean([r["recall"] for r in results]),
    "avg_latency": np.mean([r["latency"] for r in results]),
    "avg_coverage": np.mean([r["coverage"] for r in results]),
    "faithfulness_rate": np.mean([1 if r["faithful"] else 0 for r in results]),
    "hallucination_rate": np.mean([1 if r["hallucination"] else 0 for r in results])
}

print("Evaluation Results:")
for key, value in avg_metrics.items():
    print(f"{key}: {value}")

print("\nDetailed Results:")
for res in results:
    print(res) 
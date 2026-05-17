# evals/rag_eval.py
# Custom RAG evaluator — measures the same things as RAGAS but
# uses plain Python + Groq. No dependency on RAGAS internals.
#
# Three metrics:
#
# Faithfulness: does the answer only use info from retrieved context?
# We ask an LLM judge to verify each claim in the answer against context.
# Score = claims supported by context / total claims
#
# Answer Relevancy: does the answer address the question?
# We generate 3 questions from the answer and measure how similar
# they are to the original question using embeddings.
# Score = avg cosine similarity of generated questions to original
#
# Context Recall: did retrieval find documents needed to answer?
# We check each sentence of ground truth against retrieved context.
# Score = ground truth sentences supported by context / total sentences

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import ollama
from app.rag.retriever import retrieve
from app.core.config import LLM_MODEL, GROQ_API_KEY

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)


# ── Eval dataset ──────────────────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "question":    "Are there AC buses from Hyderabad to Bangalore?",
        "ground_truth":"Yes, there are AC buses including AC Sleeper operated by MSRTC at Rs 1140.",
        "origin":      "Hyderabad",
        "destination": "Bangalore",
    },
    {
        "question":    "What is the cheapest bus from Hyderabad to Bangalore?",
        "ground_truth":"The cheapest option is a Seater bus by MSRTC at Rs 456.",
        "origin":      "Hyderabad",
        "destination": "Bangalore",
    },
    {
        "question":    "Are there evening buses from Hyderabad to Bangalore?",
        "ground_truth":"Yes, there is an evening bus departing at 18:30.",
        "origin":      "Hyderabad",
        "destination": "Bangalore",
        "time_of_day": "evening",
    },
    {
        "question":    "What buses run from Bangalore to Kochi?",
        "ground_truth":"There are Sleeper buses from Bangalore to Kochi departing at 13:30 and 16:00.",
        "origin":      "Bangalore",
        "destination": "Kochi",
    },
    {
        "question":    "What is the price of an AC Sleeper from Hyderabad to Bangalore?",
        "ground_truth":"AC Sleeper costs Rs 1140.",
        "origin":      "Hyderabad",
        "destination": "Bangalore",
        "seat_type":   "AC Sleeper",
    },
]


# ── Embedding helper ──────────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-9:
        return 0.0
    return float(np.dot(a, b) / norm)


# ── Metric 1: Faithfulness ────────────────────────────────────────────────────

def measure_faithfulness(answer: str, context: str) -> float:
    """
    Ask LLM to extract claims from answer, then verify each against context.
    Score = supported claims / total claims.

    Interview point: faithfulness detects hallucination. If the answer
    says 'bus departs at 14:00' but context only shows '18:30', that
    claim is unsupported — faithfulness score drops.
    """
    # step 1: extract claims from answer
    claims_response = llm.invoke([
        SystemMessage(content=(
            "Extract all factual claims from this text as a Python list of strings. "
            "Return ONLY the list, no explanation. "
            "Example: ['Bus departs at 18:30', 'Price is Rs 1140', 'Operator is MSRTC']"
        )),
        HumanMessage(content=f"Text: {answer}")
    ])

    import ast, re
    try:
        cleaned = re.sub(r"```[\w]*\n?", "", claims_response.content).strip()
        claims = ast.literal_eval(cleaned)
    except Exception:
        # fallback: split by sentences
        claims = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

    if not claims:
        return 1.0

    # step 2: verify each claim against context
    supported = 0
    for claim in claims:
        verify_response = llm.invoke([
            SystemMessage(content=(
                "Does the context support this claim? "
                "Reply with ONLY: YES or NO"
            )),
            HumanMessage(content=f"Context: {context}\n\nClaim: {claim}")
        ])
        if "YES" in verify_response.content.strip().upper():
            supported += 1

    score = supported / len(claims)
    return round(score, 3)


# ── Metric 2: Answer Relevancy ────────────────────────────────────────────────

def measure_answer_relevancy(question: str, answer: str) -> float:
    """
    Generate 3 questions from the answer, measure similarity to original.
    Score = avg cosine similarity of generated questions to original question.

    Interview point: if the answer is relevant, questions generated from
    it should be similar to the original question. If the answer drifted
    off-topic, generated questions will be dissimilar.
    """
    gen_response = llm.invoke([
        SystemMessage(content=(
            "Generate 3 different questions that this answer could be responding to. "
            "Return ONLY a Python list of 3 strings. No explanation."
        )),
        HumanMessage(content=f"Answer: {answer}")
    ])

    import ast, re
    try:
        cleaned = re.sub(r"```[\w]*\n?", "", gen_response.content).strip()
        generated_questions = ast.literal_eval(cleaned)
    except Exception:
        return 0.5   # neutral score if parsing fails

    if not generated_questions:
        return 0.5

    # measure similarity between original question and generated questions
    original_emb = embed(question)
    similarities = []
    for gq in generated_questions[:3]:
        gq_emb = embed(gq)
        similarities.append(cosine_similarity(original_emb, gq_emb))

    return round(float(np.mean(similarities)), 3)


# ── Metric 3: Context Recall ──────────────────────────────────────────────────

def measure_context_recall(ground_truth: str, context: str) -> float:
    sentences = [s.strip() for s in ground_truth.split(".")
                 if len(s.strip()) > 5]
    if not sentences:
        return 1.0

    supported = 0
    for sentence in sentences:
        check_response = llm.invoke([
            SystemMessage(content=(
                "Can this statement be inferred from the context? "
                "Reply with ONLY: YES or NO"
            )),
            HumanMessage(content=f"Context: {context}\n\nStatement: {sentence}")
        ])
        answer = check_response.content.strip().upper()
        if "YES" in answer:
            supported += 1

    return round(supported / len(sentences), 3)


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "="*50)
    print("CUSTOM RAG EVALUATION")
    print("="*50)
    print(f"Evaluating {len(EVAL_DATASET)} questions\n")

    all_results = []

    for i, item in enumerate(EVAL_DATASET):
        print(f"[{i+1}/{len(EVAL_DATASET)}] {item['question'][:55]}")

        # retrieve documents
        results = retrieve(
            query=item["question"],
            origin=item.get("origin"),
            destination=item.get("destination"),
            seat_type=item.get("seat_type"),
            time_of_day=item.get("time_of_day"),
            top_k=5
        )

        # build context string
        context = "\n".join([
            f"{r['operator']} | {r['seat_type']} | "
            f"Rs {r['price']} | dep {r['departure']} | "
            f"arr {r['arrival']} | {r['available']} seats"
            for r in results
        ]) if results else "No results found"


        # generate answer from context
        if results:
            answer_response = llm.invoke([
                SystemMessage(content=(
                    "Answer the question using ONLY the context. "
                    "Be concise and factual."
                )),
                HumanMessage(content=(
                    f"Context:\n{context}\n\n"
                    f"Question: {item['question']}"
                ))
            ])
            answer = answer_response.content
        else:
            answer = "No buses found matching the criteria."

        # compute three metrics
        print(f"  Computing faithfulness...")
        faithfulness = measure_faithfulness(answer, context)

        print(f"  Computing answer relevancy...")
        relevancy = measure_answer_relevancy(item["question"], answer)

        print(f"  Computing context recall...")
        recall = measure_context_recall(item["ground_truth"], context)

        result = {
            "question":        item["question"],
            "answer":          answer,
            "faithfulness":    faithfulness,
            "answer_relevancy":relevancy,
            "context_recall":  recall,
        }
        all_results.append(result)

        print(f"  Scores — faith={faithfulness:.3f} "
              f"relevancy={relevancy:.3f} recall={recall:.3f}\n")

    # aggregate scores
    avg_faith     = round(np.mean([r["faithfulness"]     for r in all_results]), 3)
    avg_relevancy = round(np.mean([r["answer_relevancy"] for r in all_results]), 3)
    avg_recall    = round(np.mean([r["context_recall"]   for r in all_results]), 3)

    print("="*50)
    print("AGGREGATE SCORES")
    print("="*50)
    print(f"  Faithfulness:     {avg_faith}")
    print(f"  Answer Relevancy: {avg_relevancy}")
    print(f"  Context Recall:   {avg_recall}")
    print(f"  Overall:          {round(np.mean([avg_faith, avg_relevancy, avg_recall]), 3)}")

    # interpret scores
    print("\nINTERPRETATION")
    print(f"  Faithfulness {avg_faith}: ", end="")
    if avg_faith >= 0.8:
        print("GOOD — answers stay within retrieved context")
    elif avg_faith >= 0.6:
        print("FAIR — some claims not grounded in context")
    else:
        print("POOR — significant hallucination detected")

    print(f"  Answer Relevancy {avg_relevancy}: ", end="")
    if avg_relevancy >= 0.8:
        print("GOOD — answers address the questions well")
    elif avg_relevancy >= 0.6:
        print("FAIR — answers partially address questions")
    else:
        print("POOR — answers are off-topic")

    print(f"  Context Recall {avg_recall}: ", end="")
    if avg_recall >= 0.8:
        print("GOOD — retrieval finds relevant documents")
    elif avg_recall >= 0.6:
        print("FAIR — retrieval misses some relevant documents")
    else:
        print("POOR — retrieval is not finding the right documents")

    # save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "aggregate": {
            "faithfulness":     avg_faith,
            "answer_relevancy": avg_relevancy,
            "context_recall":   avg_recall,
            "overall":          round(np.mean([avg_faith, avg_relevancy, avg_recall]), 3)
        },
        "per_question": all_results
    }
    Path("evals/results.json").write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to evals/results.json")

    return output


if __name__ == "__main__":
    run_evaluation()

import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
import math
from ragas.run_config import RunConfig
from backend import init_pipeline, run_query, ragas_llm, ragas_embeddings
from golden_dataset import GOLDEN_DATASET


# Instantiate with strictness=1 instead of importing the default instances
faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy(strictness=1)
context_precision = ContextPrecision()

init_pipeline()

# --- Step 1: run the graph for every question ---
collected = []
start_all = time.perf_counter()

for i, item in enumerate(GOLDEN_DATASET, 1):
    print(f"[{i}/{len(GOLDEN_DATASET)}] Running: {item['question'][:60]}...")
    result = run_query(item["question"], ground_truth=item["ground_truth"])
    collected.append({
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "answer": result["answer"],
        "contexts": result["retrieved_docs"],
        "retries": result["retry_count"],
        "latency": result["latency_log"],
    })
    print(f"    -> retries={result['retry_count']}, "
          f"faithfulness(gate)={result['eval_score'].get('faithfulness')}")

# --- Step 2: one batched RAGAS call for the full metric set ---
print("\nRunning full batched evaluation (faithfulness + relevancy + precision)...")
batch_ds = Dataset.from_dict({
    "question": [c["question"] for c in collected],
    "answer": [c["answer"] for c in collected],
    "contexts": [c["contexts"] for c in collected],
    "ground_truth": [c["ground_truth"] for c in collected],
})



full_result = evaluate(
    batch_ds,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(
        max_workers=2,
        timeout=180,
        max_retries=5,
    ),
)

for i, c in enumerate(collected):
    c["full_eval_scores"] = {k: round(v, 3) for k, v in full_result.scores[i].items()}

total_time = time.perf_counter() - start_all

# --- Step 3: aggregate report ---
def safe_avg(values):
    valid = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(valid) / len(valid) if valid else float("nan")

avg_faithfulness = safe_avg([c["full_eval_scores"].get("faithfulness") for c in collected])
avg_relevancy = safe_avg([c["full_eval_scores"].get("answer_relevancy") for c in collected])
avg_precision = safe_avg([c["full_eval_scores"].get("context_precision") for c in collected])
avg_retries = sum(c["retries"] for c in collected) / len(collected)
total_retried_queries = sum(1 for c in collected if c["retries"] > 0)

print("\n" + "=" * 50)
print("GOLDEN DATASET EVALUATION SUMMARY")
print("=" * 50)
print(f"Queries evaluated:      {len(collected)}")
print(f"Avg faithfulness:       {avg_faithfulness:.3f}")
print(f"Avg answer relevancy:   {avg_relevancy:.3f}")
print(f"Avg context precision:  {avg_precision:.3f}")
print(f"Avg retries per query:  {avg_retries:.2f}")
print(f"Queries that retried:   {total_retried_queries}/{len(collected)}")
print(f"Total wall-clock time:  {total_time:.1f}s")
print("=" * 50)

# --- Step 4: per-query detail table ---
print("\nPer-query results:")
for c in collected:
    scores = c["full_eval_scores"]
    print(f"- {c['question'][:55]:<57} "
          f"faith={scores.get('faithfulness', 0):.2f} "
          f"relev={scores.get('answer_relevancy', 0):.2f} "
          f"prec={scores.get('context_precision', 0):.2f} "
          f"retries={c['retries']}")


from deepeval import evaluate
from modules.test_cases import test_cases
from modules.eval import answer_relevancy_metric, faithfulness_metric, contextual_relevancy_metric, contextual_precision_metric, contextual_recall_metric, tool_correctness_metric, task_completion_metric, json_correctness_metric, hallucination_metric, toxicity_metric, bias_metric, summarization_metric
import matplotlib.pyplot as plt

# metrics = [answer_relevancy_metric, faithfulness_metric, contextual_relevancy_metric, contextual_precision_metric, contextual_recall_metric, tool_correctness_metric, task_completion_metric, json_correctness_metric, hallucination_metric, toxicity_metric, bias_metric, summarization_metric]
metrics = [answer_relevancy_metric, tool_correctness_metric]
# Removing any `None` values
valid_metrics = [m for m in metrics if m is not None]

results = evaluate(test_cases=test_cases, metrics=valid_metrics)

metric_scores = {
    result.metrics_data[0].name: result.metrics_data[0].score for result in results.test_results
}

print(metric_scores)

plt.figure(figsize=(12, 6))
plt.bar(metric_scores.keys(), metric_scores.values(), color='skyblue')
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha='right')
plt.title("Evaluation Scores Across Metrics")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("./images/evals_plot.png")

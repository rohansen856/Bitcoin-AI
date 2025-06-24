from deepeval import evaluate
from modules.test_cases import test_cases
from modules.eval import answer_relevancy_metric, faithfulness_metric, contextual_relevancy_metric, contextual_precision_metric, contextual_recall_metric, tool_correctness_metric, task_completion_metric, json_correctness_metric, hallucination_metric, toxicity_metric, bias_metric, summarization_metric
import matplotlib.pyplot as plt

# metrics = [answer_relevancy_metric, faithfulness_metric, contextual_relevancy_metric, contextual_precision_metric, contextual_recall_metric, tool_correctness_metric, task_completion_metric, json_correctness_metric, hallucination_metric, toxicity_metric, bias_metric, summarization_metric]
metrics = [answer_relevancy_metric, tool_correctness_metric, contextual_relevancy_metric, task_completion_metric, toxicity_metric, hallucination_metric]
# Removing any `None` values
valid_metrics = [m for m in metrics if m is not None]

results = evaluate(test_cases=test_cases, metrics=valid_metrics)
print(results)

# Collecting all metric scores across all test results
metric_scores = {}
for result in results.test_results:
    for metric_data in result.metrics_data:
        if metric_data.name not in metric_scores:
            metric_scores[metric_data.name] = []
        metric_scores[metric_data.name].append(metric_data.score)

# Averaging scores for each metric
average_metric_scores = {metric: sum(scores) / len(scores) for metric, scores in metric_scores.items()}

print(average_metric_scores)

# Plotting the metrics
plt.figure(figsize=(12, 6))
plt.bar(average_metric_scores.keys(), average_metric_scores.values(), color='skyblue')
plt.ylabel("Average Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha='right')
plt.title("Average Evaluation Scores Across Metrics")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("./images/evals_plot.png")

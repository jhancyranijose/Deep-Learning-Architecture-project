import os
import pandas as pd
import matplotlib.pyplot as plt

REPORT_DIR = "outputs/reports"
SAVE_DIR = "outputs/plots"

os.makedirs(SAVE_DIR, exist_ok=True)

classes = [
    "Healthy",
    "Mild DR",
    "Moderate DR",
    "Proliferate DR",
    "Severe DR"
]

scenarios = [
    "scenario1_reports",
    "scenario2_reports",
    "scenario3_reports",
    "scenario4_reports",
    "scenario5_reports",
    "scenario6_reports"
]

data = {cls: [] for cls in classes}

def extract_recall(report_path):
    recall_scores = {}

    with open(report_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) >= 4:
            label = " ".join(parts[:-4])

            if label in classes:
                recall = float(parts[-3])
                recall_scores[label] = recall

    return recall_scores


for scenario in scenarios:

    report_file = os.path.join(REPORT_DIR, scenario, "classification_report.txt")

    scores = extract_recall(report_file)

    for cls in classes:
        data[cls].append(scores.get(cls, 0))


df = pd.DataFrame(
    data,
    index=[
        "Scenario 1",
        "Scenario 2",
        "Scenario 3",
        "Scenario 4",
        "Scenario 5",
        "Scenario 6"
    ]
)

df.plot(kind="bar", figsize=(12,6))

plt.title("Recall Comparison per Class Across Scenarios")
plt.ylabel("Recall")
plt.xlabel("Scenario")
plt.xticks(rotation=0)
plt.ylim(0,1)
plt.legend(title="Class")

plt.tight_layout()

save_path = os.path.join(SAVE_DIR,"recall_comparison.png")

plt.savefig(save_path, dpi=300)
plt.show()

print("Saved graph to:", save_path)
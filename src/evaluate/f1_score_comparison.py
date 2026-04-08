import os
import pandas as pd
import matplotlib.pyplot as plt

# location of scenario reports
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

def extract_f1(report_path):
    f1_scores = {}
    with open(report_path,"r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) >= 4:
            label = " ".join(parts[:-4])

            if label in classes:
                f1 = float(parts[-2])
                f1_scores[label] = f1

    return f1_scores


for scenario in scenarios:

    report_file = os.path.join(REPORT_DIR, scenario, "classification_report.txt")

    scores = extract_f1(report_file)

    for cls in classes:
        data[cls].append(scores.get(cls,0))


df = pd.DataFrame(data, index=scenarios)

# plot
df.plot(kind="bar", figsize=(10,6))

plt.title("F1-score Comparison per Class Across Scenarios")
plt.ylabel("F1-score")
plt.xlabel("Scenario")
plt.xticks(rotation=0)
plt.legend(title="Class")
plt.tight_layout()

save_path = os.path.join(SAVE_DIR,"f1_score_comparison.png")
plt.savefig(save_path)

print("Saved graph to:", save_path)
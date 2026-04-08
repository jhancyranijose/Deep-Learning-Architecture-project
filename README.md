<<<<<<< HEAD

# Diabetic Retinopathy Classification using EfficientNet

## Overview

This project investigates **deep learning architectures for diabetic retinopathy (DR) classification** using retinal fundus images from a Kaggle dataset.  
Multiple experimental **scenarios** were implemented to evaluate how different model architectures, loss functions, image resolutions, and preprocessing strategies affect performance.

The study compares **EfficientNetB3 and EfficientNetB4 architectures** and evaluates improvements using focal loss, class weighting, and explainability methods.

The classification task consists of predicting **five DR severity levels**:

- Healthy
- Mild DR
- Moderate DR
- Severe DR
- Proliferative DR

# Dataset

The dataset was obtained from Kaggle and contains **2,750 retinal fundus images**.

### Class Distribution

- Healthy: **1000**
- Mild DR: **370**
- Moderate DR: **900**
- Severe DR: **190**
- Proliferative DR: **290**

The dataset is **imbalanced**, which creates challenges for correctly detecting minority classes such as Severe DR and Proliferative DR.

# Experimental Scenarios

Six experimental scenarios were implemented to evaluate different deep learning configurations.

| Scenario | Model Configuration | Accuracy |
|--------|---------------------|---------|
| Scenario 1 | EfficientNetB3 + improved head + class weighting + two-stage fine-tuning | 65.5% |
| Scenario 2 | EfficientNetB3 + 300×300 resolution + capped class weights + label smoothing | 59.3% |
| Scenario 3 | EfficientNetB3 + 300×300 + focal loss + stronger fine-tuning | 67.6% |
| Scenario 4 | EfficientNetB4 + 380×380 + focal loss + stronger fine-tuning | **68.7% (Best)** |
| Scenario 5 | EfficientNetB4 + 380×380 + severity-aware cost-sensitive loss | 63.3% |
| Scenario 6 | EfficientNetB4 + retinal preprocessing + lighter augmentation | 49.5% |

The **best-performing model** was Scenario 4 using EfficientNetB4 with focal loss.

# Best Model Configuration

**Architecture:** EfficientNetB4  
**Input Resolution:** 380×380  
**Loss Function:** Focal Loss  
**Training Strategy:** Two-stage fine-tuning  

### Performance

- Accuracy: **68.7%**
- Improved detection of Severe and Proliferative DR
- More balanced performance across classes

# Explainability

Explainability techniques were applied to understand model predictions.

Methods used:

- **Grad-CAM heatmaps**
- **Overlay visualizations**
- Feature attention visualization

These techniques highlight **important retinal regions** used by the model during prediction.

Explainability outputs are stored in:

```

xai_outputs/

```

# Project Structure

```

DIABETIC_RETINOPATHY
│
├── data
│   └── DR_Image
│       ├── Healthy
│       ├── Mild DR
│       ├── Moderate DR
│       ├── Proliferate DR
│       └── Severe DR
│
├── outputs
│   ├── checkpoint
│   ├── plots
│   └── reports
│
├── src
│   ├── evaluate
│   │   └── evaluate.py
│   │
│   ├── explainability
│   │   └── explainability.py
│   │
│   └── model
│       ├── scenario1.py
│       ├── scenario1_model.keras
│       ├── scenario2.py
│       ├── scenario2_model.keras
│       ├── scenario3.py
│       ├── scenario3_model.keras
│       ├── scenario4.py
│       ├── scenario4_model.keras
│       ├── scenario5.py
│       ├── scenario5_model.keras
│       ├── scenario6.py
│       └── scenario6_model.keras
│
└── xai_outputs

````

# How to Run

### Train a Scenario Model

Example:

```bash
python src/model/scenario4.py
````

### Evaluate the Model

```bash
python src/evaluate/evaluate.py
```

This will generate:

* classification report
* confusion matrix
* evaluation plots

Outputs will be saved in:

```
outputs/reports
outputs/plots
```

### Run Explainability

```bash
python src/explainability/explainability.py
```

Generated explainability outputs will be stored in:

```
xai_outputs/
```

# Key Findings

* **Healthy images were classified consistently well across all scenarios**
* **Moderate DR often became the dominant predicted class in weaker models**
* **Focal loss significantly improved minority-class learning**
* **EfficientNetB4 improved feature extraction due to higher resolution**
* **Retinal preprocessing did not improve performance in this study**

The main challenge remained distinguishing between:

* Moderate DR
* Severe DR
* Proliferative DR

due to **visual similarity of retinal lesions across severity levels**.

# Limitations

* Dataset imbalance affects minority-class performance
* Limited dataset size for training large deep learning architectures
* Difficulty distinguishing adjacent DR severity stages
* Model accuracy still below clinical diagnostic requirements

# Future Work

Possible improvements include:

* Training on **larger DR datasets (e.g., EyePACS)**
* Applying **attention mechanisms or vision transformers**
* Using **ensemble models**
* Implementing **ordinal classification methods**
* Expanding **explainability analysis**
=======
# Deep-Learning-Architecture-project
Diabetic Retinopathy Classification 


This is a group project by me and my collegue. you can also find me as a contibutor for this at https://github.com/ami-04/diabetic-retinopathy-classification

The other Info can be found in the Readme1.md in this repository
>>>>>>> dd114100a30f977543f0255f560b7cedb26437ed

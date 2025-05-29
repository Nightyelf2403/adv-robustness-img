# 🔐 adv-robustness-img

> **Evaluating Adversarial Robustness of Pretrained Image Classifiers**

This project explores the adversarial robustness of pretrained deep learning models in image classification, using a variety of advanced attack techniques and a lightweight defense strategy.

## 📦 Dataset

We use the **Bean Leaf Classification** dataset sourced from the [GitHub repository by @trizkynoviandy](https://github.com/trizkynoviandy/bean-leaf-classification), which contains annotated images of healthy and diseased bean leaves. It provides a clean and interpretable image classification task ideal for adversarial robustness evaluation.

## 🧠 Models Used

- `ResNet-18` (Pretrained on ImageNet)
- `EfficientNet-B0` (Pretrained on ImageNet)

Both models are fine-tuned on the bean leaf dataset and tested under various adversarial scenarios.

## ⚔️ Adversarial Attacks

We apply four advanced adversarial attacks to benchmark robustness:

1. **Square Attack**  
   A black-box, query-efficient attack based on pixel perturbations.

2. **Elastic-Net Attack (EAD)**  
   A white-box attack using L1 regularization to generate sparse perturbations.

3. **ZOO Attack (Zeroth Order Optimization)** ✅ *(used instead of Spatial Transformation Attack)*  
   A black-box optimization-based evasion attack requiring only output probabilities.

4. **FGSM (Fast Gradient Sign Method)**  
   Baseline white-box attack for comparison.

## 🛡️ Defense Technique

**Feature Squeezing**  
A lightweight input preprocessing method that reduces color bit depth and applies smoothing to diminish adversarial noise while preserving key features.

## 📊 Evaluation Metrics

- **Top-1 Accuracy Drop**
- **Attack Success Rate**
- **Visual Perturbation (L2 Norm / L∞)**
- **Robustness across attacks and models**

## 🖼️ Visualizations

Visual comparisons are provided between:
- Clean images
- Adversarial examples
- Defended outputs (Feature Squeezed)

See the `notebooks/` directory and generated plots for side-by-side visuals.


## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/adv-robustness-img.git
cd adv-robustness-img

# Install dependencies
pip install -r requirements.txt

# Run attack evaluations
python attacks/zoo_attack.py

# Launch Jupyter Notebook to view results
jupyter notebook notebooks/evaluation.ipynb
````

## 📌 Notes

* The **ZOO Attack** is used in place of the Spatial Transformation Attack to provide diversity and test black-box vulnerabilities more effectively.
* The dataset can be downloaded from the linked GitHub repo or directly placed in the `data/` folder.

## 🧪 Results Summary

| Model           | Attack | Accuracy ↓ | Perturbation Norm |
| --------------- | ------ | ---------- | ----------------- |
| ResNet-18       | ZOO    | 32.4%      | \~1.7 (L2 norm)   |
| EfficientNet-B0 | EAD    | 28.1%      | \~1.2 (L1 norm)   |

(See full results and plots in `notebooks/`)

## 📚 References

* [ZOO Attack Paper](https://arxiv.org/abs/1708.03999)
* [Feature Squeezing Defense](https://arxiv.org/abs/1704.01155)
* [Bean Leaf Dataset Repo](https://github.com/trizkynoviandy/bean-leaf-classification)

---

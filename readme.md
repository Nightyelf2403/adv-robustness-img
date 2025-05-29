# 🔐 adv-robustness-img

> **Evaluating Adversarial Robustness of Pretrained Image Classifiers on Bean Leaf Dataset**

This project investigates how well pretrained deep learning image classifiers can withstand adversarial attacks. It benchmarks different attack methods—including black-box and white-box attacks—on a clean agricultural dataset of bean leaf images. The study also evaluates a lightweight defense strategy.

---

## 📦 Dataset

We use the **Bean Leaf Classification** dataset, originally from [@trizkynoviandy](https://github.com/trizkynoviandy/bean-leaf-classification), containing categorized images of bean plant leaves (healthy, angular leaf spot, and bean rust). The dataset is loaded and used from Google Drive within a Colab notebook.

---

## 🧠 Models

- `ResNet-18` (Keras, pretrained on ImageNet)
- `EfficientNet-B0` (Keras, pretrained on ImageNet)

These models were fine-tuned on the bean leaf dataset using TensorFlow/Keras and later evaluated under adversarial scenarios.

---

## ⚔️ Adversarial Attacks (ART Library)

Adversarial attacks are implemented using the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) by IBM. The following attacks were tested:

1. **FGSM (Fast Gradient Sign Method)** — A baseline white-box gradient-based attack.
2. **Square Attack** — A black-box, query-efficient pixel-level perturbation.
3. **Elastic-Net Attack (EAD)** — A white-box attack using L1-based sparse perturbations.
4. **ZOO Attack (Zeroth Order Optimization)** ✅ *(used instead of Spatial Transformation Attack)* — A black-box attack using gradient estimation with only access to model output probabilities.

---

## 🛡️ Defense Strategy

**Feature Squeezing**  
A simple yet effective defense that reduces the model's input complexity by reducing bit-depth or applying smoothing filters to mitigate the impact of adversarial noise.

---

## 📊 Evaluation

Evaluation metrics include:

- Accuracy drop on adversarial examples
- Perturbation strength (L1, L2 norms)
- Confusion matrix plots for clean vs adversarial samples
- Visual inspection of perturbed vs. original images

All visualizations are generated using `Matplotlib` and `Seaborn`.

---

## 🧪 Setup & Execution

### ⚙️ Environment

- Python 3.x (Google Colab recommended)
- TensorFlow 2.x / Keras
- IBM ART (Adversarial Robustness Toolbox)
- Matplotlib, Seaborn, NumPy, Pandas

### ▶️ Running the Notebook

1. Open `Final_487 (2).ipynb` in Google Colab.
2. Mount your Google Drive when prompted.
3. Ensure your Drive contains:
   - `train.zip`
   - `validation.zip`
   - `test.zip`

4. Run all cells to train the model, generate adversarial examples, apply defenses, and visualize performance.

---

## ✅ Notable Decisions

- 🔄 Replaced **Spatial Transformation Attack** with **ZOO Attack** to incorporate black-box testing and diversify attack vectors.
- 🧠 Models are trained using TensorFlow/Keras rather than PyTorch.
- 📁 Data is loaded directly from Google Drive ZIPs within Colab.

---

## 📌 References

- [ZOO Attack Paper](https://arxiv.org/abs/1708.03999)
- [Feature Squeezing Defense](https://arxiv.org/abs/1704.01155)
- [Bean Leaf Dataset](https://github.com/trizkynoviandy/bean-leaf-classification)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

---




# 🔍 HTHTA-ViT++: An Explainable and Efficient Vision Transformer with Hierarchical GRU-Guided Token Attention

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ViT](https://img.shields.io/badge/Backbone-ViT--B/16-purple)
![Status](https://img.shields.io/badge/Status-Preprint-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **A novel ViT-based hybrid architecture using BiGRU + attention pooling + CLS-fusion to improve interpretability and accuracy on real-world image datasets.**

---

## 🧠 Abstract

HTHTA-ViT++ integrates **Vision Transformers (ViT)** with a **Bidirectional GRU**, **interpretable attention pooling**, and **hierarchical CLS-token fusion** to enhance:
- Sequential token modeling
- Per-token attention visualizations
- Classification performance

It outperforms state-of-the-art models on multiple datasets including CIFAR-100 (+4.1%) and Tiny-ImageNet (+6.3%), while reducing FLOPs by 13%.

---

## 🧩 Key Contributions

- ✅ **BiGRU Token Sequencing** for capturing spatial relationships between image patches
- ✅ **Multi-head Attention Pooling** to visualize class-specific image regions
- ✅ **CLS Token Fusion** combining global and local cues
- ✅ **Focused Attention Percentage (FAP)**: 78.3% (↑ over ViT-B/16 by +24.6%)
- ✅ High interpretability with efficient computation (99.3M params)

---

---

## 📊 Benchmark Results

| Model           | CIFAR-10 | CIFAR-100 | Tiny-ImageNet | Intel    | Params (M) | FLOPs (G) |
| --------------- | -------- | --------- | ------------- | -------- | ---------- | --------- |
| ViT-B/16        | 96.5     | 84.6      | 76.8          | 93.1     | 86.0       | 17.6      |
| ConvNeXt-B      | 98.1     | 89.2      | 82.6          | 95.8     | 88.6       | 15.4      |
| **HTHTA-ViT++** | **98.7** | **93.3**  | **88.9**      | **97.9** | **99.3**   | **15.3**  |

---

## 🧪 Datasets Used

* **CIFAR-10** (50k train / 10k test)
* **CIFAR-100** (100 fine-grained classes)
* **Tiny-ImageNet** (64×64 images, 200 classes)
* **Intel Scene Classification** (6 scene types)

---

## 🧠 Attention Pooling Formula

```python
ei,j = vTj * tanh(Wj * hi + bj)
αi,j = softmax(ei,j)
cj = Σ αi,j * hi
```

* `hi`: BiGRU output
* `cj`: attention vector per head `j`

---

## 🎯 Focused Attention Percentage (FAP)

Measured via Grad-CAM overlap:

* HTHTA-ViT++: **78.3%**
* ViT-B/16: 53.7%
* Swin-B: 61.2%

---

## 🔬 Ablation on CIFAR-100

| Model Variant             | Accuracy (%) | FAP (%) |
| ------------------------- | ------------ | ------- |
| ViT-Base                  | 89.2         | 51.3    |
| + BiGRU                   | 90.8 (+1.6)  | 63.1    |
| + Attention Pooling       | 92.1 (+1.3)  | 72.4    |
| + CLS Fusion (Full Model) | 93.3 (+1.2)  | 76.5    |

---

## 🏁 Training Details

* Optimizer: AdamW (LR: `2e-5`)
* Batch Size: 32
* Epochs: 30
* Scheduler: Cosine Decay
* Hardware: 4× A100 (FP16)

---



## 📄 License

This repository is released under the [MIT License](LICENSE).

````

---








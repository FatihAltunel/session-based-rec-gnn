# Session-Based Recommendation System via Ultimate SR-GNN

This repository features a state-of-the-art **Session-Based Recommendation System (SBRS)** implemented in PyTorch, leveraging **Gated Graph Neural Networks (GGNN)** and a customized multi-modal feature fusion mechanism. The architecture is trained on the real-world **Retail Rocket E-commerce Dataset** to predict anonymous user interactions, effectively mitigating the classic e-commerce "Cold-Start" dilemma.

## 🚀 Key Engineering Contributions

* **Chronological Validation Pipeline:** Replaced standard random splits with strict temporal partitioning (7 days for validation, 7 days for testing) to prevent data leakage and evaluate real-time deployment dynamics.
* **Sliding Window Data Augmentation:** Implemented a robust sequence expansion pipeline (`augment=True`), translating evolving user sessions (e.g., `[1, 2, 3]`) into multi-step supervised contexts (`([1], 2)`, `([1, 2], 3)`), heavily expanding the model's training capacity.
* **Contextual Feature Fusion:** Advanced the baseline graph architecture by embedding item categories and specific interaction event weights (`view=1`, `addtocart=2`, `transaction=3`). The local graph hidden states are merged with metadata context through an explicit embedding injection layer.
* **Self-Attention Masked Aggregation:** Utilized a self-attention layer stacked over the GGNN outputs, filtering out noisy/accidental clicks while prioritizing immediate, high-weight contextual intents.

## 🛠️ Tech Stack & Advanced Paradigms

* **Framework:** PyTorch (Core deep learning model development), Pandas, NumPy, Scikit-Learn.
* **Graph Mechanics:** Bidirectional information propagation via `torch.bmm` matrix multiplication over local session adjacency matrices.
* **Regularization & Optimization:** Driven by `AdamW` optimizer (weight decay = $1\times10^{-4}$), Cross-Entropy with `Label Smoothing (0.1)` to combat overfitting, and dynamic learning rate scheduling via `ReduceLROnPlateau`.

## 📊 Architecture & Dynamics

### Model Mathematical Logic

The sequence of clicks is converted into a directed session graph $G = (V, E)$. Node updates are driven by a gated update function mirroring Recurrent Gated Units:

$$a_t^i = A_{i,:} \left[ v_1^{t-1}, \dots, v_n^{t-1} \right]^T \mathbf{H} + \mathbf{b}$$

The final recommendation profile dynamically fuses the enhanced local representation $s_{local}$ with the globally weighted attention intent vector $s_{global}$.

---

## 📈 Performance Benchmarks (Retail Rocket Dataset)

Through aggressive data preprocessing and state-of-the-art deep learning optimization, the **Ultimate Fusion SR-GNN** substantially outperformed the standard sequential model baseline.

| Architectural Variant | Advanced Features Enabled | Val Recall@20 | Test Recall@20 |
| :--- | :--- | :--- | :--- |
| **Baseline SR-GNN** | Standard Graph + Target | 40.96% | 48.66% |
| **Ultimate Fusion SR-GNN** | Augmentation + Category Fusion + Event Injection | **53.67%** | **51.39%** |

### Training Highlights:
* Early stopping safely triggered at Epoch 20 to preserve optimal generalization boundaries.
* The explicit addition of Label Smoothing and Dropout (0.4) stabilized validation convergence against massive item vocabulary scale complexities.

## 📁 Installation & Modular Usage

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt

2.Run the end-to-end production pipeline:
python src/train.py
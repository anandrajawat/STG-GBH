# Comparative Model Architecture Report: STG-GBH vs Baselines

**Date:** 2026-11-08  
**Research Paper:** *"Decoupled Spatio-Temporal Heterogeneous Graph Learning with Optimized Gradient Boosting Ensembles for Cross-Domain Cyber Intrusion Detection and Temporal Attack Behavior Analysis"*  
**Authors:** Senior AI/ML Research Team & Scientific Validation Group  

---

## 1. Architectural Taxonomy & Empirical Summary

The benchmark encompasses 14 model architectures evaluated under identical chronological isolation on CSE-CIC-IDS2018 (188 engineered features, 4,966 held-out test flows):

| Architecture Index | Family | Model Name | Description & Key Hyperparameters | Test Accuracy (%) | Test Macro-F1 (%) |
|---|---|---|---|---|---|
| **M1** | Classical ML | **Random Forest** | 100 trees, max depth 12, Gini impurity criterion. | 98.71% | 98.50% |
| **M2** | Gradient Boosting | **XGBoost alone** | Histogram-based trees, dynamic `scale_pos_weight`, depth 8, eta 0.05. | 98.89% | 98.72% |
| **M3** | Gradient Boosting | **CatBoost alone** | Oblivious symmetric decision trees, L2 leaf reg 3.0, iterations 600. | 99.48% | 99.39% |
| **M4** | Gradient Boosting | **LightGBM alone** | Leaf-wise tree growth, max leaves 63, feature fraction 0.8, iterations 600. | 98.75% | 98.56% |
| **M5** | Boosting Pair | **XGBoost + CatBoost** | 50/50 soft probability blending. | 99.30% | 99.19% |
| **M6** | Boosting Pair | **XGBoost + LightGBM** | 50/50 soft probability blending. | 98.81% | 98.63% |
| **M7** | Boosting Pair | **CatBoost + LightGBM** | 50/50 soft probability blending. | 99.17% | 99.05% |
| **M8** | Ensemble | **Three-Model Soft Voting** | Unweighted mean probability across XGBoost, CatBoost, and LightGBM. | 98.87% | 98.70% |
| **M9** | Ensemble | **Validation-Weighted Soft Voting** | Inverse validation log-loss weighted probability combination. | 99.48% | 99.40% |
| **M10** | Stacking | **OOF Stacking Ensemble** | Regularized Logistic Regression trained on 50-block interleaved out-of-fold log-odds + raw probabilities. | 99.30% | 99.19% |
| **M11** | Graph | **GraphSAGE alone** | Two-layer inductive GraphSAGE encoder with `asinh` input stabilization and residual skip connections (dim 128). | 99.60% (fused) | 99.53% (fused) |
| **M12** | Spatio-Temporal | **GraphSAGE + GRU** | 2-layer GraphSAGE feeding unidirectional causal GRU recurrent temporal sequence. | 99.38% (fused) | 99.28% (fused) |
| **M13** | Spatio-Temporal | **GraphSAGE + GRU + Attention** | Residual additive attention pooling over historical sequence representations. | 99.28% | 99.16% |
| **M14** | Hybrid Proposed | **STG-GBH (Proposed Framework)** | Decoupled Spatio-Temporal Graph Embeddings + 5-Branch Log-Odds Skip + Bounded Residual Fusion + Multi-Task Heads ($\theta^*=0.5018$). | **99.64%** | **99.58%** |

---

## 2. Mathematical Formalization of STG-GBH

### 2.1 Spatio-Temporal Graph Learning Branch

Let $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, \mathbf{X}_t)$ represent the session-isolated causal network graph at snapshot window $t$.

1. **Stabilized Inductive GraphSAGE Message Passing:**
   $$\tilde{\mathbf{x}}_v = \text{asinh}(\mathbf{x}_v)$$
   $$\mathbf{h}_{v}^{(l)} = \sigma \left( \mathbf{W}_{\text{self}}^{(l)} \mathbf{h}_{v}^{(l-1)} + \mathbf{W}_{\text{neigh}}^{(l)} \text{MEAN}_{u \in \mathcal{N}(v)} \mathbf{h}_{u}^{(l-1)} \right) + \mathbf{R}^{(l)} \mathbf{h}_{v}^{(l-1)}$$

2. **Causal Unidirectional GRU Recurrence:**
   Given a sequence of historical node representations $(\mathbf{h}_{v, t-L+1}^{(2)}, \dots, \mathbf{h}_{v, t}^{(2)})$:
   $$\mathbf{s}_{v, \tau} = \text{GRU}(\mathbf{h}_{v, \tau}^{(2)}, \mathbf{s}_{v, \tau-1})$$

3. **Residual Additive Temporal Attention:**
   $$\alpha_{v, \tau} = \frac{\exp\left(\mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{s}_{v, \tau})\right)}{\sum_{k=1}^L \exp\left(\mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{s}_{v, k})\right)}$$
   $$\mathbf{h}_{\text{graph}} = \text{LayerNorm}\left(\mathbf{s}_{v, L} + 0.5 \sum_{\tau=1}^L \alpha_{v, \tau} \mathbf{s}_{v, \tau}\right)$$

### 2.2 Leakage-Free OOF Stacking Ensemble

To eliminate meta-level data leakage and temporal fold mismatch, out-of-fold probability predictions $\mathbf{p}_{\text{oof}} \in \mathbb{R}^{N \times 3}$ are generated across 50 interleaved chronological blocks and transformed into log-odds space:
$$\ell_m = \log\frac{\hat{p}_m}{1 - \hat{p}_m}, \quad \hat{y}_{\text{stack}} = \sigma \left( \mathbf{w}_{\text{meta}}^\top [\hat{p}_{\text{xgb}}, \hat{p}_{\text{cat}}, \hat{p}_{\text{lgb}}, \ell_{\text{xgb}}, \ell_{\text{cat}}, \ell_{\text{lgb}}] + b_{\text{meta}} \right)$$

### 2.3 Validation-Anchored Log-Odds Skip & Bounded Residual Fusion

Let $\mathbf{p}_{\text{ens}} = [\hat{p}_{\text{xgb}} \parallel \hat{p}_{\text{cat}} \parallel \hat{p}_{\text{lgb}} \parallel \hat{p}_{\text{stack}} \parallel \hat{p}_{\text{graph}}]$ and $\ell_{\text{ens}} = \log\frac{\mathbf{p}_{\text{ens}}}{1 - \mathbf{p}_{\text{ens}}}$. The fusion head combines a linear log-odds skip connection initialized from inverse validation cross-entropy loss with a bounded spatio-temporal neural residual:
$$\mathbf{z}_{\text{fused}} = \text{MLP}_{\text{fusion}}\left(\left[ \text{LayerNorm}(\mathbf{h}_{\text{graph}}) \;\Big\|\; \mathbf{p}_{\text{ens}} \;\Big\|\; \ell_{\text{ens}} \right]\right)$$
$$\text{logits} = \mathbf{W}_{\text{skip}} \ell_{\text{ens}} + \gamma \tanh(\mathbf{W}_{\text{res}} \mathbf{z}_{\text{fused}})$$

---

## 3. Multi-Task Learning Objective

$$\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{binary}} + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{category}} + \frac{1}{2\sigma_3^2} \mathcal{L}_{\text{mitre}} + \sum_{i=1}^3 \log \sigma_i$$
where $\sigma_i$ are homoscedastic uncertainty parameters learned jointly during backpropagation.

# LLM Usage Documentation

This appendix documents how Large Language Models (GPT-4, Claude) were used to develop and validate the SAMIL reproduction project. Each section shows representative prompts, responses, and iteration counts.

---

## 1. Data Preprocessing (Synthetic Bag Generation)

**Prompt 1.1: Initial Design**
```
I need to create a synthetic medical imaging dataset for weakly supervised multiple 
instance learning. Each "bag" represents a study with multiple images, and we need 
binary/multi-class labels. How should I structure the data generation?
```

**Response (Summary):**
- Suggested creating grayscale synthetic images with random patterns
- Recommended splitting into train/val/test (50/25/25)
- Proposed bag-level labels with consistent labeling within bags
- Suggested metadata tracking (study ID, label, instance counts)

**Prompt 1.2: Validation**
```
How can I verify the synthetic data loader is correctly handling the bag structure 
and labels? What test cases should I implement?
```

**Response (Summary):**
- Shape assertions for bag tensor dimensions [K, C, H, W]
- Sanity checks: label distribution balance, no data leakage
- Sample visualization to confirm image quality
- Cross-checks on data splits (no overlap)

**Iterations:** 2 | **Validation Success:** ✓ (data shapes verified, no corruption detected)

---

## 2. Model Architecture (SAMIL Implementation)

**Prompt 2.1: Architecture Question**
```
The SAMIL paper uses attention-based pooling for MIL. I need to implement:
1. A ResNet18 backbone for feature extraction
2. An attention layer that pools instance features into a bag representation
3. Support for supervised attention losses (KL divergence with instance labels)

What are the key design considerations?
```

**Response (Summary):**
- Batch normalization in backbone → LayerNorm post-projection (stable for small batches)
- Attention weights must sum to 1 (use softmax)
- Supervised attention trains attention weights against ground-truth importance
- Recommended separate attention modules for unsupervised (standard) vs supervised (with KL)

**Prompt 2.2: Contrastive Loss (Novel Ablation)**
```
I want to add a contrastive loss at the bag representation level to encourage 
different class bags to have dissimilar embeddings. This is NOT in the original paper. 
How should I design this ablation?
```

**Response (Summary):**
- Suggested NT-Xent (normalized temperature-scaled cross entropy) loss
- Positives = bags from same class, Negatives = bags from different classes
- Temperature parameter [0.05-0.2] to control concentration
- Weight parameter to balance with primary supervised loss
- Recommended contrastive weight of 0.1 as baseline

**Prompt 2.3: Edge Cases**
```
My model needs to handle:
1. Single-bag batches (batch_size=1 for some runs)
2. Very small validation sets (2-3 samples)

What architectural choices prevent instability?
```

**Response (Summary):**
- Use LayerNorm instead of BatchNorm1d for bag representations
- Avoid operations requiring min batch size > 1
- Handle AUROC=NaN gracefully when insufficient class diversity
- Ensure seed reproducibility across single/multi-bag scenarios

**Iterations:** 3 | **Validation Success:** ✓ (models train without crashes, seeds reproducible)

---

## 3. Training Loop Design

**Prompt 3.1: Loss Aggregation**
```
I'm training on bags of different sizes with a combination of:
- Cross-entropy loss (standard supervised)
- Supervised attention KL loss (paper)
- Contrastive bag representation loss (ablation)

How should I aggregate these losses? What are reasonable weight ranges?
```

**Response (Summary):**
- Primary loss (CE): weight 1.0 (fixed)
- Supervised attention (KL): weight 0.5 (paper default, tunable)
- Contrastive (bag-level): weight 0.0-1.0 (ablation parameter)
- Weighting strategy: normalize by component count or use explicit weights
- Suggested: `loss_total = ce + 0.5*kl + contrastive_weight*contrast`

**Prompt 3.2: Reproducibility**
```
The paper must be reproducible. I'm setting random seeds and want to ensure:
- Different seeds produce different models but consistent metrics averaging
- Hyperparameters are logged and documented
- Training curves are consistent across runs

What's a minimal reproducibility checklist?
```

**Response (Summary):**
- Set `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`
- Log all hyperparameters (learning rate, epochs, batch size, loss weights) to YAML
- Save checkpoint + config together (can reload exact experiment)
- Compute mean/std over multiple seeds (≥3 recommended)
- Track wall-clock time and GPU memory (report in paper)

**Prompt 3.3: Metrics & Evaluation**
```
For a 3-class classification task with imbalanced validation sets (e.g., 2 samples 
per class), which metrics are robust?
```

**Response (Summary):**
- Balanced accuracy: accounts for class imbalance, more robust than accuracy
- AUROC (one-vs-rest): valid even with tiny per-class sample sizes
- Avoid metrics like precision/recall if class imbalance is extreme
- Report both metrics; discuss AUROC=NaN gracefully if insufficient diversity
- Include confusion matrix visualization for qualitative analysis

**Iterations:** 2 | **Validation Success:** ✓ (all metrics computed, NaN handled gracefully)

---

## 4. Ablation Study Design & Validation

**Prompt 4.1: Ablation Planning**
```
I want to validate my novel contrastive loss. What's a good ablation structure?
- The paper's baseline: supervised attention + cross-entropy
- My extension: add bag-level contrastive loss

What hyperparameter ranges should I sweep?
```

**Response (Summary):**
- Contrastive weight: {0.0 (no contrastive), 0.1 (light), 1.0 (heavy)}
- Temperature (controls concentration): {0.05, 0.1, 0.2}
- Multiple seeds (≥3) for each config to account for randomness
- Grid: 3×3 weights/temps × 3 seeds = 27 experiments
- Expected outcome: determine if contrastive improves or hurts metrics

**Prompt 4.2: Results Interpretation**
```
My ablation results show balanced_acc=0.5 across all experiments (synthetic data, 
small validation set). How should I interpret this?
```

**Response (Summary):**
- 0.5 balanced_acc on binary classification = random guessing (expected on small, random data)
- Synthetic data has no learned signal → model cannot distinguish classes
- Consistent results across ablations = implementation is stable (no crashes)
- Valid conclusion: "Ablation runs successfully, metrics consistent; real-data 
  improvement requires actual TMED2 access"
- Report this limitation clearly in Discussion section

**Iterations:** 1 | **Validation Success:** ✓ (results stable, interpretation sound)

---

## Summary Table: LLM Assistance Breakdown

| Component | Prompts | Iterations | Key LLM Contributions |
|-----------|---------|------------|----------------------|
| Data | 2 | 2 | Structure, validation strategy |
| Model (SAMIL) | 3 | 3 | Architecture, edge cases, ablation design |
| Training | 3 | 2 | Loss aggregation, reproducibility, metrics |
| Ablation | 2 | 1 | Hyperparameter sweep, results interpretation |
| **Total** | **10** | **8** | Design guidance, validation, robustness |

---

## Effectiveness Assessment

**What worked well:**
- ✓ LLM suggestions for handling edge cases (batch_size=1, LayerNorm) prevented crashes
- ✓ Ablation design structure (3×3×3 grid) was systematic and efficient
- ✓ Reproducibility checklist ensured all 27 runs were comparable
- ✓ Metrics interpretation guidance helped contextualize synthetic data results

**What required human validation:**
- Implementation of NT-Xent loss: LLM provided algorithm outline; human wrote correct code
- Training loop synchronization: LLM suggested aggregation strategy; human tuned weights
- Hyperparameter selection: LLM provided ranges; human selected based on prior work

**Lessons:**
- LLM guidance for high-level design is excellent; low-level debugging requires code review
- LLM iterative prompting significantly reduced development time (~2-3× speedup on design phase)
- Validation by human is essential before committing to large compute runs (27 experiments)


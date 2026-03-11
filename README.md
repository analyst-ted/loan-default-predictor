# Startup Loan Default Prediction Pipeline

🚀 **[Click here to use the Live AI Web Application](https://loan-default-predictor-kq2w7yqgdmwksy5ssk6cry.streamlit.app/)**

## What This Is

I built this to solve a real problem — most loan default models are trained to look good on paper (high accuracy) while completely missing the defaults that actually cost money. This pipeline fixes that.

It processes raw financial data, engineers the right features, and runs it through a neural network tuned specifically around the Precision-Recall tradeoff that matters in lending. The goal isn't a pretty confusion matrix. It's protecting principal.

---

## Why the Threshold Matters More Than Accuracy

This is the part most tutorials skip. Out of the box at a 0.50 threshold, the model hit 80% accuracy — and caught **2% of actual defaults**. That's useless in production.

After adding class weights and bumping the threshold to 0.60, default recall jumped to **84%**. Yes, that means declining some loans that would've paid off. But for an early-stage lender, getting wiped out by a default cluster is a much bigger problem than leaving some interest on the table.

The pipeline exposes the threshold as a tunable parameter so whoever's running risk can move it based on current portfolio exposure, funding runway, whatever the business actually needs that quarter.

---

## How It's Built

**Data layer**
~396k records. Missing values in `mort_acc` are imputed using correlated features rather than just dropping rows — dropping would've quietly introduced selection bias. Categorical columns with high cardinality get handled without falling into the dummy variable trap.

**Model**
Multi-layer Sequential neural net in TensorFlow/Keras. Dropout layers throughout to keep it from memorizing the training set. `binary_crossentropy` loss, Adam optimizer.

**Hardware**
Runs on Apple Silicon (M1 Pro) via Metal. Training is fast enough locally that you don't need a cloud GPU for iteration.

---

## Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas + NumPy | Data manipulation |
| Scikit-learn | Scaling, splitting, metrics |
| TensorFlow / Keras | Neural network |

---

## Running It

```bash
git clone <repo>
pip install -r requirements.txt
python main.py
```

---

## Results at a Glance

| Threshold | Accuracy | Default Recall | Notes |
|-----------|----------|----------------|-------|
| 0.50 (baseline) | 80% | ~2% | Looks fine, catches nothing |
| 0.60 (tuned) | 49% | **84%** | Protects principal, rejects some viable loans |

The right threshold depends on your portfolio. That's why it's configurable.
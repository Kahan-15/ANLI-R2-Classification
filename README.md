# ANLI Round 2 — Multi-Class NLI Classification

End-to-end machine learning pipeline for 3-way Natural Language Inference classification on the [Adversarial NLI (ANLI) Round 2](https://huggingface.co/datasets/facebook/anli) dataset.

## Task

Given a **premise** (context passage) and a **hypothesis** (claim), classify their relationship as:
- **Entailment** — the hypothesis is definitely true given the premise
- **Neutral** — the hypothesis might be true; not enough information
- **Contradiction** — the hypothesis is definitely false given the premise

ANLI Round 2 is an adversarially collected dataset specifically designed to challenge state-of-the-art NLI models, making it significantly harder than predecessors like SNLI and MNLI.

## Dataset

| Split | Examples |
|-------|----------|
| `train_r2` | 45,460 |
| `dev_r2` | 1,000 |
| `test_r2` | 1,000 |

**Key characteristics:**
- Labels are imbalanced in training (neutral 46.1%, entailment 31.8%, contradiction 22.1%) but balanced in dev/test (~33.3% each)
- Premises average 54 words (Wikipedia passages); hypotheses average 10 words
- Word overlap between premise and hypothesis is nearly identical across labels, confirming the adversarial design defeats surface-level heuristics
- 100% of examples fit within 256 tokens (max observed: 153 tokens)

## Results

| Model | Dev Accuracy | Test Accuracy | Dev F1 (Macro) |
|-------|-------------|---------------|----------------|
| Random Baseline | 33.3% | 33.3% | 0.333 |
| TF-IDF + Logistic Regression | 32.3% | 35.4% | 0.313 |
| **DeBERTa-v3-base (fine-tuned)** | **40.5%** | **39.3%** | **0.393** |

### Per-Class Performance (Dev Set)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Entailment | 0.360 | 0.569 | 0.441 |
| Neutral | 0.462 | 0.423 | 0.442 |
| Contradiction | 0.443 | 0.222 | 0.296 |

## Project Structure

```
anli-r2-classification/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── anli_r2_pipeline.ipynb          # Complete ML pipeline notebook
├── predict.py                      # FastAPI inference server
├── models/best_checkpoint/         # Saved model weights (gitignored)
├── results/                        # Evaluation plots
└── presentation/
```

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/anli-r2-classification.git
cd anli-r2-classification
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook anli_r2_pipeline.ipynb
```

### 3. Run Inference Server
```bash
uvicorn predict:app --host 0.0.0.0 --port 8000
```

Test: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"premise": "Idris Sultan (born January 1993) is a Tanzanian Actor.", "hypothesis": "Idris Sultan was born in Tanzania."}'`

## Docker Deployment

```bash
docker build -t anli-r2-classifier .
docker run -p 8000:8000 anli-r2-classifier
```

Or: `docker-compose up --build`

Health check: `curl http://localhost:8000/health`

## Approach

**Baseline:** TF-IDF + Logistic Regression (32.3% — near random chance, proving ANLI R2 defeats surface features)

**Primary Model:** DeBERTa-v3-base fine-tuned for 3 epochs with lr=2e-5, batch_size=16, warmup=300 steps

**Why DeBERTa-v3:** Disentangled attention + RTD pre-training makes it robust against adversarial NLI examples that were designed to fool BERT/RoBERTa

## Future Improvements

1. Data augmentation with SNLI + MNLI + FEVER-NLI (achieves ~70% on ANLI)
2. DeBERTa-v3-large (304M params) for ~5-8% improvement
3. Class weighting to address training set imbalance
4. Ensemble of multiple checkpoints
5. Hyperparameter optimization

## References

```bibtex
@InProceedings{nie2019adversarial,
    title={Adversarial NLI: A New Benchmark for Natural Language Understanding},
    author={Nie, Yixin and Williams, Adina and Dinan, Emily
            and Bansal, Mohit and Weston, Jason and Kiela, Douwe},
    booktitle={Proceedings of the 58th Annual Meeting of the ACL},
    year={2020},
    publisher={Association for Computational Linguistics},
}
```

License: CC-BY-NC 4.0

# Word Vectorizer Benchmark — Toxic Comment Classification

A rigorous, reproducible benchmark comparing **four deep learning word vectorization methods** on the Jigsaw Toxic Comment Classification dataset.

Built as part of the Machine Learning in Cybersecurity (24CYS214) course assignment, this project goes beyond the standard textbook treatment by implementing a clean, modular benchmark framework with consistent evaluation methodology.

---

## Why Toxic Comment Classification?

Toxic comment detection is a **hard NLP problem** for word vectorizers because:

1. **Adversarial text**: users deliberately misspell ("st00pid", "f\*ck") to evade keyword filters — a challenge that exposes the OOV weakness of static embeddings
2. **Negation and context**: "I would never hurt you" vs "I would hurt you" — identical bag-of-words representation, opposite semantics
3. **Class imbalance**: real toxic corpora are heavily skewed — F1-macro is a more honest metric than accuracy
4. **Domain mismatch**: GloVe trained on Wikipedia may not cover internet slang — an explainable, testable hypothesis

These properties make this dataset a natural stress-test that produces **meaningful, explainable performance differences** between vectorizers.

---

## Methods Compared

| Vectorizer | Type | Trained On | Dim | Key Property |
|---|---|---|---|---|
| **Word2Vec** (Skip-gram) | Static DL | This corpus | 100 | Local context windows |
| **FastText** (Skip-gram + n-grams) | Static DL + subword | This corpus | 100 | OOV handling via char n-grams |
| **GloVe** | Static pretrained | Wikipedia + Gigaword | 100 | Global co-occurrence statistics |
| **BERT** ([CLS] token, frozen) | Contextual DL | Wikipedia + BooksCorpus | 768 | Contextual, bidirectional |

**Classifier (same for all):** Logistic Regression with `class_weight='balanced'` — the classifier is held constant to isolate the vectorizer's contribution.

> Note: BoW and TF-IDF baselines are handled by a teammate as part of the same submission. Results are combined in the final report.

---

## Results

> ⚠️ Run the benchmark to populate this table with your actual results.

| Vectorizer | Dim | Fit Time | Transform Time | Accuracy | F1-Macro |
|---|---|---|---|---|---|
| Word2Vec | 100 | — | — | — | — |
| FastText | 100 | — | — | — | — |
| GloVe | 100 | — | — | — | — |
| BERT | 768 | — | — | — | — |

---

## Key Findings

*(To be filled after running experiments — see `results/benchmark_results.json`)*

Expected narrative based on prior literature:
- **FastText > Word2Vec** on this corpus because adversarial misspellings create many OOV tokens that Word2Vec maps to zero vectors
- **GloVe < Word2Vec** because Wikipedia vocabulary poorly covers internet slang
- **BERT > all static methods** due to contextual understanding of negation and sarcasm
- **Trade-off**: BERT achieves the best F1 but at ~10–50× the compute cost of Word2Vec — an important practical consideration

---

## Project Structure

```
word-vectorizer-benchmark/
├── data/
│   ├── prepare_dataset.py     # Download + subsample Jigsaw dataset
│   ├── raw/                   # Place train.csv here (from Kaggle)
│   ├── processed/             # Generated: jigsaw_binary.csv
│   └── glove/                 # Place glove.6B.100d.txt here
├── preprocessing/
│   └── text_cleaner.py        # Shared cleaning pipeline
├── vectorizers/
│   ├── base.py                # Abstract base class
│   ├── word2vec_vectorizer.py
│   ├── fasttext_vectorizer.py
│   ├── glove_vectorizer.py
│   └── bert_vectorizer.py
├── experiments/
│   └── run_benchmark.py       # Main experiment runner
├── analysis/
│   ├── plot_results.py        # Bar charts, heatmap, trade-off scatter
│   └── tsne_visualizer.py     # t-SNE embedding visualizations
├── results/
│   ├── benchmark_results.json # Auto-generated
│   └── plots/                 # Auto-generated
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Clone and install

```bash
git clone https://github.com/yourusername/word-vectorizer-benchmark
cd word-vectorizer-benchmark
pip install -r requirements.txt
```

For GPU support (RTX 3050 — CUDA):
```bash
# Check pytorch.org for the right CUDA version for your driver
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Get the data

**Jigsaw dataset:**
```bash
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip -d data/raw/
python data/prepare_dataset.py
```

**GloVe pretrained vectors:**
```bash
# Download from https://nlp.stanford.edu/projects/glove/
# Extract glove.6B.100d.txt and place at:
mkdir -p data/glove
# mv glove.6B.100d.txt data/glove/
```

### 3. Run the benchmark

```bash
# All four vectorizers
python experiments/run_benchmark.py

# Without BERT (CPU-only machines)
python experiments/run_benchmark.py --skip-bert

# Specific vectorizers
python experiments/run_benchmark.py --vectorizers word2vec fasttext
```

### 4. Generate plots

```bash
python analysis/plot_results.py
python analysis/tsne_visualizer.py
```

---

## Design Decisions

**Why mean pooling for Word2Vec / FastText / GloVe?**
Mean pooling is a simple, well-established baseline for sentence embeddings from word vectors (Mitchell & Lapata, 2010). More sophisticated pooling (max, attention-weighted) is intentionally avoided to keep the vectorizer comparison fair.

**Why frozen BERT?**
Fine-tuning BERT would improve accuracy but conflates the classifier and the vectorizer. A frozen [CLS] token measures how much pre-trained contextual representations help — a fair comparison with the other three methods.

**Why no stemming?**
Word2Vec, FastText, GloVe, and BERT all require real word forms. Stemming would corrupt vocabulary lookups and degrade embedding quality.

**Why Logistic Regression as the classifier?**
LR is convex, interpretable, and fast. Using the same classifier across all vectorizers ensures performance differences are attributable to the embedding quality, not the classifier's capacity.

---

## References

1. Mikolov, T., et al. (2013). Distributed representations of words and phrases. *NeurIPS 2013*. https://arxiv.org/abs/1310.4546
2. Bojanowski, P., et al. (2017). Enriching word vectors with subword information. *TACL, 5*, 135–146. https://arxiv.org/abs/1607.04606
3. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP 2014*. https://aclanthology.org/D14-1162
4. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL 2019*. https://arxiv.org/abs/1810.04805
5. van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR, 9*, 2579–2605.

---

## License

MIT — free to use, modify, and build on.

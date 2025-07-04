# Tokenizer Benchmarking: WordPiece vs BPE on IMDb Sentiment

## Description
This project investigates how two popular subword tokenization algorithms—**WordPiece** and **Byte-Pair Encoding (BPE)**—affect performance on a sentiment-classification task.  
We train both tokenizers from scratch on a subset of the IMDb movie-review corpus and use their tokens as features in a simple logistic-regression classifier.

The goal is to showcase a **lightweight, fully reproducible experiment** illustrating:
- How to train custom tokenizers with the [`tokenizers`](https://github.com/huggingface/tokenizers) library  
- How to plug those tokenizers into scikit-learn workflows  
- The trade-offs in accuracy, F1, and training time

## Dataset
- **IMDb Reviews** (binary sentiment)  
  Loaded automatically via `datasets.load_dataset("imdb")`.  
  We subsample 5 000 train and 5 000 test examples for quick iteration.

## Installation
```bash
git clone https://github.com/your-handle/tokenizer-benchmark-imdb.git
cd tokenizer-benchmark-imdb
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install echo datasets tokenizers scikit-learn pandas numpy tqdm
```

## Usage
```bash
python main.py                # default 5 000-sample benchmark
python main.py --sample-size 10000   # larger sample
```
## Example Output
```bash
2025-07-04 09:02:11 │ INFO │ Training wordpiece tokenizer (vocab=16000)…
...
WORDPIECE → accuracy=0.8784 | f1=0.8784
BPE       → accuracy=0.8766 | f1=0.8766

=== Benchmark Results ===
Tokenizer  Accuracy    F1  Time_sec
 WORDPIECE    0.8784 0.8784    64.23
      BPE     0.8766 0.8766    62.51

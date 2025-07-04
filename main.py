from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

logging.basicConfig(
    format="%(asctime)s │ %(levelname)s │ %(message)s", level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tokenizer utilities
# --------------------------------------------------------------------------- #
def train_tokenizer(
    texts: List[str], model_type: str, vocab_size: int = 16_000
) -> Tokenizer:
    """
    Train a tokenizer.

    Parameters
    ----------
    texts : list[str]
        Training corpus.
    model_type : {"wordpiece", "bpe"}
        Which tokenizer algorithm to use.
    vocab_size : int, default 16_000
        Maximum vocabulary size.

    Returns
    -------
    tokenizers.Tokenizer
    """
    if model_type == "wordpiece":
        model = WordPiece(unk_token="[UNK]")
        trainer = WordPieceTrainer(
            vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        )
    elif model_type == "bpe":
        model = BPE(unk_token="<unk>")
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>"])
    else:
        raise ValueError("model_type must be 'wordpiece' or 'bpe'")

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()

    LOGGER.info("Training %s tokenizer (vocab=%d)…", model_type, vocab_size)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def make_sklearn_tokenizer(tokenizer: Tokenizer) -> Callable[[str], List[str]]:
    """
    Wrap a `tokenizers.Tokenizer` into a function usable by scikit-learn.
    Returns token **strings** (not IDs) so that CountVectorizer can build a vocabulary.
    """

    def tok(text: str) -> List[str]:
        return tokenizer.encode(text).tokens

    return tok


# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #
def run_experiment(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    tokenizer_type: str,
) -> Tuple[float, float]:
    """Train / evaluate a sentiment model using the specified tokenizer."""
    tokenizer = train_tokenizer(train_texts, tokenizer_type)
    vectorizer = CountVectorizer(
        tokenizer=make_sklearn_tokenizer(tokenizer),
        lowercase=False,  # tokenizer already normalizes
        max_features=20_000,
    )

    LOGGER.info("Vectorizing data with %s tokenizer…", tokenizer_type)
    X_train = vectorizer.fit_transform(tqdm(train_texts, desc="Fit-transform"))
    X_test = vectorizer.transform(tqdm(test_texts, desc="Transform"))

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    LOGGER.info("Training classifier…")
    clf.fit(X_train, train_labels)

    preds = clf.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    LOGGER.info("%s → accuracy=%.4f | f1=%.4f", tokenizer_type.upper(), acc, f1)
    return acc, f1


# --------------------------------------------------------------------------- #
# Main entry-point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenizer benchmarking on IMDb")
    p.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of examples to sample from each split (smaller = faster).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Loading IMDb dataset…")
    dataset = load_dataset("imdb")
    train_ds = dataset["train"].shuffle(seed=args.random_state).select(
        range(args.sample_size)
    )
    test_ds = dataset["test"].shuffle(seed=args.random_state).select(
        range(args.sample_size)
    )

    train_texts, train_labels = train_ds["text"], train_ds["label"]
    test_texts, test_labels = test_ds["text"], test_ds["label"]

    results = []

    for tok_type in ("wordpiece", "bpe"):
        start = time.time()
        acc, f1 = run_experiment(
            train_texts, train_labels, test_texts, test_labels, tok_type
        )
        duration = time.time() - start
        results.append(
            {
                "Tokenizer": tok_type.upper(),
                "Accuracy": acc,
                "F1": f1,
                "Time_sec": duration,
            }
        )

    # Use concat (no append) to build final results table
    df_results = pd.concat([pd.DataFrame([r]) for r in results], ignore_index=True)
    print("\n=== Benchmark Results ===")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()

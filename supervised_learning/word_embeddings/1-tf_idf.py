#!/usr/bin/env python3
"""TF-IDF embedding"""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding

    sentences: list of sentences to analyze
    vocab: list of vocabulary words to use; if None, all words are used
    Returns: embeddings (np.ndarray of shape (s, f)), features
    """
    tokens = [re.findall(r"[a-z]+", re.sub(r"'s\b", "", s.lower()))
              for s in sentences]
    if vocab is None:
        vocab = sorted({w for sent in tokens for w in sent})
    features = np.array(vocab)
    index = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros((len(sentences), len(vocab)))
    for i, sent in enumerate(tokens):
        for w in sent:
            if w in index:
                tf[i, index[w]] += 1
    n = len(sentences)
    df = np.count_nonzero(tf, axis=0)
    idf = np.log((1 + n) / (1 + df)) + 1
    embeddings = tf * idf
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms, features

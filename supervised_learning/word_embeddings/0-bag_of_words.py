#!/usr/bin/env python3
"""Bag of words embedding"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix

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
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, sent in enumerate(tokens):
        for w in sent:
            if w in index:
                embeddings[i, index[w]] += 1
    return embeddings, features

#!/usr/bin/env python3
"""Write a function that performs K-means on a dataset:
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
Returns: C, clss, or None, None on failure

    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or \
            k <= 0 or not isinstance(iterations, int) or iterations <= 0:
        return (None, None)
    else:
        try:
            d = X.shape[1]
            mins = X.min(axis=0)
            maxs = X.max(axis=0)

            C = np.random.uniform(mins, maxs, size=(k, d))
            clss = None

            for _ in range(iterations):
                C_old = C.copy()

                """ Step 1: assign each point to nearest centroid

                use broadcasting trick to make the substraction
                w/o loops """
                diffs = X[:, np.newaxis, :] - C

                # euclidian distance to all 5 centroids
                dists = np.sqrt((diffs ** 2).sum(axis=2))
                # find which cluster point belongs to
                clss = dists.argmin(axis=1)

                """ Step 2: recompute centroids as mean of points"""
                for j in range(k):
                    # True where condition holds
                    mask = (clss == j)
                    # returns True if any element of iterable is True
                    if mask.any():
                        # average along the rows
                        C[j] = X[mask].mean(axis=0)
                    else:
                        C[j] = np.random.uniform(mins, maxs)

                # re-compute clss with new C (if C changed)
                diffs = X[:, np.newaxis, :] - C
                dists = np.sqrt((diffs ** 2).sum(axis=2))
                clss = dists.argmin(axis=1)

                """ Step 3: early stop if centroids didn't move"""
                # test if all elements are identical
                if np.all(C == C_old):
                    break

            return (C, clss)

        except Exception:
            return (None, None)

r"""
This module is a collection of metrics commonly used during pretraining and
downstream evaluation. Two main classes here are:

- :class:`TopkAccuracy` used for ImageNet linear classification evaluation.
- :class:`CocoCaptionsEvaluator` used for caption evaluation (CIDEr and SPICE).

Parts of this module (:meth:`tokenize`, :meth:`cider` and :meth:`spice`) are
adapted from `coco-captions evaluation code <https://github.com/tylin/coco-caption>`_.
"""
from collections import defaultdict
from typing import Dict, List

import numpy as np


# -------------------------------------------------------------------------
def to_ngrams(sentence: str, n: int = 4):
    r"""Convert a sentence into n-grams and their counts."""
    words = sentence.split()
    counts = defaultdict(int)  # type: ignore
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] += 1
    return counts


def counts2vec(cnts, document_frequency, log_reference_length, n: int = 4):
    r"""Function maps counts of ngram to vector of tfidf weights."""
    vec = [defaultdict(float) for _ in range(n)]
    length = 0
    norm = [0.0 for _ in range(n)]
    for ngram, term_freq in cnts.items():
        df = np.log(max(1.0, document_frequency[ngram]))
        # tf (term_freq) * idf (precomputed idf) for n-grams
        vec[len(ngram) - 1][ngram] = float(term_freq) * (log_reference_length - df)
        # Compute norm for the vector: will be used for computing similarity
        norm[len(ngram) - 1] += pow(vec[len(ngram) - 1][ngram], 2)

        if len(ngram) == 2:
            length += term_freq
    norm = [np.sqrt(nn) for nn in norm]
    return vec, norm, length


def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref, n: int = 4, sigma: float = 6.0):
    r"""Compute the cosine similarity of two vectors."""
    delta = float(length_hyp - length_ref)
    val = np.array([0.0 for _ in range(n)])
    for nn in range(n):
        for ngram, _count in vec_hyp[nn].items():
            val[nn] += min(vec_hyp[nn][ngram], vec_ref[nn][ngram]) * vec_ref[nn][ngram]

        val[nn] /= (norm_hyp[nn] * norm_ref[nn]) or 1
        val[nn] *= np.e ** (-(delta**2) / (2 * sigma**2))
    return val


# -------------------------------------------------------------------------


def cider(
    predictions: Dict[int, List[str]],
    ground_truth: Dict[int, List[str]],
    n: int = 4,
    sigma: float = 6.0,
) -> float:
    r"""Compute CIDEr score given ground truth captions and predictions."""

    ctest = [to_ngrams(predictions[image_id][0]) for image_id in ground_truth]
    crefs = [[to_ngrams(gt) for gt in ground_truth[image_id]] for image_id in ground_truth]
    # Build document frequency and compute IDF.
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1

    # Compute log reference length.
    log_reference_length = np.log(float(len(crefs)))

    scores = []
    for test, refs in zip(ctest, crefs):
        # Compute vector for test captions.
        vec, norm, length = counts2vec(test, document_frequency, log_reference_length, n)
        # Compute vector for ref captions.
        score = np.array([0.0 for _ in range(n)])
        for ref in refs:
            vec_ref, norm_ref, length_ref = counts2vec(ref, document_frequency, log_reference_length, n)
            score += sim(vec, vec_ref, norm, norm_ref, length, length_ref, n, sigma)

        score_avg = np.mean(score)
        score_avg /= len(refs)
        score_avg *= 10.0
        scores.append(score_avg)

    return np.mean(scores)

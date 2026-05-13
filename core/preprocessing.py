"""
Preprocessing module for cosmetics review NLP web app.

This module contains:
1. Custom sklearn transformers for embedding-based feature extraction
2. Text preprocessing functions matching the task1.ipynb pipeline
3. Collocation dictionary builders for bigram discovery
"""

import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# Regex pattern for tokenization (matches task1.ipynb)
REGEX_PATTERN = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"


# ==============================================================================
# CUSTOM SKLEARN TRANSFORMERS
# ==============================================================================

class UnweightedVectorTransformer(BaseEstimator, TransformerMixin):
    """Transform text to unweighted mean of word embeddings.

    For each document, computes the mean of all word embeddings (no TF-IDF weighting).
    Used in Model C (GloVe embeddings + metadata).
    """

    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict
        self.dim = len(next(iter(embeddings_dict.values()))) if embeddings_dict else 300

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for text in X:
            tokens = str(text).split()
            vecs = [self.embeddings_dict[w] for w in tokens if w in self.embeddings_dict]
            result.append(np.mean(vecs, axis=0) if vecs else np.zeros(self.dim))
        return np.array(result)


class WeightedVectorTransformer(BaseEstimator, TransformerMixin):
    """Transform text to TF-IDF weighted mean of word embeddings.

    For each document, computes weighted average of word embeddings using TF-IDF scores.
    Used in advanced models that require importance weighting.
    """

    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict
        self.tfidf = None
        self.dim = len(next(iter(embeddings_dict.values()))) if embeddings_dict else 300

    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(X)
        return self

    def transform(self, X):
        tfidf_matrix = self.tfidf.transform(X)
        vocab = self.tfidf.vocabulary_
        result = []
        for row_num, text in enumerate(X):
            tokens = str(text).split()
            vecs, wts = [], []
            for w in tokens:
                if w in self.embeddings_dict and w in vocab:
                    vecs.append(self.embeddings_dict[w])
                    wts.append(tfidf_matrix[row_num, vocab[w]])
            if vecs and sum(wts) > 0:
                result.append(np.average(vecs, axis=0, weights=wts))
            else:
                result.append(np.zeros(self.dim))
        return np.array(result)


# ==============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ==============================================================================

def load_stopwords(filepath):
    """Load stopwords from file (one per line).

    Args:
        filepath: Path to stopwords file

    Returns:
        Set of lowercase stopwords
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f if line.strip())


def preprocess_text(text, colloc_dict, stop_words, lemmatizer=None):
    """Preprocess raw review text using the exact pipeline from task1.ipynb.

    Pipeline steps:
    1. Lowercase + apply collocation replacements (bigram → hyphenated)
    2. Regex tokenization
    3. Short word removal (length < 2)
    4. Stopword removal
    5. Lemmatization

    Args:
        text: Raw review text
        colloc_dict: Dictionary mapping bigrams (with space) to hyphenated replacements
        stop_words: Set of stopwords to remove
        lemmatizer: WordNetLemmatizer instance (created if None)

    Returns:
        Space-separated string of processed tokens
    """
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    # 1. Lowercase + apply collocation replacements (bigram → hyphenated)
    text = str(text).lower()
    for bigram, replacement in colloc_dict.items():
        if bigram in text:
            text = text.replace(bigram, replacement)

    # 2. Regex tokenization
    tokens = [match.group(0) for match in re.finditer(REGEX_PATTERN, text)]

    # 3. Short word removal (len < 2)
    tokens = [t for t in tokens if len(t) >= 2]

    # 4. Stopword removal
    tokens = [t for t in tokens if t not in stop_words]

    # 5. Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


# ==============================================================================
# COLLOCATION DICTIONARY BUILDERS
# ==============================================================================

# Manual patterns from task1.ipynb (bigrams that should always be hyphenated)
MANUAL_PATTERNS = [
    'kay beauty', 'smudge proof', 'matte finish', 'highly pigmented', 'dark circles',
    'light weight', 'lip balm', 'medium tone', 'highly recommend', 'creamy texture',
    'dry lips', 'medium coverage', 'fair tone', 'fair medium', 'lip liner',
    'nail polish', 'travel friendly', 'beauty products', 'contour sticks', 'loose powder',
    'eye shadow', 'daily wear', 'matte lipstick', 'affordable price', 'full coverage',
    'price range', 'white cast', 'frizz free', 'natural finish', 'jet black',
    'water proof', 'argan oil', 'oily scalp', 'eye liner', 'light medium',
    'setting spray', 'daily basis', 'rose gold', 'liquid lipstick', 'coconut milk',
    'paraben free', 'makeup pouch', 'lip gloss', 'reasonable price', 'dewy finish',
    'smudge free', 'high end', 'makeup products', 'medium fair', 'brown nude',
    'dark brown', 'compact powder', 'winged liner'
]


def build_collocation_dict():
    """Build collocation dictionary from manual patterns.

    Returns dictionary mapping bigrams (with space) to hyphenated replacements.
    This is a lightweight version for inference - just converts the manual patterns.
    """
    colloc_dict = {}
    for pattern in MANUAL_PATTERNS:
        hyphenated = pattern.replace(' ', '-')
        colloc_dict[pattern] = hyphenated
    return colloc_dict


def build_collocation_dict_from_corpus(reviews, vocab_set, min_freq=5, pmi_threshold=3.0):
    """Build complete collocation dictionary using PMI-based discovery.

    This replicates the full notebook pipeline:
    1. Tokenize all reviews with the regex pattern
    2. Use BigramCollocationFinder to discover frequent bigrams
    3. Filter by PMI score and minimum frequency
    4. Merge with manual patterns
    5. Keep only bigrams that appear in final vocab (as hyphenated form)

    Args:
        reviews: Iterable of raw review texts
        vocab_set: Set of words in final vocabulary (to filter results)
        min_freq: Minimum frequency for a bigram to be considered
        pmi_threshold: Minimum PMI score for a bigram to be considered

    Returns:
        Dictionary mapping bigrams (with space) to hyphenated replacements
    """
    # Tokenize all reviews
    all_tokens = []
    for review in reviews:
        text = str(review).lower()
        tokens = [match.group(0) for match in re.finditer(REGEX_PATTERN, text)]
        all_tokens.extend(tokens)

    # Use BigramCollocationFinder for PMI-based discovery
    finder = BigramCollocationFinder.from_words(all_tokens)

    # Apply frequency filter
    finder.apply_freq_filter(min_freq)

    # Get bigrams with PMI scores
    pmi_measure = BigramAssocMeasures.pmi
    scored_bigrams = finder.score_ngrams(pmi_measure)

    # Build dictionary from high-PMI bigrams
    colloc_dict = {}
    for (word1, word2), score in scored_bigrams:
        if score >= pmi_threshold:
            bigram_space = f"{word1} {word2}"
            bigram_hyphen = f"{word1}-{word2}"
            # Only include if hyphenated form is in final vocab
            if bigram_hyphen in vocab_set:
                colloc_dict[bigram_space] = bigram_hyphen

    # Add manual patterns (these override auto-discovered ones)
    for pattern in MANUAL_PATTERNS:
        hyphenated = pattern.replace(' ', '-')
        # Only include if in vocab
        if hyphenated in vocab_set:
            colloc_dict[pattern] = hyphenated

    return colloc_dict

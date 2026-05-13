"""
Data loader module for APDS-A3 Gradio web application.
Loads all data assets at startup: datasets, models, embeddings, and pre-computed aggregations.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any

# Import custom transformers for pickle deserialization
from core.preprocessing import UnweightedVectorTransformer, WeightedVectorTransformer

# Alias module so pickled objects referencing 'preprocessing' can be deserialized
import sys
import core.preprocessing
sys.modules['preprocessing'] = core.preprocessing

# In-memory review store for new reviews
new_reviews = []


def add_review(review_dict: Dict[str, Any]) -> None:
    """Add a new review to the in-memory store."""
    new_reviews.append(review_dict)


def get_all_reviews_for_product(
    product_id: str,
    df: pd.DataFrame,
    new_reviews_list: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get all reviews for a product, combining original dataset and new reviews.

    Args:
        product_id: Product identifier
        df: Original reviews DataFrame
        new_reviews_list: List of new reviews from in-memory store

    Returns:
        Combined list of review dictionaries
    """
    original = df[df['product_id'] == product_id].to_dict('records')
    new = [r for r in new_reviews_list if r.get('product_id') == product_id]
    return original + new


def load_processed_csv(base_path: str) -> pd.DataFrame:
    """Load the preprocessed reviews dataset."""
    csv_path = os.path.join(base_path, 'notebooks', 'processed.csv')
    print(f"Loading processed.csv from {csv_path}...")
    start = time.time()
    df = pd.read_csv(csv_path)
    elapsed = time.time() - start
    print(f"  ✓ Loaded {len(df):,} reviews in {elapsed:.2f}s")
    return df


def load_vocab(base_path: str) -> Dict[str, int]:
    """Load vocabulary mapping from word to index."""
    vocab_path = os.path.join(base_path, 'notebooks', 'vocab.txt')
    print(f"Loading vocab.txt from {vocab_path}...")
    start = time.time()
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                word, idx = line.rsplit(':', 1)
                vocab[word] = int(idx)
    elapsed = time.time() - start
    print(f"  ✓ Loaded {len(vocab):,} vocabulary words in {elapsed:.2f}s")
    return vocab


def load_stopwords(base_path: str) -> Set[str]:
    """Load stopwords set."""
    stopwords_path = os.path.join(base_path, 'notebooks', 'stopwords_en.txt')
    print(f"Loading stopwords_en.txt from {stopwords_path}...")
    start = time.time()
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = {line.strip().lower() for line in f if line.strip()}
    elapsed = time.time() - start
    print(f"  ✓ Loaded {len(stopwords):,} stopwords in {elapsed:.2f}s")
    return stopwords


def load_glove_embeddings(base_path: str) -> Dict[str, np.ndarray]:
    """
    Load FULL GloVe embeddings (all ~400K words).
    Prints progress every 100K lines.
    """
    glove_path = os.path.join(base_path, 'notebooks', 'glove.6B.300d.txt')
    print(f"Loading glove.6B.300d.txt from {glove_path}...")
    print("  (This will take a while - ~400K words)")

    start = time.time()
    embeddings = {}
    line_count = 0

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1

            # Print progress every 100K lines
            if line_count % 100000 == 0:
                elapsed = time.time() - start
                print(f"  ... {line_count:,} words loaded ({elapsed:.1f}s)")

            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector

    elapsed = time.time() - start
    print(f"  ✓ Loaded {len(embeddings):,} GloVe embeddings in {elapsed:.2f}s")
    return embeddings


def load_pickled_models(base_path: str) -> Dict[str, Any]:
    """Load all pickled models and artifacts."""
    models_dir = os.path.join(base_path, 'models')
    print(f"Loading pickled models from {models_dir}/...")

    model_files = [
        'model_a_bow.pkl',
        'model_b_bow_meta.pkl',
        'model_c_glove_meta.pkl',
        'label_encoder.pkl',
        'collocation_dict.pkl',
        'product_vectors.pkl'
    ]

    models = {}
    start = time.time()

    for filename in model_files:
        filepath = os.path.join(models_dir, filename)
        key = filename.replace('.pkl', '')

        with open(filepath, 'rb') as f:
            models[key] = pickle.load(f)

        print(f"  ✓ Loaded {filename}")

    elapsed = time.time() - start
    print(f"  Total model loading time: {elapsed:.2f}s")
    return models


def compute_product_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute product-level aggregations.
    Groups by product_id and aggregates key fields.
    """
    print("Computing product aggregations...")
    start = time.time()

    products_df = df.groupby('product_id').agg({
        'brand_name': 'first',
        'product_title': 'first',
        'price': 'first',
        'avg_product_rating': 'first',
        'product_rating_count': 'first',
        'product_url': 'first',
        'review_id': 'count'  # Count reviews per product
    }).reset_index()

    # Rename review_id count to review_count
    products_df.rename(columns={'review_id': 'review_count'}, inplace=True)

    elapsed = time.time() - start
    print(f"  ✓ Aggregated {len(products_df):,} unique products in {elapsed:.2f}s")
    return products_df


def build_reviews_by_product(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a dictionary mapping product_id to list of review dictionaries.
    """
    print("Building reviews_by_product index...")
    start = time.time()

    reviews_by_product = {}
    for _, row in df.iterrows():
        product_id = row['product_id']
        if product_id not in reviews_by_product:
            reviews_by_product[product_id] = []
        reviews_by_product[product_id].append(row.to_dict())

    elapsed = time.time() - start
    print(f"  ✓ Indexed reviews for {len(reviews_by_product):,} products in {elapsed:.2f}s")
    return reviews_by_product


def load_all() -> Dict[str, Any]:
    """
    Load all data assets and return a dictionary containing everything.

    Returns:
        Dictionary with keys:
            - df: reviews DataFrame
            - vocab: vocabulary dict
            - stopwords: stopwords set
            - glove_embeddings: GloVe embeddings dict
            - models: dict of pickled models
            - products_df: product aggregations DataFrame
            - reviews_by_product: dict mapping product_id to review list
    """
    print("=" * 60)
    print("LOADING ALL DATA ASSETS")
    print("=" * 60)

    total_start = time.time()
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load all assets
    df = load_processed_csv(base_path)
    vocab = load_vocab(base_path)
    stopwords = load_stopwords(base_path)
    glove_embeddings = load_glove_embeddings(base_path)
    models = load_pickled_models(base_path)
    products_df = compute_product_aggregations(df)
    reviews_by_product = build_reviews_by_product(df)

    total_elapsed = time.time() - total_start

    print("=" * 60)
    print(f"✓ ALL DATA LOADED SUCCESSFULLY in {total_elapsed:.2f}s")
    print("=" * 60)

    return {
        'df': df,
        'vocab': vocab,
        'stopwords': stopwords,
        'glove_embeddings': glove_embeddings,
        'models': models,
        'products_df': products_df,
        'reviews_by_product': reviews_by_product
    }


if __name__ == '__main__':
    # Test loading
    data = load_all()
    print("\nData assets loaded:")
    print(f"  - Reviews: {len(data['df']):,}")
    print(f"  - Vocabulary: {len(data['vocab']):,} words")
    print(f"  - Stopwords: {len(data['stopwords']):,} words")
    print(f"  - GloVe embeddings: {len(data['glove_embeddings']):,} words")
    print(f"  - Models: {list(data['models'].keys())}")
    print(f"  - Products: {len(data['products_df']):,}")
    print(f"  - Reviews by product index: {len(data['reviews_by_product']):,} products")

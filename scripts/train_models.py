#!/usr/bin/env python3
"""
Train ML models for cosmetics review buyer/non-buyer prediction.

This script trains 3 high-performing models and saves 6 pickle files to models/:
1. rf_bow_extra.pkl - BoW + Extra Metadata + Random Forest
2. rf_unweighted_extra.pkl - Unweighted GloVe + Extra Metadata + Random Forest
3. rf_weighted_extra.pkl - Weighted GloVe + Extra Metadata + Random Forest
4. label_encoder.pkl - Brand name encoder
5. collocation_dict.pkl - Bigram collocation dictionary
6. product_vectors.pkl - Product-level GloVe embeddings

Input files (from notebooks/ subdirectory):
- processed.csv (61,275 reviews)
- vocab.txt (7,366 words)
- glove.6B.300d.txt (GloVe 300d embeddings)
- stopwords_en.txt (570 stopwords)

Usage:
    python train_models.py
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.preprocessing import (
    UnweightedVectorTransformer,
    WeightedVectorTransformer,
    preprocess_text,
    load_stopwords,
    build_collocation_dict_from_corpus,
    REGEX_PATTERN
)
import re
from nltk.stem import WordNetLemmatizer


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def load_vocab(filepath):
    """Load vocabulary from word:index file.

    Args:
        filepath: Path to vocab.txt

    Returns:
        Dictionary mapping words to indices
    """
    vocab_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                word, idx = line.rsplit(':', 1)
                vocab_dict[word] = int(idx)
    return vocab_dict


def load_glove(filepath, vocab_set):
    """Load GloVe embeddings filtered to vocabulary.

    Args:
        filepath: Path to glove.6B.300d.txt
        vocab_set: Set of words to keep (for filtering)

    Returns:
        Dictionary mapping words to 300-dim numpy arrays
    """
    embeddings_dict = {}
    total_lines = 0
    loaded_lines = 0

    print(f"Loading GloVe embeddings from {filepath}...")
    print(f"Filtering to vocabulary of {len(vocab_set)} words...")

    start_time = time.time()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if total_lines % 100000 == 0:
                print(f"  Processed {total_lines:,} lines, kept {loaded_lines:,} vectors...")

            parts = line.strip().split()
            word = parts[0]

            # Only load embeddings for words in our vocabulary
            if word in vocab_set:
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings_dict[word] = vector
                loaded_lines += 1

    elapsed = time.time() - start_time
    print(f"Loaded {loaded_lines:,} / {len(vocab_set)} vocab words in {elapsed:.1f}s")
    print(f"Total GloVe file had {total_lines:,} vectors")

    return embeddings_dict


def main():
    """Main training pipeline."""

    print_section("Cosmetics Review ML Model Training")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    print_section("STEP 1: Loading Data")

    # Load processed reviews
    print("Loading processed.csv...")
    df = pd.read_csv('notebooks/processed.csv')
    print(f"  Loaded {len(df):,} reviews")

    # Filter out rows with missing processed_review_text
    df = df[df['processed_review_text'].notna()]
    print(f"  After filtering missing text: {len(df):,} reviews")

    # Load vocabulary
    print("\nLoading vocab.txt...")
    vocab_dict = load_vocab('notebooks/vocab.txt')
    vocab_set = set(vocab_dict.keys())
    print(f"  Loaded {len(vocab_dict):,} words")

    # Load GloVe embeddings (filtered to vocab)
    print("\nLoading GloVe embeddings...")
    start_glove = time.time()
    embeddings_dict = load_glove('notebooks/glove.6B.300d.txt', vocab_set)
    glove_time = time.time() - start_glove
    print(f"  GloVe loading took {glove_time:.1f}s")

    # Load stopwords
    print("\nLoading stopwords...")
    stopwords = load_stopwords('notebooks/stopwords_en.txt')
    print(f"  Loaded {len(stopwords):,} stopwords")

    # -------------------------------------------------------------------------
    # STEP 2: Prepare Features
    # -------------------------------------------------------------------------
    print_section("STEP 2: Preparing Features")

    # Build collocation dictionary from corpus
    print("Building collocation dictionary from corpus...")
    start_colloc = time.time()
    colloc_dict = build_collocation_dict_from_corpus(
        df['review_text'].fillna(''),
        vocab_set,
        min_freq=5,
        pmi_threshold=3.0
    )
    colloc_time = time.time() - start_colloc
    print(f"  Built {len(colloc_dict)} collocations in {colloc_time:.1f}s")

    # Preprocess review titles
    print("\nPreprocessing review titles...")
    lemmatizer = WordNetLemmatizer()
    df['processed_title'] = df['review_title'].fillna('').apply(
        lambda x: preprocess_text(x, colloc_dict, stopwords, lemmatizer)
    )
    print(f"  Processed {len(df):,} titles")

    # Encode brand names
    print("\nEncoding brand names...")
    le = LabelEncoder()
    df['brand_encoded'] = le.fit_transform(df['brand_name'].fillna('unknown'))
    print(f"  Encoded {len(le.classes_)} unique brands")

    # Prepare target variable (handle both string and boolean values)
    print("\nPreparing target variable...")
    df['is_a_buyer'] = df['is_a_buyer'].map({
        True: 1, False: 0,
        'True': 1, 'False': 0,
        1: 1, 0: 0
    })
    y = df['is_a_buyer'].astype(int)
    print(f"  Target distribution:")
    print(f"    Buyers: {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
    print(f"    Non-buyers: {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")

    # Stratified train/test split
    print("\nPerforming stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # -------------------------------------------------------------------------
    # STEP 3-5: Train Top 3 Models (Random Forest Ensemble)
    # -------------------------------------------------------------------------
    print_section("STEP 3-5: Training Top 3 Random Forest Models")

    # Configuration for the new models
    RF_CONFIG = {'n_estimators': 200, 'random_state': 42, 'n_jobs': -1}
    extra_cols = ['price', 'avg_product_rating', 'product_rating_count', 'review_rating', 'brand_encoded']
    
    meta_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    print("Building pipelines for Top 3 Models...")

    # Model A: BoW (Text+Title+Extra) + RF
    model_a = ImbPipeline([
        ('features', ColumnTransformer([
            ('text',  CountVectorizer(vocabulary=vocab_dict), 'processed_review_text'),
            ('title', CountVectorizer(), 'processed_title'),
            ('meta',  meta_pipe, extra_cols)
        ])),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(**RF_CONFIG))
    ])

    # Model B: Unweighted GloVe (Text+Title+Extra) + RF
    model_b = ImbPipeline([
        ('features', ColumnTransformer([
            ('text',  UnweightedVectorTransformer(embeddings_dict), 'processed_review_text'),
            ('title', UnweightedVectorTransformer(embeddings_dict), 'processed_title'),
            ('meta',  meta_pipe, extra_cols)
        ])),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(**RF_CONFIG))
    ])

    # Model C: Weighted GloVe (Text+Title+Extra) + RF
    model_c = ImbPipeline([
        ('features', ColumnTransformer([
            ('text',  WeightedVectorTransformer(embeddings_dict), 'processed_review_text'),
            ('title', WeightedVectorTransformer(embeddings_dict), 'processed_title'),
            ('meta',  meta_pipe, extra_cols)
        ])),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(**RF_CONFIG))
    ])

    # Training models
    print("\nTraining Model A (BoW + RF)...")
    start_a = time.time()
    model_a.fit(X_train, y_train)
    train_a_time = time.time() - start_a
    print(f"  Training took {train_a_time:.1f}s")

    print("\nTraining Model B (Unweighted GloVe + RF)...")
    start_b = time.time()
    model_b.fit(X_train, y_train)
    train_b_time = time.time() - start_b
    print(f"  Training took {train_b_time:.1f}s")

    print("\nTraining Model C (Weighted GloVe + RF)...")
    start_c = time.time()
    model_c.fit(X_train, y_train)
    train_c_time = time.time() - start_c
    print(f"  Training took {train_c_time:.1f}s")

    # Evaluation
    print("\n" + "="*40)
    print("EVALUATION ON TEST SET")
    print("="*40)

    print("\nEvaluating Model A (BoW + RF):")
    y_pred_a = model_a.predict(X_test)
    print(classification_report(y_test, y_pred_a, target_names=['Non-buyer', 'Buyer']))

    print("\nEvaluating Model B (Unweighted GloVe + RF):")
    y_pred_b = model_b.predict(X_test)
    print(classification_report(y_test, y_pred_b, target_names=['Non-buyer', 'Buyer']))

    print("\nEvaluating Model C (Weighted GloVe + RF):")
    y_pred_c = model_c.predict(X_test)
    print(classification_report(y_test, y_pred_c, target_names=['Non-buyer', 'Buyer']))

    # -------------------------------------------------------------------------
    # STEP 6: Compute Product Vectors
    # -------------------------------------------------------------------------
    print_section("STEP 6: Computing Product-Level Embeddings")

    print("Computing product vectors (mean of review embeddings)...")
    start_pv = time.time()
    product_vectors = {}

    for pid, group in df.groupby('product_id'):
        texts = group['processed_review_text'].tolist()
        vecs = []
        for text in texts:
            tokens = str(text).split()
            word_vecs = [embeddings_dict[w] for w in tokens if w in embeddings_dict]
            if word_vecs:
                vecs.append(np.mean(word_vecs, axis=0))

        if vecs:
            product_vectors[pid] = np.mean(vecs, axis=0)
        else:
            product_vectors[pid] = np.zeros(300)

    pv_time = time.time() - start_pv
    print(f"  Computed vectors for {len(product_vectors):,} products in {pv_time:.1f}s")

    # -------------------------------------------------------------------------
    # STEP 7: Save All Pickles
    # -------------------------------------------------------------------------
    print_section("STEP 7: Saving Models and Artifacts")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Save all artifacts
    artifacts = [
        ('models/rf_bow_extra.pkl', model_a, 'Model A (BoW + Extra + RF)'),
        ('models/rf_unweighted_extra.pkl', model_b, 'Model B (Unweighted GloVe + Extra + RF)'),
        ('models/rf_weighted_extra.pkl', model_c, 'Model C (Weighted GloVe + Extra + RF)'),
        ('models/label_encoder.pkl', le, 'Label Encoder'),
        ('models/collocation_dict.pkl', colloc_dict, 'Collocation Dictionary'),
        ('models/product_vectors.pkl', product_vectors, 'Product Vectors')
    ]

    for filepath, obj, name in artifacts:
        print(f"Saving {name} to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

        # Print file size
        size_bytes = os.path.getsize(filepath)
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        print(f"  Saved {size_str}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Training Complete")

    print("Timing Summary:")
    print(f"  GloVe loading:    {glove_time:>8.1f}s")
    print(f"  Collocation dict: {colloc_time:>8.1f}s")
    print(f"  Model A training: {train_a_time:>8.1f}s")
    print(f"  Model B training: {train_b_time:>8.1f}s")
    print(f"  Model C training: {train_c_time:>8.1f}s")
    print(f"  Product vectors:  {pv_time:>8.1f}s")

    print("\nAll models saved to models/ directory:")
    for filepath, _, name in artifacts:
        print(f"  ✓ {filepath}")

    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

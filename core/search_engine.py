"""
Task 1: Item Search Engine
Multi-strategy product search with string matching and semantic similarity.
"""

import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Optional


def get_query_embedding(query: str, embeddings_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Compute query embedding as mean of GloVe vectors for query words.

    Args:
        query: raw user search string
        embeddings_dict: dict[word → np.array(300)] (GloVe embeddings)

    Returns:
        np.array(300) or None if no words found in vocabulary
    """
    tokens = query.lower().split()
    vecs = [embeddings_dict[w] for w in tokens if w in embeddings_dict]
    if vecs:
        return np.mean(vecs, axis=0)
    return None


def normalized_string_match(query: str, products_df: pd.DataFrame, similarity_threshold: float = 0.7) -> List[int]:
    """
    Strategy 1: Normalized string matching with fuzzy brand name matching.

    Args:
        query: search query (lowercased internally)
        products_df: DataFrame with brand_name, product_title, product_id
        similarity_threshold: minimum SequenceMatcher ratio for brand variants

    Returns:
        list of product_ids matching the query
    """
    query_lower = query.lower()
    query_words = query_lower.split()
    matched_ids = []

    # Per-word AND matching: product must contain ALL query words in brand OR title
    for _, row in products_df.iterrows():
        brand_lower = str(row['brand_name']).lower()
        title_lower = str(row['product_title']).lower()
        combined = brand_lower + " " + title_lower

        if all(word in combined for word in query_words):
            matched_ids.append(row['product_id'])

    # Fuzzy brand name matching for variants (e.g., "Maybeline" ≈ "Maybelline New York")
    if len(query_lower) > 3:  # Only fuzzy match for queries > 3 chars
        for idx, row in products_df.iterrows():
            brand_lower = str(row['brand_name']).lower()

            # Extract first word of brand name for comparison
            brand_first_word = brand_lower.split()[0] if brand_lower else ""
            query_first_word = query_lower.split()[0]

            # Compare query with full brand and first word of brand
            similarity_full = SequenceMatcher(None, query_lower, brand_lower).ratio()
            similarity_first = SequenceMatcher(None, query_first_word, brand_first_word).ratio()

            if similarity_full >= similarity_threshold or similarity_first >= similarity_threshold:
                if row['product_id'] not in matched_ids:
                    matched_ids.append(row['product_id'])

    return list(set(matched_ids))


def semantic_search(query: str, products_df: pd.DataFrame, product_vectors: Dict[int, np.ndarray],
                    embeddings_dict: Dict[str, np.ndarray], similarity_threshold: float = 0.45) -> Dict[int, float]:
    """
    Strategy 2: Semantic search using GloVe embeddings and cosine similarity.

    Args:
        query: search query
        products_df: DataFrame with product_id
        product_vectors: dict[product_id → np.array(300)]
        embeddings_dict: dict[word → np.array(300)]
        similarity_threshold: minimum cosine similarity score

    Returns:
        dict[product_id → cosine_score] for products above threshold
    """
    query_embedding = get_query_embedding(query, embeddings_dict)

    if query_embedding is None:
        return {}

    results = {}
    query_embedding = query_embedding.reshape(1, -1)

    for product_id, product_vec in product_vectors.items():
        if product_id not in products_df['product_id'].values:
            continue

        product_vec = product_vec.reshape(1, -1)
        similarity = cosine_similarity(query_embedding, product_vec)[0][0]

        if similarity >= similarity_threshold:
            results[product_id] = float(similarity)

    return results


def search_products(query: str, products_df: pd.DataFrame, product_vectors: Dict[int, np.ndarray],
                   embeddings_dict: Dict[str, np.ndarray], top_n: int = 20) -> List[Dict]:
    """
    Multi-strategy product search with fusion scoring.

    Combines:
    - Strategy 1: Normalized string matching (exact + fuzzy)
    - Strategy 2: Semantic similarity using GloVe embeddings

    String matches get score=1.0, semantic matches get their cosine score.
    Results are deduplicated, sorted by score descending, and limited to top_n.

    Args:
        query: raw user search string
        products_df: DataFrame with one row per product (product_id, brand_name, product_title,
                     price, avg_product_rating, product_rating_count, product_url, review_count)
        product_vectors: dict[product_id → np.array(300)]
        embeddings_dict: dict[word → np.array(300)] (full GloVe, 400K words)
        top_n: max results to return

    Returns:
        list of dicts: [{product_id, brand_name, product_title, price, avg_product_rating,
                        review_count, score}, ...]
    """
    if not query or not query.strip():
        return []

    # Strategy 1: String matching
    string_match_ids = normalized_string_match(query, products_df)

    # Strategy 2: Semantic search
    semantic_results = semantic_search(query, products_df, product_vectors, embeddings_dict)

    # Fusion: combine scores
    # String matches get score=1.0, semantic matches get their cosine score
    # If a product appears in both, take max score
    all_scores = {}

    for product_id in string_match_ids:
        all_scores[product_id] = 1.0

    for product_id, score in semantic_results.items():
        if product_id in all_scores:
            all_scores[product_id] = max(all_scores[product_id], score)
        else:
            all_scores[product_id] = score

    # Build result list with product details
    results = []
    for product_id, score in all_scores.items():
        product = products_df[products_df['product_id'] == product_id]
        if product.empty:
            continue

        product = product.iloc[0]
        results.append({
            'product_id': product_id,
            'brand_name': product['brand_name'],
            'product_title': product['product_title'],
            'price': product['price'],
            'avg_product_rating': product['avg_product_rating'],
            'review_count': product.get('review_count', 0),
            'score': score
        })

    # Sort by score descending and limit to top_n
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


def get_product_details(product_id: int, products_df: pd.DataFrame, reviews_df: pd.DataFrame) -> Optional[Dict]:
    """
    Get full product details including all reviews.

    Args:
        product_id: product identifier
        products_df: DataFrame with product information
        reviews_df: DataFrame with all reviews

    Returns:
        dict with product details and list of reviews, or None if not found
    """
    product = products_df[products_df['product_id'] == product_id]
    if product.empty:
        return None

    product = product.iloc[0]
    product_reviews = reviews_df[reviews_df['product_id'] == product_id]

    reviews_list = []
    for _, review in product_reviews.iterrows():
        reviews_list.append({
            'review_title': review.get('review_title', ''),
            'review_text': review.get('review_text', ''),
            'rating': review.get('rating', 0),
            'skin_tone': review.get('skin_tone', ''),
            'skin_type': review.get('skin_type', ''),
            'is_buyer': review.get('is_buyer', None)
        })

    return {
        'product_id': product_id,
        'brand_name': product['brand_name'],
        'product_title': product['product_title'],
        'price': product['price'],
        'avg_product_rating': product['avg_product_rating'],
        'product_rating_count': product.get('product_rating_count', 0),
        'product_url': product.get('product_url', ''),
        'reviews': reviews_list
    }

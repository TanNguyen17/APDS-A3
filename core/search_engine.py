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


def normalized_string_match(query: str, products_df: pd.DataFrame, similarity_threshold: float = 0.7) -> Dict[int, float]:
    """
    Strategy 1: Weighted string matching with brand and title relevance.
    
    Args:
        query: raw user search string
        products_df: DataFrame with product information
        similarity_threshold: minimum similarity ratio for fuzzy matching
        
    Returns:
        Dict mapping product_id to relevancy score (0.0 to 1.0)
    """
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 1]
    if not query_words:
        return {}

    results = {}
    
    # Identify if any query word is a brand match
    all_brands = set(products_df['brand_name'].str.lower().unique())
    brand_query_words = [w for w in query_words if w in all_brands]

    for _, row in products_df.iterrows():
        brand_lower = str(row['brand_name']).lower()
        title_lower = str(row['product_title']).lower()
        
        # Calculate word-based match stats
        matches = [word for word in query_words if word in title_lower or word in brand_lower]
        if not matches:
            # Try fuzzy brand match as fallback
            brand_first_word = brand_lower.split()[0] if brand_lower else ""
            query_first_word = query_words[0]
            sim = SequenceMatcher(None, query_first_word, brand_first_word).ratio()
            if sim >= similarity_threshold:
                # Fuzzy brand match gets a low base score
                results[row['product_id']] = 0.3 * sim
            continue

        # Scoring Logic:
        # 1. Base score = % of query words found
        match_ratio = len(matches) / len(query_words)
        
        # 2. Category match bonus (if query has multiple words and we match the 'non-brand' parts)
        category_words = [w for w in query_words if w not in brand_query_words]
        category_match_ratio = 0
        if category_words:
            category_matches = [w for w in category_words if w in title_lower]
            category_match_ratio = len(category_matches) / len(category_words)
        
        # 3. Brand alignment bonus
        brand_match = 0
        if any(w in brand_lower for w in query_words):
            brand_match = 1.0

        # Final score calculation:
        # - High weight on matching ALL words (match_ratio)
        # - Medium weight on matching the 'category/product' part (category_match_ratio)
        # - Small weight on brand alignment
        score = (0.5 * match_ratio) + (0.4 * category_match_ratio) + (0.1 * brand_match)
        
        # Exact match for everything gets a slight boost to 1.0
        if len(matches) == len(query_words) and brand_match > 0:
            score = 1.0
            
        results[row['product_id']] = float(score)

    return results


def semantic_search(query: str, products_df: pd.DataFrame, product_vectors: Dict[int, np.ndarray],
                    embeddings_dict: Dict[str, np.ndarray], similarity_threshold: float = 0.45) -> Dict[int, float]:
    """
    Strategy 2: Semantic search using GloVe embeddings and cosine similarity.

    Args:
        query: raw user search string
        products_df: DataFrame with product information
        product_vectors: dict mapping product_id to GloVe vectors
        embeddings_dict: full GloVe embeddings dictionary
        similarity_threshold: minimum cosine similarity score

    Returns:
        Dict mapping product_id to similarity score
    """
    query_embedding = get_query_embedding(query, embeddings_dict)

    if query_embedding is None:
        return {}

    results = {}
    query_embedding = query_embedding.reshape(1, -1)

    for product_id, product_vec in product_vectors.items():
        # Optimization: only check products that actually exist in current dataframe
        if product_id not in products_df['product_id'].values:
            continue

        product_vec = product_vec.reshape(1, -1)
        similarity = cosine_similarity(query_embedding, product_vec)[0][0]

        if similarity >= similarity_threshold:
            # Semantic search results are capped at 0.9 to ensure string matches win
            results[product_id] = float(similarity) * 0.9

    return results


def search_products(query: str, products_df: pd.DataFrame, product_vectors: Dict[int, np.ndarray],
                   embeddings_dict: Dict[str, np.ndarray], top_n: int = 20) -> List[Dict]:
    """
    Multi-strategy product search with improved relevancy ranking.

    Args:
        query: raw user search string
        products_df: DataFrame with product information
        product_vectors: dict mapping product_id to GloVe vectors
        embeddings_dict: full GloVe embeddings dictionary
        top_n: maximum number of results to return

    Returns:
        List of dictionaries containing product details and relevancy scores
    """
    if not query or not query.strip():
        return []

    # Get results from both strategies (now both return scores)
    string_results = normalized_string_match(query, products_df)
    semantic_results = semantic_search(query, products_df, product_vectors, embeddings_dict)

    # Fusion: combine scores using max
    all_scores = {}
    for pid, score in string_results.items():
        all_scores[pid] = score
    
    for pid, score in semantic_results.items():
        if pid in all_scores:
            all_scores[pid] = max(all_scores[pid], score)
        else:
            all_scores[pid] = score

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

    # Sort by score descending and limit
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

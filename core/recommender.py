"""
Task 3: Item Recommendation System

Hybrid similarity-based product recommendation using:
- Text similarity (GloVe embeddings)
- Brand matching
- Price similarity
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_recommendations(product_id, product_vectors, products_df, n=5):
    """
    Get top-N similar products using hybrid similarity.

    Args:
        product_id: the selected product's ID
        product_vectors: dict[product_id → np.array(300)] pre-computed mean GloVe vectors
        products_df: DataFrame with columns: product_id, brand_name, product_title,
                     price, avg_product_rating, product_rating_count, review_count
        n: number of recommendations

    Returns:
        list of dicts: [{product_id, brand_name, product_title, price,
                         avg_product_rating, review_count, similarity_score}, ...]
    """
    if product_id not in product_vectors:
        return []

    target_row = products_df[products_df['product_id'] == product_id]
    if target_row.empty:
        return []

    target_vector = product_vectors[product_id].reshape(1, -1)
    target_brand = target_row.iloc[0]['brand_name']
    target_price = float(target_row.iloc[0]['price'])

    # Build matrices for vectorized computation
    other_ids = [pid for pid in product_vectors if pid != product_id]
    if not other_ids:
        return []

    other_matrix = np.array([product_vectors[pid] for pid in other_ids])

    # Vectorized cosine similarity
    text_sims = cosine_similarity(target_vector, other_matrix)[0]

    # Build product lookup for fast access
    prod_lookup = products_df.set_index('product_id')

    candidates = []
    for i, other_id in enumerate(other_ids):
        if other_id not in prod_lookup.index:
            continue

        other_row = prod_lookup.loc[other_id]
        other_price = float(other_row['price'])

        # Brand match
        brand_match = 1.0 if target_brand == other_row['brand_name'] else 0.0

        # Price similarity
        if target_price == 0 and other_price == 0:
            price_sim = 1.0
        elif target_price == 0 or other_price == 0:
            price_sim = 0.0
        else:
            price_diff = abs(target_price - other_price)
            price_max = max(target_price, other_price, 1)
            price_sim = max(0.0, 1.0 - price_diff / price_max)

        # Hybrid score
        final_score = 0.7 * float(text_sims[i]) + 0.15 * brand_match + 0.15 * price_sim

        candidates.append({
            'product_id': other_id,
            'brand_name': other_row['brand_name'],
            'product_title': other_row['product_title'],
            'price': other_row['price'],
            'avg_product_rating': other_row['avg_product_rating'],
            'review_count': other_row['review_count'],
            'similarity_score': final_score
        })

    candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
    return candidates[:n]

"""
Task 1: Item Search Engine
Multi-strategy product search with string matching and semantic similarity.

Improvements applied:
  1. Inverted index  — word→product_id lookup replaces O(n) full DataFrame scan
  2. Query expansion — NLTK WordNet synonyms + cosmetics domain map
  3. Min-max normalisation — aligns semantic and string score scales before fusion
  4. OOV-aware weighting — semantic search suppressed when >50 % of tokens are OOV
"""

import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Optional, Tuple

# ── WordNet setup ────────────────────────────────────────────────────────────
try:
    from nltk.corpus import wordnet as _wordnet
    _wordnet.synsets("test")           # force corpus load; raises LookupError if missing
    _WORDNET_AVAILABLE = True
except LookupError:
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    try:
        from nltk.corpus import wordnet as _wordnet
        _WORDNET_AVAILABLE = True
    except Exception:
        _WORDNET_AVAILABLE = False
except ImportError:
    _WORDNET_AVAILABLE = False

# Cosmetics-domain synonyms that WordNet handles poorly
_COSMETICS_MAP: Dict[str, List[str]] = {
    "spf":          ["sunscreen", "sunblock", "sun protection", "uv protection"],
    "sunscreen":    ["spf", "sunblock", "sun protection"],
    "bb":           ["bb cream", "blemish balm", "tinted moisturiser"],
    "cc":           ["cc cream", "colour correcting", "color correcting"],
    "moisturiser":  ["moisturizer", "hydrating", "hydration", "lotion"],
    "moisturizer":  ["moisturiser", "hydrating", "hydration", "lotion"],
    "toner":        ["toning", "astringent", "essence"],
    "serum":        ["essence", "treatment", "ampoule"],
    "primer":       ["base", "pore minimiser", "pore minimizer"],
    "foundation":   ["coverage", "base makeup", "complexion"],
    "lipstick":     ["lip colour", "lip color", "lip product"],
    "eyeliner":     ["eye liner", "kohl", "kajal"],
    "mascara":      ["lash", "lashes", "volume"],
    "concealer":    ["corrector", "coverage", "undereye"],
    "highlighter":  ["illuminator", "glow", "strobing"],
    "blush":        ["blusher", "cheek colour", "cheek color"],
    "bronzer":      ["bronzing", "contour", "sun-kissed"],
}


# ── 1. Inverted Index ────────────────────────────────────────────────────────

def build_inverted_index(products_df: pd.DataFrame) -> Dict[str, set]:
    """
    Build word → set[product_id] mapping over brand names and product titles.
    Call once at startup; pass the result into search_products as inverted_index.
    """
    index: Dict[str, set] = {}
    for _, row in products_df.iterrows():
        pid = row["product_id"]
        text = (str(row["brand_name"]) + " " + str(row["product_title"])).lower()
        for word in text.split():
            if len(word) > 1:
                index.setdefault(word, set()).add(pid)
    return index


# ── 2. Query Expansion ───────────────────────────────────────────────────────

def expand_query(query: str) -> str:
    """
    Expand the query with domain synonyms and WordNet lemmas.
    Returns the original query string extended with additional terms.
    Only used for semantic search — string matching still uses the original query.
    """
    tokens = query.lower().split()
    extra: List[str] = []

    for token in tokens:
        if token in _COSMETICS_MAP:
            extra.extend(_COSMETICS_MAP[token])
            continue

        if _WORDNET_AVAILABLE and len(token) > 3:
            try:
                synsets = _wordnet.synsets(token, pos=[_wordnet.NOUN, _wordnet.ADJ])[:2]
                for syn in synsets:
                    for lemma in syn.lemmas()[:2]:
                        name = lemma.name().replace("_", " ")
                        if name != token and len(name) > 2:
                            extra.append(name)
            except Exception:
                pass

    if extra:
        return query + " " + " ".join(extra)
    return query


# ── 3. OOV-aware embedding ───────────────────────────────────────────────────

def get_query_embedding(
    query: str, embeddings_dict: Dict[str, np.ndarray]
) -> Tuple[Optional[np.ndarray], float]:
    """
    Compute mean GloVe vector for the query.

    Returns:
        (embedding, oov_ratio) — embedding is None when ALL tokens are OOV.
        oov_ratio is the fraction of tokens not found in GloVe vocabulary.
    """
    tokens = query.lower().split()
    if not tokens:
        return None, 1.0

    vecs = []
    oov_count = 0
    for w in tokens:
        if w in embeddings_dict:
            vecs.append(embeddings_dict[w])
        else:
            oov_count += 1

    oov_ratio = oov_count / len(tokens)
    if vecs:
        return np.mean(vecs, axis=0), oov_ratio
    return None, 1.0


# ── Score normalisation helper ───────────────────────────────────────────────

def _minmax_normalize(
    scores: Dict[int, float], lo: float = 0.0, hi: float = 0.85
) -> Dict[int, float]:
    """Min-max scale a score dict into [lo, hi]."""
    if not scores:
        return scores
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {pid: hi for pid in scores}
    span = max_v - min_v
    return {pid: lo + (s - min_v) / span * (hi - lo) for pid, s in scores.items()}


# ── Strategy 1: Weighted String Matching ────────────────────────────────────

def normalized_string_match(
    query: str,
    products_df: pd.DataFrame,
    similarity_threshold: float = 0.7,
    inverted_index: Optional[Dict[str, set]] = None,
) -> Dict[int, float]:
    """
    Weighted string matching with brand and title relevance.

    When inverted_index is supplied, only candidate products (those containing
    at least one query word) are scored — O(k) instead of O(n).  The full
    O(n) fuzzy-brand fallback is only triggered when the index returns no hits.

    Returns:
        Dict mapping product_id to relevancy score (0.0 to 1.0)
    """
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 1]
    if not query_words:
        return {}

    all_brands = set(products_df["brand_name"].str.lower().unique())
    brand_query_words = [w for w in query_words if w in all_brands]

    results: Dict[int, float] = {}

    def _score_row(row) -> None:
        brand_lower = str(row["brand_name"]).lower()
        title_lower = str(row["product_title"]).lower()

        matches = [w for w in query_words if w in title_lower or w in brand_lower]
        if not matches:
            # Fuzzy brand fallback
            brand_fw = brand_lower.split()[0] if brand_lower else ""
            sim = SequenceMatcher(None, query_words[0], brand_fw).ratio()
            if sim >= similarity_threshold:
                results[row["product_id"]] = 0.3 * sim
            return

        match_ratio = len(matches) / len(query_words)

        category_words = [w for w in query_words if w not in brand_query_words]
        category_match_ratio = 0.0
        if category_words:
            cat_hits = [w for w in category_words if w in title_lower]
            category_match_ratio = len(cat_hits) / len(category_words)

        brand_match = 1.0 if any(w in brand_lower for w in query_words) else 0.0

        score = 0.5 * match_ratio + 0.4 * category_match_ratio + 0.1 * brand_match
        if len(matches) == len(query_words) and brand_match:
            score = 1.0

        results[row["product_id"]] = float(score)

    if inverted_index is not None:
        candidate_ids: set = set()
        for word in query_words:
            candidate_ids |= inverted_index.get(word, set())

        if candidate_ids:
            candidate_df = products_df[products_df["product_id"].isin(candidate_ids)]
            for _, row in candidate_df.iterrows():
                _score_row(row)
        else:
            # No index hits — full fuzzy scan (misspelled brand, etc.)
            for _, row in products_df.iterrows():
                _score_row(row)
    else:
        for _, row in products_df.iterrows():
            _score_row(row)

    return results


# ── Strategy 2: Semantic Search ──────────────────────────────────────────────

def semantic_search(
    query: str,
    products_df: pd.DataFrame,
    product_vectors: Dict[int, np.ndarray],
    embeddings_dict: Dict[str, np.ndarray],
    similarity_threshold: float = 0.45,
    _precomputed_embedding: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """
    Semantic search using GloVe embeddings and cosine similarity.
    Accepts a pre-computed query embedding to avoid redundant computation.
    Returns raw cosine scores (normalisation is applied in search_products).

    Returns:
        Dict mapping product_id to cosine similarity score
    """
    if _precomputed_embedding is not None:
        query_embedding = _precomputed_embedding
    else:
        query_embedding, _ = get_query_embedding(query, embeddings_dict)

    if query_embedding is None:
        return {}

    query_embedding = query_embedding.reshape(1, -1)
    valid_ids = set(products_df["product_id"])

    results: Dict[int, float] = {}
    for product_id, product_vec in product_vectors.items():
        if product_id not in valid_ids:
            continue
        sim = cosine_similarity(query_embedding, product_vec.reshape(1, -1))[0][0]
        if sim >= similarity_threshold:
            results[product_id] = float(sim)

    return results


# ── Fusion ───────────────────────────────────────────────────────────────────

def search_products(
    query: str,
    products_df: pd.DataFrame,
    product_vectors: Dict[int, np.ndarray],
    embeddings_dict: Dict[str, np.ndarray],
    top_n: int = 20,
    inverted_index: Optional[Dict[str, set]] = None,
) -> List[Dict]:
    """
    Multi-strategy product search with weighted relevancy ranking.

    Pipeline:
      1. Expand query (semantic only) via WordNet + domain synonyms
      2. Compute query embedding; derive OOV ratio
      3. Strategy 1: weighted string match on original query (fast via inverted index)
      4. Strategy 2: GloVe semantic search on expanded query (suppressed if OOV > 50 %)
      5. Min-max normalise semantic scores to [0, 0.85] so both scales are comparable
      6. Max-fusion: take the higher score when a product appears in both result sets

    Returns:
        List of product dicts sorted by descending relevancy score
    """
    if not query or not query.strip():
        return []

    # Step 1 & 2: expand for semantic, compute embedding once
    expanded_query = expand_query(query)
    query_embedding, oov_ratio = get_query_embedding(expanded_query, embeddings_dict)

    # Step 3: string matching on original query
    string_results = normalized_string_match(query, products_df, inverted_index=inverted_index)

    # Step 4: semantic search (suppressed when most tokens are OOV)
    semantic_results: Dict[int, float] = {}
    if oov_ratio <= 0.5 and query_embedding is not None:
        semantic_results = semantic_search(
            expanded_query,
            products_df,
            product_vectors,
            embeddings_dict,
            _precomputed_embedding=query_embedding,
        )

    # Step 5: normalise semantic scores to [0, 0.85]
    semantic_results = _minmax_normalize(semantic_results, lo=0.0, hi=0.85)

    # Step 6: max-fusion
    all_scores: Dict[int, float] = {**string_results}
    for pid, score in semantic_results.items():
        all_scores[pid] = max(all_scores.get(pid, 0.0), score)

    # Build result list
    results: List[Dict] = []
    for product_id, score in all_scores.items():
        product = products_df[products_df["product_id"] == product_id]
        if product.empty:
            continue
        product = product.iloc[0]
        results.append(
            {
                "product_id": product_id,
                "brand_name": product["brand_name"],
                "product_title": product["product_title"],
                "price": product["price"],
                "avg_product_rating": product["avg_product_rating"],
                "review_count": product.get("review_count", 0),
                "score": score,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ── Product Detail ───────────────────────────────────────────────────────────

def get_product_details(
    product_id: int, products_df: pd.DataFrame, reviews_df: pd.DataFrame
) -> Optional[Dict]:
    """
    Get full product details including all reviews.

    Returns:
        dict with product details and list of reviews, or None if not found
    """
    product = products_df[products_df["product_id"] == product_id]
    if product.empty:
        return None

    product = product.iloc[0]
    product_reviews = reviews_df[reviews_df["product_id"] == product_id]

    reviews_list = []
    for _, review in product_reviews.iterrows():
        reviews_list.append(
            {
                "review_title": review.get("review_title", ""),
                "review_text": review.get("review_text", ""),
                "rating": review.get("rating", 0),
                "skin_tone": review.get("skin_tone", ""),
                "skin_type": review.get("skin_type", ""),
                "is_buyer": review.get("is_buyer", None),
            }
        )

    return {
        "product_id": product_id,
        "brand_name": product["brand_name"],
        "product_title": product["product_title"],
        "price": product["price"],
        "avg_product_rating": product["avg_product_rating"],
        "product_rating_count": product.get("product_rating_count", 0),
        "product_url": product.get("product_url", ""),
        "reviews": reviews_list,
    }

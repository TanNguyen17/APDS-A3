"""
classifier.py

Task 2: Buyer/Non-Buyer prediction using 3-model ensemble with soft voting.

This module provides prediction functionality for the buyer/non-buyer classification
using three pre-trained models:
- Model A: MLP + GloVe embeddings (text only, unweighted mean pooling)
- Model B: MLP + BoW (review text + title, no metadata)
- Model C: Random Forest + GloVe embeddings (text + title, unweighted mean pooling) + metadata

The final prediction uses soft voting (probability averaging) across all three models.

predict_proba Implementation Notes:
- All models use predict_proba() which returns [P(non-buyer), P(buyer)] array for binary classification
- We extract index [1] to get P(buyer) probability used in the ensemble
- Soft voting averages these three P(buyer) probabilities to produce the final prediction
- Final prediction: buyer if average probability >= 0.5, non-buyer otherwise
"""

import pandas as pd
import numpy as np
from core.preprocessing import preprocess_text


def predict_buyer(review_text, review_title, review_rating, product_info,
                  model_a, model_b, model_c, label_encoder,
                  colloc_dict, stop_words, lemmatizer=None):
    """
    Predict buyer/non-buyer using 3-model ensemble with soft voting.

    Args:
        review_text: raw review text from user
        review_title: raw review title from user
        review_rating: int 1-5
        product_info: dict with keys: price, avg_product_rating, product_rating_count, brand_name
        model_a, model_b, model_c: loaded sklearn pipelines
        label_encoder: fitted LabelEncoder for brand_name
        colloc_dict: collocation dictionary for preprocessing
        stop_words: set of stopwords
        lemmatizer: optional WordNetLemmatizer instance

    Returns:
        dict with keys:
            model_a_pred: 0 or 1
            model_a_prob: float (probability of buyer)
            model_b_pred: 0 or 1
            model_b_prob: float
            model_c_pred: 0 or 1
            model_c_prob: float
            fused_pred: 0 or 1
            fused_prob: float
            label: "Buyer" or "Non-Buyer"
    """

    # Step 1: Preprocess text fields
    processed_text = preprocess_text(
        review_text,
        colloc_dict=colloc_dict,
        stop_words=stop_words,
        lemmatizer=lemmatizer
    )

    processed_title = preprocess_text(
        review_title,
        colloc_dict=colloc_dict,
        stop_words=stop_words,
        lemmatizer=lemmatizer
    )

    # Step 2: Encode brand name with fallback for unseen brands
    try:
        brand_encoded = label_encoder.transform([product_info['brand_name']])[0]
    except (ValueError, KeyError):
        # Unseen brand or missing brand_name -> fallback to 0
        brand_encoded = 0

    # Step 3: Model A prediction (MLP + GloVe embeddings, text only)
    model_a_proba = model_a.predict_proba([processed_text])[0]
    model_a_prob = model_a_proba[1]  # Extract P(buyer) probability (index 1)
    model_a_pred = 1 if model_a_prob >= 0.5 else 0

    # Step 4: Construct DataFrame for Model B and Model C
    input_df = pd.DataFrame({
        'processed_review_text': [processed_text],
        'processed_title': [processed_title],
        'price': [product_info['price']],
        'avg_product_rating': [product_info['avg_product_rating']],
        'product_rating_count': [product_info['product_rating_count']],
        'review_rating': [review_rating],
        'brand_encoded': [brand_encoded]
    })

    # Step 5: Model B prediction (MLP + BoW, text + title only)
    model_b_proba = model_b.predict_proba(input_df)[0]
    model_b_prob = model_b_proba[1]  # Extract P(buyer) probability (index 1)
    model_b_pred = 1 if model_b_prob >= 0.5 else 0

    # Step 6: Model C prediction (Random Forest + GloVe embeddings + metadata)
    model_c_proba = model_c.predict_proba(input_df)[0]
    model_c_prob = model_c_proba[1]  # Extract P(buyer) probability (index 1)
    model_c_pred = 1 if model_c_prob >= 0.5 else 0

    # Step 7: Ensemble fusion via soft voting (simple average of P(buyer) probabilities)
    fused_prob = np.mean([model_a_prob, model_b_prob, model_c_prob])
    fused_pred = 1 if fused_prob >= 0.5 else 0

    # Step 8: Convert to human-readable label
    label = "Buyer" if fused_pred == 1 else "Non-Buyer"

    return {
        'model_a_pred': model_a_pred,
        'model_a_prob': float(model_a_prob),
        'model_b_pred': model_b_pred,
        'model_b_prob': float(model_b_prob),
        'model_c_pred': model_c_pred,
        'model_c_prob': float(model_c_prob),
        'fused_pred': fused_pred,
        'fused_prob': float(fused_prob),
        'label': label
    }

# GlamReview — Cosmetics Review Platform

A Gradio-based web application for browsing, reviewing, and analyzing cosmetics/beauty products. Built for COSC3801/3015 Advanced Programming for Data Science (Milestone II).

## Features

| Task | Description |
|------|-------------|
| **Search** | Multi-strategy product search (string matching + GloVe semantic similarity). Handles variant forms ("Maybeline" ≈ "Maybelline New York") |
| **Write Review** | Create reviews with 3-model ensemble buyer prediction (BoW LR + BoW+Meta LR + GloVe+Meta LR), soft-voting fusion, user override |
| **Recommendations** | Content-based similar products using hybrid similarity (70% text + 15% brand + 15% price) on GloVe embeddings |
| **Sentiment Insights** | Per-product sentiment breakdown, key terms extraction, and two-product comparative analysis |

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (first time only)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4. Train models and generate pickle files (first time only, ~60s)
python scripts/train_models.py

# 5. Generate product images (first time only, ~30s)
python scripts/image_scraper.py

# 6. Launch the web app
python app.py
```

Open http://localhost:7860 in your browser.

## Project Structure

```
APDS-A3/
├── app.py                  # Entry point: load data, compose tabs, launch Gradio
├── core/                   # Backend logic modules
│   ├── data_loader.py            # Load CSV, GloVe, vocab, pickles at startup
│   ├── preprocessing.py          # Text pipeline + custom sklearn transformers
│   ├── classifier.py             # Task 2: 3-model ensemble buyer prediction
│   ├── search_engine.py          # Task 1: multi-strategy product search
│   ├── recommender.py            # Task 3: hybrid content-based recommendations
│   └── sentiment.py              # Task 4: sentiment dashboard + comparison
├── ui/                     # Gradio UI modules
│   ├── components.py             # Shared render functions (card, grid, predictions)
│   ├── product_detail.py         # Product detail page renderer
│   ├── recommendation_grid.py    # Recommendation results grid renderer
│   ├── search_tab.py             # Tab 1: Search Products
│   ├── review_tab.py             # Tab 2: Write a Review
│   ├── recommendation_tab.py     # Tab 3: Similar Products
│   └── sentiment_tab.py          # Tab 4: Sentiment Insights
├── static/                 # Frontend assets
│   ├── styles.css                # Custom CSS
│   └── scripts.js                # Client-side JS (star rating, filters)
├── scripts/                # Offline utility scripts
│   ├── train_models.py           # Train & save models as pkl
│   └── image_scraper.py          # Generate product placeholder images
├── models/                 # Pre-trained model pickle files
│   ├── model_a_bow.pkl           # BoW + Logistic Regression (text only)
│   ├── model_b_bow_meta.pkl      # BoW + title + metadata + LR
│   ├── model_c_glove_meta.pkl    # GloVe + metadata + LR
│   ├── label_encoder.pkl         # Brand name encoder (11 brands)
│   ├── collocation_dict.pkl      # 111 bigram patterns for preprocessing
│   └── product_vectors.pkl       # 295 product-level GloVe embeddings
├── images/                 # Product thumbnail images (295 files)
├── notebooks/              # Milestone I notebooks and data
│   ├── task1_report.ipynb        # Preprocessing pipeline
│   ├── task2_3_report.ipynb      # Classification models
│   ├── processed.csv             # 61,275 preprocessed reviews
│   ├── glove.6B.300d.txt         # GloVe 300d embeddings (~1GB)
│   ├── vocab.txt                 # 7,366 word vocabulary
│   └── stopwords_en.txt          # 570 stopwords
├── requirements.txt
└── README.md
```

## Data

- **Dataset**: 61,275 cosmetics/beauty product reviews from Nykaa.com
- **Products**: 295 unique products across 11 brands
- **Labels**: Binary buyer/non-buyer classification (78.7% / 21.3%)
- **Embeddings**: GloVe 6B 300d (400K words, loaded fully at runtime)

## Models (HD: 3 different data types, fused)

| Model | Features | Algorithm | F1 (macro) |
|-------|----------|-----------|------------|
| A | BoW text only (sparse, 7366d) | Logistic Regression | 0.56 |
| B | BoW text + title + metadata (sparse + numeric) | Logistic Regression | 0.66 |
| C | GloVe embeddings + title + metadata (dense 300d + numeric) | Logistic Regression | 0.68 |

**Fusion**: Soft-voting — average predicted probabilities from all 3 models, threshold at 0.5.

## Text Preprocessing Pipeline

1. Collocation replacement (111 bigram patterns, e.g. "dry skin" → "dry-skin")
2. Regex tokenization: `[a-zA-Z]+(?:[-'][a-zA-Z]+)?`
3. Short word removal (length < 2)
4. Stopword removal (570 words)
5. Lemmatization (WordNet)

## Commands Reference

```bash
# Run the app
python app.py

# Retrain models (if data changes)
python scripts/train_models.py

# Regenerate product images
python scripts/image_scraper.py
```

## Requirements

- Python 3.12+
- ~2GB RAM (GloVe embeddings + models)
- ~1.2GB disk (GloVe file + data + models)
- Startup time: ~10s (GloVe loading dominates)

## Team

RMIT University — COSC3801/3015 Advanced Programming for Data Science, 2026 Semester 1

## Notes

- New reviews created via Task 2 persist in-memory only (lost on app restart)
- Product images are generated placeholders (Nykaa URLs used as fallback source)
- GloVe embeddings must be present at `notebooks/glove.6B.300d.txt` before running

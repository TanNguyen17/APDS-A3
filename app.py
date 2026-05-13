"""
GlamReview - Cosmetics Review Platform
Main Gradio web application for APDS-A3 Milestone 2
"""

import os
import gradio as gr
from core.data_loader import load_all, new_reviews

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
gr.set_static_paths(paths=[IMAGES_DIR])

# Load static assets
with open(os.path.join(BASE_DIR, "static", "styles.css")) as f:
    CUSTOM_CSS = f.read()
with open(os.path.join(BASE_DIR, "static", "scripts.js")) as f:
    HEAD_JS = "<script>\n" + f.read() + "\n</script>"

# ============================================================================
# LOAD ALL DATA ASSETS
# ============================================================================

print("=" * 70)
print("STARTING GLAMREVIEW WEB APPLICATION")
print("=" * 70)

data = load_all()

df = data['df']
vocab_dict = data['vocab']
stop_words = data['stopwords']
embeddings_dict = data['glove_embeddings']
models = data['models']
products_df = data['products_df']
reviews_by_product = data['reviews_by_product']

product_vectors = models['product_vectors']

print("\n" + "=" * 70)
print("DATA LOADED SUCCESSFULLY")
print("=" * 70)
print(f"Total products: {len(products_df):,}")
print(f"Total reviews: {len(df):,}")
print(f"GloVe vocabulary size: {len(embeddings_dict):,}")
print("=" * 70 + "\n")

# ============================================================================
# PRODUCT DROPDOWN CHOICES
# ============================================================================

product_choices = []
for _, row in products_df.iterrows():
    product_id = row['product_id']
    brand = row['brand_name']
    title = row['product_title']
    display_title = title[:60] + "..." if len(title) > 60 else title
    display_str = f"{display_title} | {brand}"
    product_choices.append((display_str, product_id))

# Shared context dict passed to UI modules
ctx = {
    'products_df': products_df,
    'reviews_by_product': reviews_by_product,
    'new_reviews': new_reviews,
    'images_dir': IMAGES_DIR,
    'product_vectors': product_vectors,
    'get_recommendations': __import__('core.recommender', fromlist=['get_recommendations']).get_recommendations,
}

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

from ui.search_tab import build_search_tab
from ui.review_tab import build_review_tab
from ui.recommendation_tab import build_recommendation_tab
from ui.sentiment_tab import build_sentiment_tab

BANNER_HTML = """
<div class="full-width-banner shopee-header">
    <!-- Top utility bar -->
    <div class="shopee-top-bar">
        <div class="shopee-top-left">
            <span>Seller Centre</span>
            <span class="shopee-divider">|</span>
            <span>Download App</span>
            <span class="shopee-divider">|</span>
            <span>Connect</span>
        </div>
        <div class="shopee-top-right">
            <span>Notifications</span>
            <span class="shopee-divider">|</span>
            <span>Help</span>
            <span class="shopee-divider">|</span>
            <span>English</span>
        </div>
    </div>
    <!-- Main bar with logo -->
    <div class="shopee-main-bar">
        <div class="shopee-logo">
            <h1>GLAMREVIEW</h1>
            <span class="shopee-tagline">Premium Beauty Destination</span>
        </div>
        <div class="shopee-right-links">
            <span>Free Shipping on Orders Over ₹500</span>
        </div>
    </div>
</div>
"""

with gr.Blocks(title="GlamReview — Cosmetics") as demo:
    gr.HTML(BANNER_HTML)

    with gr.Tabs():
        build_search_tab(products_df, product_vectors, embeddings_dict, ctx)
        build_review_tab(product_choices, products_df, models, stop_words, ctx)
        build_recommendation_tab(product_choices, products_df, product_vectors, ctx)
        build_sentiment_tab(product_choices, df, products_df)

# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LAUNCHING GRADIO WEB APPLICATION")
    print("=" * 70)
    print("Server will be available at:")
    print("  - Local: http://localhost:7860")
    print("  - Network: http://0.0.0.0:7860")
    print("=" * 70 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
        head=HEAD_JS,
        allowed_paths=[IMAGES_DIR]
    )

"""
GlamReview - Cosmetics Review Platform
Main Gradio web application for APDS-A3 Milestone 2
"""

import os
import html as html_module
import gradio as gr
from fastapi import Form
from fastapi.responses import HTMLResponse
from core.data_loader import load_all, new_reviews, set_df, get_df, pending_reviews, pop_pending_review, add_review
from core.recommender import get_recommendations
from ui.search_tab import build_search_tab
from ui.review_tab import build_review_tab
from ui.recommendation_tab import build_recommendation_tab
from ui.sentiment_tab import build_sentiment_tab

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
set_df(df)
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
    'get_recommendations': get_recommendations,
}

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

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

with gr.Blocks(title="GlamReview — Cosmetics", css=CUSTOM_CSS, head=HEAD_JS) as demo:
    gr.HTML(BANNER_HTML)

    with gr.Tabs(elem_classes=["glamreview-tabs"]):
        build_search_tab(products_df, product_vectors, embeddings_dict, ctx)
        build_review_tab(product_choices, products_df, models, stop_words, ctx)
        build_recommendation_tab(product_choices, products_df, product_vectors, ctx)
        build_sentiment_tab(product_choices, df, products_df)

# ============================================================================
# REVIEW DETAIL PAGE
# ============================================================================

def _stars_html(rating_val):
    full = int(float(rating_val or 0))
    empty = 5 - full
    filled = '★'
    outline = '☆'
    return f'<span style="color:#EE4D2D;font-size:20px;">{filled * full}{outline * empty}</span>'


def render_review_page(review: dict) -> str:
    r_title = html_module.escape(str(review.get('review_title', '') or ''))
    r_text = html_module.escape(str(review.get('review_text', '') or ''))
    r_author = html_module.escape(str(review.get('author', 'Anonymous') or 'Anonymous'))
    r_date = html_module.escape(str(review.get('review_date', '') or ''))
    r_rating = float(review.get('review_rating', 0) or review.get('rating', 0) or 0)
    product_title = html_module.escape(str(review.get('product_title', '') or ''))
    brand_name = html_module.escape(str(review.get('brand_name', '') or ''))
    is_buyer = review.get('is_a_buyer', review.get('is_buyer', None))
    buyer_badge = '<span style="background:#EE4D2D;color:#fff;padding:2px 8px;border-radius:3px;font-size:12px;margin-left:8px;">Verified Buyer</span>' if is_buyer in [1, True, 'True'] else ''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Review — {r_title or "GlamReview"}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f5; margin: 0; padding: 24px; }}
  .card {{ background: #fff; border-radius: 8px; max-width: 700px; margin: 0 auto; padding: 32px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
  .back-link {{ display: inline-block; margin-bottom: 20px; color: #EE4D2D; text-decoration: none; font-size: 14px; }}
  .back-link:hover {{ text-decoration: underline; }}
  .product-label {{ font-size: 13px; color: #757575; margin-bottom: 4px; }}
  .product-name {{ font-size: 18px; font-weight: 600; color: #222; margin-bottom: 16px; }}
  .review-meta {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
  .avatar {{ width: 40px; height: 40px; border-radius: 50%; background: #EE4D2D; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 18px; flex-shrink: 0; }}
  .author {{ font-weight: 600; color: #222; }}
  .date {{ font-size: 13px; color: #999; }}
  .review-title {{ font-size: 16px; font-weight: 600; color: #222; margin-bottom: 12px; }}
  .review-text {{ font-size: 15px; color: #444; line-height: 1.7; white-space: pre-wrap; }}
</style>
</head>
<body>
<div class="card">
  <a class="back-link" href="/">← Back to GlamReview</a>
  <div class="product-label">{brand_name}</div>
  <div class="product-name">{product_title}</div>
  <div class="review-meta">
    <div class="avatar">{r_author[0].upper() if r_author else 'A'}</div>
    <div>
      <div class="author">{r_author}{buyer_badge}</div>
      <div class="date">{r_date}</div>
    </div>
    <div>{_stars_html(r_rating)} {r_rating:.1f}</div>
  </div>
  <div class="review-title">{r_title}</div>
  <div class="review-text">{r_text}</div>
</div>
</body>
</html>'''


@demo.app.get("/review/{review_id}")
def view_review_page(review_id: str):
    review = next((r for r in new_reviews if str(r.get('review_id', '')) == review_id), None)
    if review is None:
        source_df = get_df()
        if source_df is not None:
            matches = source_df[source_df['review_id'].astype(str) == review_id]
            if not matches.empty:
                review = matches.iloc[0].to_dict()
    if review is None:
        return HTMLResponse("<h2 style='font-family:sans-serif;padding:40px;'>Review not found.</h2>", status_code=404)
    return HTMLResponse(render_review_page(review))


# ============================================================================
# REVIEW CONFIRMATION PAGE
# ============================================================================

_CONFIRM_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; min-height: 100vh; }
.site-header { background: linear-gradient(180deg, #f53d2d, #f63); padding: 12px 40px; display: flex; align-items: center; justify-content: space-between; }
.site-logo { color: #fff; font-size: 22px; font-weight: 800; letter-spacing: 1px; text-decoration: none; }
.site-nav a { color: rgba(255,255,255,0.85); text-decoration: none; margin-left: 24px; font-size: 14px; }
.site-nav a:hover { color: #fff; }
.page-body { max-width: 760px; margin: 40px auto; padding: 0 20px 60px; }
h1 { font-size: 24px; font-weight: 700; color: #222; margin-bottom: 24px; }
.review-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 24px; margin-bottom: 20px; }
.review-card p { font-size: 15px; color: #333; line-height: 1.6; margin-bottom: 10px; }
.review-card p:last-child { margin-bottom: 0; }
.review-card strong { color: #111; }
.description-block { margin-top: 4px; }
.prediction-box { background: #e8f4fd; border: 1px solid #b3d9f5; border-radius: 6px; padding: 16px 20px; margin-bottom: 20px; font-size: 15px; color: #1a5276; }
.prediction-box strong { font-weight: 700; }
.question-section { margin-bottom: 24px; }
.question-section p { font-size: 15px; font-weight: 500; color: #222; margin-bottom: 12px; }
.radio-group label { display: flex; align-items: center; gap: 8px; font-size: 15px; color: #333; margin-bottom: 10px; cursor: pointer; }
.radio-group input[type=radio] { width: 17px; height: 17px; accent-color: #EE4D2D; cursor: pointer; }
.btn-row { display: flex; gap: 12px; margin-top: 8px; }
.btn-back { padding: 10px 22px; border: 1px solid #ccc; background: #fff; border-radius: 4px; font-size: 14px; cursor: pointer; color: #444; text-decoration: none; display: inline-flex; align-items: center; }
.btn-back:hover { background: #f5f5f5; }
.btn-confirm { padding: 10px 22px; background: #EE4D2D; color: #fff; border: none; border-radius: 4px; font-size: 14px; font-weight: 600; cursor: pointer; }
.btn-confirm:hover { background: #d44326; }
"""

_SUCCESS_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
.toast-card { background: #fff; border-radius: 8px; padding: 40px 48px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1); max-width: 420px; width: 100%; }
.check { font-size: 48px; margin-bottom: 16px; }
h2 { font-size: 20px; font-weight: 700; color: #222; margin-bottom: 8px; }
p { font-size: 14px; color: #757575; }
"""


def render_confirm_page(review: dict) -> str:
    product_title = html_module.escape(str(review.get('product_title', '') or ''))
    brand_name = html_module.escape(str(review.get('brand_name', '') or ''))
    r_title = html_module.escape(str(review.get('review_title', '') or ''))
    r_text = html_module.escape(str(review.get('review_text', '') or ''))
    r_rating = int(float(review.get('review_rating', 0) or review.get('rating', 0) or 0))
    temp_id = html_module.escape(str(review.get('_temp_id', '')))
    predicted_label = str(review.get('predicted_label', 'Buyer'))

    yes_checked = 'checked' if predicted_label == 'Buyer' else ''
    no_checked = 'checked' if predicted_label != 'Buyer' else ''
    pred_display = html_module.escape(predicted_label)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Confirm Your Review — GlamReview</title>
<style>{_CONFIRM_CSS}</style>
</head>
<body>
<header class="site-header">
  <a class="site-logo" href="/">GLAMREVIEW</a>
  <nav class="site-nav">
    <a href="/">Home</a>
    <a href="/">Catalog</a>
  </nav>
</header>
<div class="page-body">
  <h1>Confirm Your Review</h1>
  <div class="review-card">
    <p><strong>Item:</strong> {product_title} &mdash; {brand_name}</p>
    <p><strong>Title:</strong> {r_title}</p>
    <p><strong>Rating:</strong> {r_rating}/5</p>
    <p><strong>Description:</strong></p>
    <p class="description-block">{r_text}</p>
  </div>
  <div class="prediction-box">
    Based on your review, would you buy this product?<br>
    Our prediction: <strong>{pred_display}</strong>
  </div>
  <form method="post" action="/reviews/confirm/{temp_id}">
    <div class="question-section">
      <p>Would you buy this item?</p>
      <div class="radio-group">
        <label><input type="radio" name="chosen_label" value="Buyer" {yes_checked}> Yes</label>
        <label><input type="radio" name="chosen_label" value="Non-Buyer" {no_checked}> No</label>
      </div>
    </div>
    <div class="btn-row">
      <a class="btn-back" href="javascript:history.back()">Back</a>
      <button class="btn-confirm" type="submit">Confirm and Save</button>
    </div>
  </form>
</div>
</body>
</html>'''


def render_success_page() -> str:
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Review Saved — GlamReview</title>
<style>{_SUCCESS_CSS}</style>
<script>setTimeout(() => window.location.href = window.location.origin, 2000);</script>
</head>
<body>
<div class="toast-card">
  <div class="check">✅</div>
  <h2>Review submitted successfully!</h2>
  <p>Redirecting you back to the shop&hellip;</p>
</div>
</body>
</html>'''


@demo.app.get("/reviews/confirm/{temp_id}")
def get_confirm_page(temp_id: str):
    review = pending_reviews.get(temp_id)
    if review is None:
        return HTMLResponse(
            "<div style='font-family:sans-serif;padding:40px;'><h2>This confirmation link has expired or is invalid.</h2>"
            "<p><a href='/'>← Back to GlamReview</a></p></div>",
            status_code=404
        )
    return HTMLResponse(render_confirm_page(review))


@demo.app.post("/reviews/confirm/{temp_id}")
async def post_confirm_page(temp_id: str, chosen_label: str = Form(...)):
    review = pop_pending_review(temp_id)
    if review is None:
        return HTMLResponse(
            "<div style='font-family:sans-serif;padding:40px;'><h2>This review has already been submitted or the link expired.</h2>"
            "<p><a href='/'>← Back to GlamReview</a></p></div>",
            status_code=410
        )
    review['is_buyer'] = 1 if chosen_label == 'Buyer' else 0
    review.pop('_temp_id', None)
    add_review(review)
    return HTMLResponse(render_success_page())


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
        allowed_paths=[IMAGES_DIR]
    )

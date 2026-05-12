"""Reusable UI rendering components for the Gradio web application.

Contains functions that generate HTML fragments for product cards, grids,
model prediction displays, and key terms tables.
"""

import os


def render_product_card(product, images_dir, onclick_js=""):
    """Render a single product card with optional onclick JavaScript.

    Args:
        product: Dict with keys product_id, brand_name, product_title, price.
        images_dir: Absolute path to the directory containing product images.
        onclick_js: Optional HTML onclick attribute string.
    """
    product_id = product['product_id']
    brand = product['brand_name']
    title = product['product_title']
    price = product['price']

    display_title = title[:55] + "..." if len(title) > 55 else title

    img_path = os.path.join(images_dir, f"{product_id}.png")
    if not os.path.exists(img_path):
        img_inner = f'<div class="product-img" style="display:flex;align-items:center;justify-content:center;"><span style="color:#999;font-size:32px;font-weight:700;">{brand[0]}</span></div>'
    else:
        img_inner = f'<img src="/gradio_api/file={img_path}" class="product-img" alt="{brand}"/>'

    return f"""
    <div class="product-card" {onclick_js}>
        <div class="product-img-wrap">
            {img_inner}
        </div>
        <div class="product-info">
            <span class="brand">{brand}</span>
            <h4 class="title">{display_title}</h4>
            <div class="price">₹{price:.0f}</div>
        </div>
    </div>
    """


def render_product_grid(products, images_dir):
    """Render a grid of product cards.

    Args:
        products: List of product dicts.
        images_dir: Absolute path to the directory containing product images.
    """
    if not products:
        return "<p style='text-align: center; color: #666; padding: 40px;'>No products found.</p>"

    cards_html = []
    for product in products:
        pid = product["product_id"]
        onclick = f'onclick="setGradioValue(\'#selected_product_id\', \'{pid}_\' + Date.now())"'
        cards_html.append(render_product_card(product, images_dir, onclick))

    grid_html = '<div class="product-grid">' + ''.join(cards_html) + '</div>'
    return grid_html


def render_model_predictions(pred_result):
    """Render prediction results from all 3 models + fusion.

    Args:
        pred_result: Dict with keys model_a_pred, model_b_pred, model_c_pred,
            fused_pred, model_a_prob, model_b_prob, model_c_prob, fused_prob, label.
    """
    model_a_label = "Buyer" if pred_result['model_a_pred'] == 1 else "Non-Buyer"
    model_b_label = "Buyer" if pred_result['model_b_pred'] == 1 else "Non-Buyer"
    model_c_label = "Buyer" if pred_result['model_c_pred'] == 1 else "Non-Buyer"
    fused_label = pred_result['label']

    model_a_class = "buyer" if pred_result['model_a_pred'] == 1 else "non-buyer"
    model_b_class = "buyer" if pred_result['model_b_pred'] == 1 else "non-buyer"
    model_c_class = "buyer" if pred_result['model_c_pred'] == 1 else "non-buyer"
    fused_class = "buyer" if pred_result['fused_pred'] == 1 else "non-buyer"

    # Show confidence as probability of the PREDICTED class (not always buyer)
    model_a_conf = pred_result['model_a_prob'] if pred_result['model_a_pred'] == 1 else (1 - pred_result['model_a_prob'])
    model_b_conf = pred_result['model_b_prob'] if pred_result['model_b_pred'] == 1 else (1 - pred_result['model_b_prob'])
    model_c_conf = pred_result['model_c_prob'] if pred_result['model_c_pred'] == 1 else (1 - pred_result['model_c_prob'])
    fused_conf = pred_result['fused_prob'] if pred_result['fused_pred'] == 1 else (1 - pred_result['fused_prob'])

    html = """
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
    """

    html += f"""
    <div class="model-card">
        <div class="model-name">Model A — BoW (Text Only)</div>
        <div class="model-pred {model_a_class}">{model_a_label}</div>
        <div class="confidence">Confidence: {model_a_conf:.0%} | P(Buyer): {pred_result['model_a_prob']:.0%}</div>
    </div>
    """

    html += f"""
    <div class="model-card">
        <div class="model-name">Model B — BoW + Metadata</div>
        <div class="model-pred {model_b_class}">{model_b_label}</div>
        <div class="confidence">Confidence: {model_b_conf:.0%} | P(Buyer): {pred_result['model_b_prob']:.0%}</div>
    </div>
    """

    html += f"""
    <div class="model-card">
        <div class="model-name">Model C — GloVe Weighted + Metadata</div>
        <div class="model-pred {model_c_class}">{model_c_label}</div>
        <div class="confidence">Confidence: {model_c_conf:.0%} | P(Buyer): {pred_result['model_c_prob']:.0%}</div>
    </div>
    """

    html += f"""
    <div class="model-card" style="border: 2px solid #EE4D2D;">
        <div class="model-name" style="font-weight:700;">Fused Prediction (Ensemble)</div>
        <div class="model-pred {fused_class}" style="font-size: 22px;">{fused_label}</div>
        <div class="confidence">Confidence: {fused_conf:.0%} | P(Buyer): {pred_result['fused_prob']:.0%}</div>
    </div>
    """

    html += "</div>"
    return html


def render_key_terms_table(terms, sentiment_type):
    """Render a table of key terms.

    Args:
        terms: List of (term, count) tuples.
        sentiment_type: Either "positive" or "negative".
    """
    if not terms:
        return f"<p style='color: #666;'>No {sentiment_type} reviews found.</p>"

    border_class = "terms-positive" if sentiment_type == "positive" else "terms-negative"

    html = f"""
    <table class="terms-table {border_class}">
        <thead>
            <tr>
                <th>Rank</th>
                <th>Term</th>
                <th>Frequency</th>
            </tr>
        </thead>
        <tbody>
    """

    for i, (term, count) in enumerate(terms, 1):
        html += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{term}</strong></td>
            <td>{count}</td>
        </tr>
        """

    html += "</tbody></table>"
    return html

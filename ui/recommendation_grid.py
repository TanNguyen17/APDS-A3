import os

from ui.components import render_product_card


def render_recommendation_grid(selected_product, recommendations, images_dir):
    """Render selected product + recommendation cards with click-through."""
    html = "<h3 style='color: #333; margin: 20px 0 12px 0;'>Selected Product:</h3>"
    pid = selected_product['product_id']
    onclick = f'''onclick="setGradioValue('#selected_rec_product_id', '{pid}_' + Date.now())"'''
    html += '<div style="max-width: 300px; margin: 0 auto;">' + render_product_card(selected_product, images_dir, onclick) + '</div>'

    if not recommendations:
        html += "<p style='text-align: center; color: #666; padding: 40px;'>No similar products found.</p>"
        return html

    html += "<h3 style='color: #333; margin: 32px 0 12px 0;'>Similar Products:</h3>"
    html += '<div class="product-grid">'

    for rec in recommendations:
        rec_pid = rec['product_id']
        rec_img_path = os.path.join(images_dir, f"{rec_pid}.png")
        if os.path.exists(rec_img_path):
            rec_img_html = f'<img src="/gradio_api/file={rec_img_path}" class="product-img" alt="{rec["brand_name"]}"/>'
        else:
            rec_img_html = f'<div class="product-img" style="background:#f5f5f5;display:flex;align-items:center;justify-content:center;"><span style="color:#999;font-size:28px;font-weight:700;">{rec["brand_name"][0]}</span></div>'

        rec_onclick = f'''onclick="setGradioValue('#selected_rec_product_id', '{rec_pid}_' + Date.now())"'''

        card_html = f"""
        <div class="rec-card" style="cursor:pointer;" {rec_onclick}>
            {rec_img_html}
            <div class="product-info">
                <span class="brand">{rec['brand_name']}</span>
                <h4 class="title">{rec['product_title'][:50]}</h4>
                <div class="rating">★ {rec['avg_product_rating']:.1f} ({rec['review_count']} reviews)</div>
                <div class="price">₹{rec['price']:.0f}</div>
                <div class="similarity-score">Similarity: {rec['similarity_score']:.0%}</div>
            </div>
        </div>
        """
        html += card_html

    html += '</div>'
    return html

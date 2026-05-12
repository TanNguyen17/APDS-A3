import os
import html as html_module


def render_product_detail(product_id, ctx, target_elem_id="selected_product_id"):
    """Render full product detail page with image, info, reviews, and similar products."""
    products_df = ctx['products_df']
    reviews_by_product = ctx['reviews_by_product']
    new_reviews = ctx['new_reviews']
    images_dir = ctx['images_dir']
    product_vectors = ctx['product_vectors']
    get_recommendations = ctx['get_recommendations']

    product_row = products_df[products_df['product_id'] == product_id]
    if product_row.empty:
        return "<p style='color: #e74c3c;'>Product not found.</p>"

    product_row = product_row.iloc[0]
    brand = product_row['brand_name']
    title = product_row['product_title']
    price = product_row['price']
    rating = product_row['avg_product_rating']
    rating_count = product_row.get('product_rating_count', 0)
    product_url = product_row.get('product_url', '')

    # Get product tags (from pre-indexed reviews)
    product_tags = ''
    if product_id in reviews_by_product:
        first_review = reviews_by_product[product_id][0] if reviews_by_product[product_id] else {}
        tag_val = str(first_review.get('product_tags', '') or '').strip()
        if tag_val and tag_val != 'nan':
            product_tags = tag_val

    # Product image
    img_path = os.path.join(images_dir, f"{product_id}.png")
    if os.path.exists(img_path):
        img_html = f'<img src="/gradio_api/file={img_path}" style="width:100%;max-height:500px;object-fit:contain;background:#f5f5f5;" alt="{brand}"/>'
    else:
        img_html = f'<div style="width:100%;height:400px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;"><span style="color:#999;font-size:48px;font-weight:700;">{brand[0]}</span></div>'

    # SVG stars helper (black filled/outline)
    def stars_html(rating_val):
        full = int(rating_val)
        empty = 5 - full
        filled = '<svg width="18" height="18" viewBox="0 0 24 24" fill="#1a1a1a" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>'
        outline = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1a1a1a" stroke-width="1.5" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>'
        return filled * full + outline * empty

    # Tags HTML
    tags_html = ''
    if product_tags:
        tags = [t.strip() for t in product_tags.split(',') if t.strip()]
        if tags:
            tags_html = '<div class="pdp-tags">' + ''.join(f'<span class="pdp-tag">{t}</span>' for t in tags) + '</div>'

    # URL link
    url_html = ''
    if product_url and str(product_url).startswith('http'):
        url_html = f'<a href="{product_url}" target="_blank" class="pdp-link">View on Nykaa →</a>'

    # Get all reviews from pre-indexed dict + new reviews (fast, no DataFrame filter)
    all_reviews = reviews_by_product.get(product_id, []) + [r for r in new_reviews if r.get('product_id') == product_id]

    # Build reviews list HTML (show all, sorted by rating desc)
    all_reviews_sorted = sorted(all_reviews, key=lambda r: float(r.get('review_rating', 0) or r.get('rating', 0) or 0), reverse=True)
    display_reviews = all_reviews_sorted

    # Compute rating breakdown for filter bar
    rating_counts = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for r in all_reviews:
        rv = int(float(r.get('review_rating', 0) or r.get('rating', 0) or 0))
        if rv in rating_counts:
            rating_counts[rv] += 1

    reviews_items_html = ''
    for review in display_reviews:
        r_title = html_module.escape(str(review.get('review_title', '') or ''))
        r_text = html_module.escape(str(review.get('review_text', '') or ''))
        r_author = html_module.escape(str(review.get('author', 'Anonymous') or 'Anonymous'))
        r_date = str(review.get('review_date', '') or '')
        r_rating = float(review.get('review_rating', 0) or review.get('rating', 0) or 0)
        is_buyer = review.get('is_a_buyer', review.get('is_buyer', None))
        skin_type = str(review.get('skin_type', '') or '')
        skin_tone = str(review.get('skin_tone', '') or '')

        buyer_badge = '<span class="shopee-buyer-badge">Verified Buyer</span>' if is_buyer in [1, True, 'True'] else ''
        title_sort = r_title.lower().replace('"', '').replace("'", '')

        # Meta info line (skin type, skin tone)
        meta_parts = []
        if skin_type and skin_type not in ['', 'nan', 'N/A']:
            meta_parts.append(f'Skin type: <b>{html_module.escape(skin_type)}</b>')
        if skin_tone and skin_tone not in ['', 'nan', 'N/A']:
            meta_parts.append(f'Skin tone: <b>{html_module.escape(skin_tone)}</b>')
        meta_html = f'<div class="shopee-review-meta">{" | ".join(meta_parts)}</div>' if meta_parts else ''

        # Star color: Shopee uses red stars
        def shopee_stars(rating_val):
            full = int(rating_val)
            empty = 5 - full
            filled = '<svg width="14" height="14" viewBox="0 0 24 24" fill="#EE4D2D" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>'
            outline = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#EE4D2D" stroke-width="1.5" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>'
            return filled * full + outline * empty

        reviews_items_html += f'''
        <div class="shopee-review-item" data-rating="{r_rating}" data-title="{title_sort}">
            <div class="shopee-review-header">
                <div class="shopee-review-avatar">{r_author[0].upper() if r_author else 'A'}</div>
                <div class="shopee-review-user-info">
                    <span class="shopee-review-username">{r_author}{buyer_badge}</span>
                    <span class="shopee-review-stars">{shopee_stars(r_rating)}</span>
                    <span class="shopee-review-date">{r_date}</span>
                </div>
            </div>
            {meta_html}
            <div class="shopee-review-title">{r_title}</div>
            <div class="shopee-review-text">{r_text}</div>
        </div>
        '''

    sort_js = ""

    # Similar products — carousel
    recs = get_recommendations(product_id, product_vectors, products_df, n=8)
    similar_html = ''
    if recs:
        similar_cards = ''
        for rec in recs:
            rec_pid = rec['product_id']
            rec_img_path = os.path.join(images_dir, f"{rec_pid}.png")
            if os.path.exists(rec_img_path):
                rec_img = f'<img src="/gradio_api/file={rec_img_path}" class="carousel-card-img" alt="{rec["brand_name"]}"/>'
            else:
                rec_img = f'<div class="carousel-card-img" style="display:flex;align-items:center;justify-content:center;background:#f5f5f5;"><span style="color:#999;font-size:28px;font-weight:700;">{rec["brand_name"][0]}</span></div>'

            rec_onclick = f'''onclick="setGradioValue('#{target_elem_id}', '{rec_pid}_' + Date.now())"'''

            similar_cards += f'''
            <div class="carousel-card" {rec_onclick}>
                <div class="carousel-card-img-wrap">{rec_img}</div>
                <div class="carousel-card-info">
                    <h4 class="carousel-card-title">{rec['product_title'][:50]}...</h4>
                    <div class="carousel-card-price">₹{rec['price']:.0f}</div>
                    <div class="carousel-card-rating">★ {rec['avg_product_rating']:.1f}</div>
                </div>
            </div>
            '''

        similar_html = f'''
        <div class="pdp-similar-section">
            <div class="carousel-header">
                <h3 class="carousel-title">SIMILAR PRODUCTS</h3>
            </div>
            <div class="carousel-wrapper">
                <button class="carousel-btn carousel-btn-left" onclick="document.getElementById('similar-carousel').scrollBy({{left: -220, behavior: 'smooth'}})">‹</button>
                <div class="carousel-track" id="similar-carousel">
                    {similar_cards}
                </div>
                <button class="carousel-btn carousel-btn-right" onclick="document.getElementById('similar-carousel').scrollBy({{left: 220, behavior: 'smooth'}})">›</button>
            </div>
        </div>
        '''

    # Assemble full PDP
    detail_html = f'''
    <div class="detail-panel">
        <div class="pdp-top">
            <div class="pdp-image">{img_html}</div>
            <div class="pdp-info">
                <div class="pdp-brand">{brand}</div>
                <h2 class="pdp-title">{title}</h2>
                <div class="pdp-price">₹{price:.2f}</div>
                <div class="pdp-rating">{stars_html(rating)} {rating:.1f} ({rating_count} ratings)</div>
                <div class="pdp-meta">{len(all_reviews)} customer reviews</div>
                {url_html}
                {tags_html}
            </div>
        </div>

        <div class="shopee-reviews-section">
            <h3 class="shopee-reviews-title">PRODUCT REVIEWS</h3>
            <div class="shopee-reviews-overview">
                <div class="shopee-rating-big">
                    <span class="shopee-rating-number">{rating:.1f}</span>
                    <span class="shopee-rating-outof">out of 5</span>
                    <span class="shopee-rating-stars-big">{stars_html(rating)}</span>
                </div>
                <div class="shopee-filter-tabs">
                    <button class="shopee-filter-btn active" onclick="filterReviews('all', this)">All ({len(all_reviews)})</button>
                    <button class="shopee-filter-btn" onclick="filterReviews('5', this)">5 Star ({rating_counts[5]})</button>
                    <button class="shopee-filter-btn" onclick="filterReviews('4', this)">4 Star ({rating_counts[4]})</button>
                    <button class="shopee-filter-btn" onclick="filterReviews('3', this)">3 Star ({rating_counts[3]})</button>
                    <button class="shopee-filter-btn" onclick="filterReviews('2', this)">2 Star ({rating_counts[2]})</button>
                    <button class="shopee-filter-btn" onclick="filterReviews('1', this)">1 Star ({rating_counts[1]})</button>
                </div>
            </div>
            <div class="shopee-sort-bar">
                <span style="font-size:13px;color:#757575;">Sort by:</span>
                <button class="shopee-sort-btn active" onclick="sortReviews('rating-desc', this)">Rating ↓</button>
                <button class="shopee-sort-btn" onclick="sortReviews('rating-asc', this)">Rating ↑</button>
                <button class="shopee-sort-btn" onclick="sortReviews('alpha-az', this)">A→Z</button>
            </div>
            <div class="shopee-reviews-container" id="pdp-reviews-container">
                {reviews_items_html}
            </div>
        </div>

        {similar_html}
        {sort_js}
    </div>
    '''

    return detail_html

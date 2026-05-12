"""Tab 2: Write a Review — form, buyer label prediction, and submission."""

import gradio as gr
from ui.components import render_model_predictions
from core.classifier import predict_buyer
from core.data_loader import add_review, new_reviews


def build_review_tab(product_choices, products_df, models, stop_words, ctx):
    """Build the Write a Review tab UI and wire event handlers."""

    model_a = models['model_a_bow']
    model_b = models['model_b_bow_meta']
    model_c = models['model_c_glove_meta']
    label_encoder = models['label_encoder']
    colloc_dict = models['collocation_dict']

    with gr.Tab("✍️ Write a Review"):
        gr.Markdown("### Share your experience with a product")

        with gr.Group(elem_id="review-form-group"):
            review_product = gr.Dropdown(
                choices=product_choices,
                label="Select Product",
                filterable=True
            )

            review_title = gr.Textbox(label="Review Title", placeholder="Short summary of your experience")
            review_text = gr.Textbox(label="Review Text", lines=4, placeholder="Tell us more about your experience...")

            gr.HTML("""
            <div class="star-rating-widget">
                <label style="font-size:14px;font-weight:500;color:#222;display:block;margin-bottom:8px;">Rating</label>
                <div class="star-rating-stars" id="star-rating-stars">
                    <span class="star-icon active" data-value="1">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                    </span>
                    <span class="star-icon active" data-value="2">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                    </span>
                    <span class="star-icon active" data-value="3">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                    </span>
                    <span class="star-icon active" data-value="4">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                    </span>
                    <span class="star-icon" data-value="5">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                    </span>
                </div>
                <span class="star-rating-text" id="star-rating-text">4 out of 5</span>
            </div>
            """)

        review_rating = gr.Textbox(
            value="4",
            elem_id="star_rating_hidden",
            visible=True,
            elem_classes=["hidden-input"]
        )

        predict_btn = gr.Button("Predict Buyer Label", variant="primary")

        prediction_results = gr.HTML(label="Model Predictions")

        gr.HTML("""
        <div class="final-label-widget">
            <label style="font-size:14px;font-weight:500;color:#222;display:block;margin-bottom:10px;">Final Label (override if needed)</label>
            <div class="label-btn-group">
                <button class="label-btn active" onclick="selectLabel('Buyer', this)">Buyer</button>
                <button class="label-btn" onclick="selectLabel('Non-Buyer', this)">Non-Buyer</button>
            </div>
        </div>
        """)

        final_label = gr.Textbox(
            value="Buyer",
            elem_id="final_label_hidden",
            visible=True,
            elem_classes=["hidden-input"]
        )

        submit_btn = gr.Button("Confirm & Submit Review", variant="secondary")
        confirmation_msg = gr.Markdown("")

        def on_predict(product_id, title, text, rating_str):
            if not product_id or not text:
                return "<p style='color: #e74c3c;'>Please select a product and enter review text.</p>"

            try:
                rating = int(rating_str) if rating_str else 4
            except (ValueError, TypeError):
                rating = 4

            product = products_df[products_df['product_id'] == product_id]
            if product.empty:
                return "<p style='color: #e74c3c;'>Product not found.</p>"

            product = product.iloc[0]
            product_info = {
                'price': product['price'],
                'avg_product_rating': product['avg_product_rating'],
                'product_rating_count': product.get('product_rating_count', 0),
                'brand_name': product['brand_name']
            }

            pred_result = predict_buyer(
                text, title, rating, product_info,
                model_a, model_b, model_c, label_encoder,
                colloc_dict, stop_words
            )

            pred_html = render_model_predictions(pred_result)
            fused_label = pred_result['label']

            return pred_html, fused_label

        predict_btn.click(
            fn=on_predict,
            inputs=[review_product, review_title, review_text, review_rating],
            outputs=[prediction_results, final_label],
            show_progress="hidden"
        )

        def on_submit(product_id, title, text, rating_str, label):
            if not product_id or not text:
                return "⚠️ Please select a product and enter review text."

            try:
                rating = int(rating_str) if rating_str else 4
            except (ValueError, TypeError):
                rating = 4

            product = products_df[products_df['product_id'] == product_id]
            if product.empty:
                return "⚠️ Product not found."

            product = product.iloc[0]

            review_dict = {
                'product_id': product_id,
                'brand_name': product['brand_name'],
                'product_title': product['product_title'],
                'review_title': title,
                'review_text': text,
                'review_rating': rating,
                'rating': rating,
                'is_buyer': 1 if label == "Buyer" else 0,
                'price': product['price'],
                'avg_product_rating': product['avg_product_rating'],
                'product_rating_count': product.get('product_rating_count', 0),
                'skin_type': 'N/A',
                'skin_tone': 'N/A'
            }

            add_review(review_dict)

            return f"✅ **Review submitted successfully!** Your review for **{product['product_title']}** has been added. Total reviews on this platform: {len(new_reviews)}"

        submit_btn.click(
            fn=on_submit,
            inputs=[review_product, review_title, review_text, review_rating, final_label],
            outputs=[confirmation_msg],
            show_progress="hidden"
        )

"""Tab 2: Write a Review — form, buyer label prediction, and in-tab confirmation panel."""

import html
import gradio as gr
from core.classifier import predict_buyer
from core.data_loader import add_review


def build_review_tab(product_choices, products_df, models, stop_words, ctx):
    """Build the Write a Review tab UI and wire event handlers."""

    model_a = models['rf_bow_extra']
    model_b = models['rf_unweighted_extra']
    model_c = models['rf_weighted_extra']
    label_encoder = models['label_encoder']
    colloc_dict = models['collocation_dict']

    with gr.Tab("Write a Review", elem_id="tab-review", elem_classes=["glamreview-tab"]):
        gr.Markdown("### Share your experience with a product")

        pending_state = gr.State(value=None)

        # ── Form panel ────────────────────────────────────────────────────────
        with gr.Column(visible=True, elem_id="review-form-col") as form_col:
            with gr.Group(elem_id="review-form-group"):
                review_product = gr.Dropdown(
                    choices=product_choices,
                    label="Select Product",
                    filterable=True,
                    elem_id="review-product-dropdown",
                    elem_classes=["dropdown-container"]
                )
                review_product_trigger = gr.Textbox(
                    elem_id="review_product_trigger",
                    visible=True,
                    elem_classes=["hidden-input"]
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

            submit_btn = gr.Button("Submit Review", variant="primary")
            submit_error = gr.HTML("")

        # ── Confirmation panel ────────────────────────────────────────────────
        with gr.Column(visible=False, elem_id="review-confirm-col") as confirm_col:
            gr.Markdown("### Confirm Your Review")
            confirm_summary = gr.HTML("", elem_id="confirm-summary-html")
            confirm_buyer_html = gr.HTML("", elem_id="confirm-buyer-pills")
            confirm_buyer_val = gr.Textbox(
                value="Yes, I would buy it",
                elem_id="confirm_buyer_val",
                visible=False,
                elem_classes=["hidden-input"]
            )
            with gr.Row():
                back_btn = gr.Button("Back", variant="secondary", elem_classes=["confirm-back-btn"])
                confirm_btn = gr.Button("Confirm and Save", variant="primary", elem_classes=["confirm-save-btn"])
            confirm_success = gr.HTML("")
            redirect_trigger = gr.Textbox(
                value="",
                elem_id="confirm_redirect_trigger",
                visible=False,
                elem_classes=["hidden-input"]
            )

        # ── Handlers ─────────────────────────────────────────────────────────

        def _buyer_pills_html(selected):
            yes_active = "active" if selected == "Yes, I would buy it" else ""
            no_active = "active" if selected == "No, I would not buy it" else ""
            return f"""
    <div style="margin-bottom:8px;">
        <p style="font-size:14px;font-weight:500;color:#222;margin:0 0 12px;">Based on your experience, would you buy this product?</p>
        <div class="label-btn-group">
            <button class="label-btn {yes_active}"
                onclick="(function(){{
                    this.parentElement.querySelectorAll('.label-btn').forEach(b=>b.classList.remove('active'));
                    this.classList.add('active');
                    setGradioValue('#confirm_buyer_val', 'Yes, I would buy it');
                }}).call(this)">Yes, I would buy it</button>
            <button class="label-btn {no_active}"
                onclick="(function(){{
                    this.parentElement.querySelectorAll('.label-btn').forEach(b=>b.classList.remove('active'));
                    this.classList.add('active');
                    setGradioValue('#confirm_buyer_val', 'No, I would not buy it');
                }}).call(this)">No, I would not buy it</button>
        </div>
    </div>
    """

        def _error(msg):
            """Return a 6-tuple that leaves everything unchanged except showing msg."""
            return gr.update(), gr.update(), gr.update(), msg, gr.update(), gr.update()

        def on_submit(product_id, title, text, rating_str):
            if not product_id or not text:
                return _error("<p style='color:#e74c3c;'>Please select a product and enter review text.</p>")

            try:
                rating = int(rating_str) if rating_str else 4
            except (ValueError, TypeError):
                rating = 4

            product_row = products_df[products_df['product_id'] == product_id]
            if product_row.empty:
                return _error("<p style='color:#e74c3c;'>Product not found.</p>")

            product = product_row.iloc[0]
            product_info = {
                'price': product['price'],
                'avg_product_rating': product['avg_product_rating'],
                'product_rating_count': product.get('product_rating_count', 0),
                'brand_name': product['brand_name'],
            }

            pred_result = predict_buyer(
                text, title, rating, product_info,
                model_a, model_b, model_c, label_encoder,
                colloc_dict, stop_words
            )

            pred_label = pred_result['label']  # "Buyer" or "Non-Buyer"

            review_dict = {
                'product_id': product_id,
                'brand_name': product['brand_name'],
                'product_title': product['product_title'],
                'review_title': title,
                'review_text': text,
                'review_rating': rating,
                'rating': rating,
                'price': product['price'],
                'avg_product_rating': product['avg_product_rating'],
                'product_rating_count': product.get('product_rating_count', 0),
                'skin_type': 'N/A',
                'skin_tone': 'N/A',
                'predicted_label': pred_label,
            }

            # Build summary HTML
            stars_filled = "★" * rating + "☆" * (5 - rating)
            display_note = "You would buy it" if pred_label == "Buyer" else "You would not buy it"
            radio_preselect = "Yes, I would buy it" if pred_label == "Buyer" else "No, I would not buy it"

            summary_html = f"""
            <div class="model-card" style="margin-bottom:16px;">
                <p style="margin:0 0 4px;font-size:12px;color:#757575;font-weight:500;text-transform:uppercase;letter-spacing:.5px;">Item</p>
                <p style="margin:0 0 14px;font-size:15px;font-weight:600;color:#222;">{html.escape(product['product_title'])} &mdash; {html.escape(product['brand_name'])}</p>
                <p style="margin:0 0 4px;font-size:12px;color:#757575;font-weight:500;text-transform:uppercase;letter-spacing:.5px;">Title</p>
                <p style="margin:0 0 14px;font-size:14px;color:#333;">{html.escape(title or '(no title)')}</p>
                <p style="margin:0 0 4px;font-size:12px;color:#757575;font-weight:500;text-transform:uppercase;letter-spacing:.5px;">Rating</p>
                <p style="margin:0 0 14px;font-size:18px;color:#EE4D2D;">{stars_filled} <span style="font-size:13px;color:#555;">({rating}/5)</span></p>
                <p style="margin:0 0 4px;font-size:12px;color:#757575;font-weight:500;text-transform:uppercase;letter-spacing:.5px;">Description</p>
                <p style="margin:0;font-size:14px;color:#333;line-height:1.7;">{html.escape(text)}</p>
            </div>
            <div style="background:#e8f4fd;border:1px solid #b3d9f5;border-radius:8px;
                        padding:14px 18px;margin-bottom:8px;font-size:14px;color:#1a5276;">
                <div style="margin-bottom:4px;">Our prediction: <strong>'{display_note}'</strong></div>
                <div style="font-size:12px;opacity:0.8;">
                    Ensemble Confidence: 
                    A: {pred_result['model_a_prob']:.0%} | 
                    B: {pred_result['model_b_prob']:.0%} | 
                    C: {pred_result['model_c_prob']:.0%}
                </div>
            </div>
            """

            return (
                {'review_dict': review_dict, 'pred_label': pred_label},
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                summary_html,
                _buyer_pills_html(radio_preselect),
            )

        submit_btn.click(
            fn=on_submit,
            inputs=[review_product, review_title, review_text, review_rating],
            outputs=[pending_state, form_col, confirm_col, submit_error, confirm_summary, confirm_buyer_html],
            show_progress="hidden",
        )

        def on_back(_state):
            return gr.update(visible=True), gr.update(visible=False), ""

        back_btn.click(
            fn=on_back,
            inputs=[pending_state],
            outputs=[form_col, confirm_col, confirm_success],
            show_progress="hidden",
        )

        def on_confirm_save(state, buyer_val):
            if not state or 'review_dict' not in state:
                return (
                    gr.update(),
                    gr.update(),
                    "<p style='color:#e74c3c;'>Session expired. Please go back and resubmit.</p>",
                )
            review_dict = dict(state['review_dict'])
            review_dict['is_buyer'] = 1 if buyer_val == "Yes, I would buy it" else 0
            add_review(review_dict)
            product_id = review_dict.get('product_id', '')
            success_html = """
            <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
                        padding:16px 20px;margin-top:12px;font-size:14px;color:#166534;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:20px;">&#10003;</span>
                    <strong>Review submitted! Redirecting to product page...</strong>
                </div>
            </div>
            """
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                success_html,
                str(product_id),
            )

        confirm_btn.click(
            fn=on_confirm_save,
            inputs=[pending_state, confirm_buyer_val],
            outputs=[confirm_btn, back_btn, confirm_success, redirect_trigger],
            show_progress="hidden",
        )

        redirect_trigger.change(
            fn=None,
            inputs=[redirect_trigger],
            outputs=[],
            js="""(pid) => {
                if (!pid) return;
                setTimeout(function() {
                    setGradioValue('#selected_product_id', pid + '_' + Date.now());
                    var tabs = document.querySelectorAll('.glamreview-tabs [role="tablist"] button');
                    if (tabs.length > 0) tabs[0].click();
                }, 800);
            }""",
        )

        def on_review_trigger(val):
            if not val:
                return gr.update()
            try:
                pid = int(val.split("_")[0])
                return gr.update(value=pid)
            except (ValueError, IndexError):
                return gr.update()

        review_product_trigger.change(
            fn=on_review_trigger,
            inputs=[review_product_trigger],
            outputs=[review_product],
            show_progress="hidden",
        )

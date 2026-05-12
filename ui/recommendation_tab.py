"""Tab 3: Similar Products — find and browse product recommendations."""

import gradio as gr
from ui.product_detail import render_product_detail
from ui.recommendation_grid import render_recommendation_grid
from core.recommender import get_recommendations


def build_recommendation_tab(product_choices, products_df, product_vectors, ctx):
    """Build the Similar Products tab UI and wire event handlers."""

    images_dir = ctx['images_dir']

    with gr.Tab("🔗 Similar Products"):
        gr.Markdown("### Find products similar to one you like")

        rec_product = gr.Dropdown(
            choices=product_choices,
            label="Select a Product",
            filterable=True
        )

        find_similar_btn = gr.Button("Find Similar Products", variant="primary")

        rec_results = gr.HTML()

        selected_rec_product_id = gr.Textbox(
            elem_id="selected_rec_product_id",
            visible=True,
            elem_classes=["hidden-input"]
        )

        rec_detail_panel = gr.HTML(value="", visible=True)
        back_to_rec_btn = gr.Button("← Back to Recommendations", visible=False)

        def on_find_similar(product_id):
            if not product_id:
                return [
                    "<p style='color: #666;'>Please select a product.</p>",
                    "",
                    gr.update(visible=False)
                ]

            selected = products_df[products_df['product_id'] == product_id]
            if selected.empty:
                return [
                    "<p style='color: #e74c3c;'>Product not found.</p>",
                    "",
                    gr.update(visible=False)
                ]

            selected = selected.iloc[0]
            selected_dict = {
                'product_id': product_id,
                'brand_name': selected['brand_name'],
                'product_title': selected['product_title'],
                'price': selected['price'],
                'avg_product_rating': selected['avg_product_rating'],
                'review_count': selected.get('review_count', 0)
            }

            recommendations = get_recommendations(product_id, product_vectors, products_df, n=5)

            html = render_recommendation_grid(selected_dict, recommendations, images_dir)
            return [html, "", gr.update(visible=False)]

        find_similar_btn.click(
            fn=on_find_similar,
            inputs=[rec_product],
            outputs=[rec_results, rec_detail_panel, back_to_rec_btn],
            show_progress="hidden"
        )

        def on_rec_product_select(product_id_str):
            if not product_id_str or product_id_str.strip() == "":
                return ["", gr.update(visible=False), gr.update()]

            try:
                raw_id = product_id_str.split("_")[0]
                product_id = int(raw_id)
                detail_html = render_product_detail(product_id, ctx, target_elem_id="selected_rec_product_id")
                return [detail_html, gr.update(visible=True), ""]
            except Exception as e:
                print(f"Error loading rec product detail: {e}")
                return [gr.update(), gr.update(), gr.update()]

        selected_rec_product_id.change(
            fn=on_rec_product_select,
            inputs=[selected_rec_product_id],
            outputs=[rec_detail_panel, back_to_rec_btn, rec_results],
            show_progress="hidden"
        )

        def on_back_rec(product_id):
            if not product_id:
                return ["", gr.update(visible=False), "", ""]

            selected = products_df[products_df['product_id'] == product_id]
            if selected.empty:
                return ["", gr.update(visible=False), "", ""]

            selected = selected.iloc[0]
            selected_dict = {
                'product_id': product_id,
                'brand_name': selected['brand_name'],
                'product_title': selected['product_title'],
                'price': selected['price'],
                'avg_product_rating': selected['avg_product_rating'],
                'review_count': selected.get('review_count', 0)
            }
            recommendations = get_recommendations(product_id, product_vectors, products_df, n=5)
            html = render_recommendation_grid(selected_dict, recommendations, images_dir)

            return ["", gr.update(visible=False), "", html]

        back_to_rec_btn.click(
            fn=on_back_rec,
            inputs=[rec_product],
            outputs=[rec_detail_panel, back_to_rec_btn, selected_rec_product_id, rec_results],
            show_progress="hidden"
        )

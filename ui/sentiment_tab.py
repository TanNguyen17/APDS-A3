"""Tab 4: Sentiment Insights — analyze sentiment and compare products."""

import pandas as pd
import gradio as gr
from ui.components import render_key_terms_table
from core.sentiment import (
    get_sentiment_breakdown,
    get_key_terms,
    create_sentiment_chart,
    create_comparison_chart,
)
from core.data_loader import new_reviews


def build_sentiment_tab(product_choices, df, products_df):
    """Build the Sentiment Insights tab UI and wire event handlers."""

    with gr.Tab("📊 Sentiment Insights"):
        gr.Markdown("### Analyze sentiment and key terms from product reviews")

        with gr.Row():
            sentiment_product_1 = gr.Dropdown(
                choices=product_choices,
                label="Product 1",
                filterable=True
            )

            sentiment_product_2 = gr.Dropdown(
                choices=product_choices,
                label="Product 2 (optional for comparison)",
                filterable=True
            )

        analyze_btn = gr.Button("Analyze", variant="primary")

        sentiment_chart = gr.Plot(label="Sentiment Distribution")
        comparison_chart = gr.Plot(label="Comparison", visible=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 😊 Key Positive Terms")
                positive_terms = gr.HTML()

            with gr.Column():
                gr.Markdown("### 😞 Key Negative Terms")
                negative_terms = gr.HTML()

        def on_analyze(product_id_1, product_id_2):
            if not product_id_1:
                return [None, gr.update(visible=False), "", ""]

            new_reviews_df = pd.DataFrame(new_reviews) if new_reviews else None

            breakdown_1 = get_sentiment_breakdown(product_id_1, df, new_reviews_df)
            chart_1 = create_sentiment_chart(breakdown_1)

            pos_terms = get_key_terms(product_id_1, df, 'positive', top_n=15, new_reviews=new_reviews_df)
            neg_terms = get_key_terms(product_id_1, df, 'negative', top_n=15, new_reviews=new_reviews_df)

            pos_html = render_key_terms_table(pos_terms, "positive")
            neg_html = render_key_terms_table(neg_terms, "negative")

            if product_id_2:
                p1 = products_df[products_df['product_id'] == product_id_1].iloc[0]
                p2 = products_df[products_df['product_id'] == product_id_2].iloc[0]

                name_1 = f"{p1['brand_name']} - {p1['product_title'][:30]}"
                name_2 = f"{p2['brand_name']} - {p2['product_title'][:30]}"

                breakdown_2 = get_sentiment_breakdown(product_id_2, df, new_reviews_df)
                chart_2 = create_comparison_chart(breakdown_1, breakdown_2, name_1, name_2)

                return [chart_1, gr.update(value=chart_2, visible=True), pos_html, neg_html]
            else:
                return [chart_1, gr.update(visible=False), pos_html, neg_html]

        analyze_btn.click(
            fn=on_analyze,
            inputs=[sentiment_product_1, sentiment_product_2],
            outputs=[sentiment_chart, comparison_chart, positive_terms, negative_terms],
            show_progress="hidden"
        )

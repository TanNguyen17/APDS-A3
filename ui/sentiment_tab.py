"""Tab 4: Sentiment Insights — analyze and compare product sentiment."""

import math
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ui.components import render_key_terms_table
from core.sentiment import (
    get_sentiment_breakdown,
    get_key_terms,
    create_multi_comparison_chart,
)
from core.data_loader import new_reviews


def build_sentiment_tab(product_choices, df, products_df):
    """Build the Sentiment Insights tab UI and wire event handlers."""

    with gr.Tab("📊 Sentiment Insights"):
        gr.Markdown("### Analyze sentiment and key terms from product reviews")
        gr.Markdown("Select one or more products to analyze and compare their sentiment.")

        product_selector = gr.Dropdown(
            choices=product_choices,
            label="Select Products",
            multiselect=True,
            filterable=True,
            value=[],
        )

        analyze_btn = gr.Button("Analyze", variant="primary")

        # Outputs
        comparison_chart = gr.Plot(label="Sentiment Comparison")
        sentiment_chart = gr.Plot(label="Individual Analysis")
        key_terms_html = gr.HTML()

        def on_analyze(selected_products):
            if not selected_products:
                return [None, None, ""]

            # Deduplicate (shouldn't happen with multiselect, but be safe)
            seen = set()
            active_ids = []
            for pid in selected_products:
                if pid and pid not in seen:
                    seen.add(pid)
                    active_ids.append(pid)

            if not active_ids:
                return [None, None, ""]

            new_reviews_df = pd.DataFrame(new_reviews) if new_reviews else None

            def get_name(pid):
                row = products_df[products_df['product_id'] == pid].iloc[0]
                return f"{row['brand_name']} - {row['product_title']}"

            breakdowns = []
            names = []
            for pid in active_ids:
                bd = get_sentiment_breakdown(pid, df, new_reviews_df)
                breakdowns.append(bd)
                names.append(get_name(pid))

            # Comparison chart (2+ products)
            comp_fig = create_multi_comparison_chart(breakdowns, names) if len(active_ids) >= 2 else None

            # Individual bar charts in 2-col grid
            n = len(active_ids)
            cols = min(n, 2)
            rows = math.ceil(n / cols)
            short_titles = [
                (name[:25] + "…") if len(name) > 25 else name
                for name in names
            ]

            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"{short_titles[i]} (n={breakdowns[i]['total']})" for i in range(n)],
                vertical_spacing=0.22,
                horizontal_spacing=0.12,
            )

            categories = ['Positive', 'Neutral', 'Negative']
            bar_colors = ['#2ecc71', '#95a5a6', '#EE4D2D']

            for i, bd in enumerate(breakdowns):
                r = i // cols + 1
                c = i % cols + 1
                pcts = [bd['positive_pct'], bd['neutral_pct'], bd['negative_pct']]
                fig.add_trace(
                    go.Bar(
                        x=categories, y=pcts,
                        marker_color=bar_colors,
                        text=[f'{p:.1f}%' for p in pcts],
                        textposition='inside',
                        textfont=dict(color='white', size=11),
                        showlegend=False,
                    ),
                    row=r, col=c,
                )
                fig.update_yaxes(range=[0, 100], gridcolor='#f0f0f0', row=r, col=c)

            fig.update_layout(
                height=300 * rows,
                margin=dict(t=50, b=30, l=40, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif'),
                showlegend=False,
            )

            # Key terms for all products
            terms_sections = []
            for i, pid in enumerate(active_ids):
                pos_terms = get_key_terms(pid, df, 'positive', top_n=15, new_reviews=new_reviews_df)
                neg_terms = get_key_terms(pid, df, 'negative', top_n=15, new_reviews=new_reviews_df)
                pos_html = render_key_terms_table(pos_terms, "positive")
                neg_html = render_key_terms_table(neg_terms, "negative")

                section = f"""
                <div style="margin-bottom: 24px; padding: 16px; border: 1px solid #e0e0e0; border-radius: 8px;">
                    <h4 style="margin: 0 0 12px 0; color: #333;">{names[i]}</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div>
                            <h5 style="margin: 0 0 8px 0;">😊 Positive Terms</h5>
                            {pos_html}
                        </div>
                        <div>
                            <h5 style="margin: 0 0 8px 0;">😞 Negative Terms</h5>
                            {neg_html}
                        </div>
                    </div>
                </div>
                """
                terms_sections.append(section)

            return [comp_fig, fig, "".join(terms_sections)]

        analyze_btn.click(
            fn=on_analyze,
            inputs=[product_selector],
            outputs=[comparison_chart, sentiment_chart, key_terms_html],
            show_progress="hidden",
        )

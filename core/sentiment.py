"""
Task 4: Sentiment Analysis Dashboard

Provides sentiment breakdown, key term extraction, and product comparison
functionality for the Gradio web application.
"""

import math
import plotly.graph_objects as go
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional


def get_sentiment_breakdown(
    product_id: str,
    df: pd.DataFrame,
    new_reviews: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Calculate sentiment distribution for a product's reviews.

    Args:
        product_id: Product identifier to filter reviews
        df: Full reviews DataFrame with 'product_id' and 'text_sentiment' columns
        new_reviews: Optional DataFrame of new reviews to include

    Returns:
        Dict containing counts and percentages for positive/neutral/negative sentiment
    """
    # Filter reviews for the product
    product_reviews = df[df['product_id'] == product_id].copy()

    # Add new reviews if provided
    if new_reviews is not None and not new_reviews.empty:
        new_product_reviews = new_reviews[new_reviews['product_id'] == product_id].copy()
        product_reviews = pd.concat([product_reviews, new_product_reviews], ignore_index=True)

    if 'text_sentiment' not in product_reviews.columns:
        product_reviews['text_sentiment'] = 'neutral'
    else:
        product_reviews['text_sentiment'] = product_reviews['text_sentiment'].fillna('neutral')

    total = len(product_reviews)

    # Handle edge case: no reviews
    if total == 0:
        return {
            'positive_count': 0,
            'neutral_count': 0,
            'negative_count': 0,
            'total': 0,
            'positive_pct': 0.0,
            'neutral_pct': 0.0,
            'negative_pct': 0.0
        }

    # Classify by rating
    positive_count = len(product_reviews[product_reviews['text_sentiment'] == 'positive'])
    neutral_count = len(product_reviews[product_reviews['text_sentiment'] == 'neutral'])
    negative_count = len(product_reviews[product_reviews['text_sentiment'] == 'negative'])

    return {
        'positive_count': positive_count,
        'neutral_count': neutral_count,
        'negative_count': negative_count,
        'total': total,
        'positive_pct': (positive_count / total) * 100,
        'neutral_pct': (neutral_count / total) * 100,
        'negative_pct': (negative_count / total) * 100
    }


def get_key_terms(
    product_id: str,
    df: pd.DataFrame,
    sentiment: str = 'positive',
    top_n: int = 15,
    new_reviews: Optional[pd.DataFrame] = None
) -> List[Tuple[str, int]]:
    """
    Extract most frequent words from reviews of a specific sentiment category.

    Args:
        product_id: Product identifier to filter reviews
        df: Full reviews DataFrame
        sentiment: 'positive' (rating >= 4), 'neutral' (rating == 3), or 'negative' (rating <= 2)
        top_n: Number of top terms to return
        new_reviews: Optional DataFrame of new reviews to include

    Returns:
        List of (word, count) tuples sorted by frequency descending
    """
    # Filter reviews for the product
    product_reviews = df[df['product_id'] == product_id].copy()

    # Add new reviews if provided
    if new_reviews is not None and not new_reviews.empty:
        new_product_reviews = new_reviews[new_reviews['product_id'] == product_id].copy()
        product_reviews = pd.concat([product_reviews, new_product_reviews], ignore_index=True)

    if 'text_sentiment' not in product_reviews.columns:
        product_reviews['text_sentiment'] = 'neutral'
    else:
        product_reviews['text_sentiment'] = product_reviews['text_sentiment'].fillna('neutral')

    # Filter by sentiment
    if sentiment in ('positive','neutral', 'negative'):
        sentiment_reviews = product_reviews[product_reviews['text_sentiment'] == sentiment]
    else:
        raise ValueError(f"Invalid sentiment: {sentiment}. Must be 'positive', 'neutral', or 'negative'")

    # Handle edge case: no reviews in this sentiment category
    if len(sentiment_reviews) == 0:
        return []

    # Count word frequencies across all reviews
    word_counter = Counter()

    for text in sentiment_reviews['processed_review_text']:
        # Skip NaN or empty values
        if pd.isna(text) or text == '':
            continue

        # Split space-separated tokens and count
        tokens = text.split()
        word_counter.update(tokens)

    # Return top N terms
    return word_counter.most_common(top_n)


COMPARISON_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']


def create_multi_comparison_chart(
    breakdowns: List[Dict],
    names: List[str]
) -> go.Figure:
    """
    Create a grouped bar chart comparing sentiment distributions of 2-5 products.
    """
    categories = ['Positive', 'Neutral', 'Negative']

    fig = go.Figure()

    for i, (breakdown, name) in enumerate(zip(breakdowns, names)):
        pcts = [breakdown['positive_pct'], breakdown['neutral_pct'], breakdown['negative_pct']]
        fig.add_trace(go.Bar(
            x=categories,
            y=pcts,
            name=f'{name} (n={breakdown["total"]} reviews)',
            marker_color=COMPARISON_COLORS[i % len(COMPARISON_COLORS)],
            text=[f'{p:.1f}%' for p in pcts],
            textposition='outside',
            textfont=dict(color='black', size=11)
        ))

    n = len(breakdowns)
    legend_rows = math.ceil(n / 2)
    extra_legend_height = max(0, legend_rows - 1) * 28
    chart_height = 420 + extra_legend_height
    bottom_margin = 80 + extra_legend_height

    fig.update_layout(
        title=dict(
            text='Sentiment Comparison',
            font=dict(size=16, family='-apple-system, BlinkMacSystemFont, sans-serif'),
            x=0.5
        ),
        barmode='group',
        yaxis=dict(title='Percentage of Reviews', range=[0, 100], gridcolor='#f0f0f0'),
        xaxis=dict(title=''),
        legend=dict(orientation='h', yanchor='top', y=-0.12, xanchor='center', x=0.5),
        margin=dict(t=60, b=bottom_margin, l=50, r=20),
        height=chart_height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif'),
        bargap=0.2,
        bargroupgap=0.1
    )

    return fig



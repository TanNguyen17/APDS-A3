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
        df: Full reviews DataFrame with 'product_id' and 'review_rating' columns
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
    positive_count = len(product_reviews[product_reviews['review_rating'] >= 4])
    neutral_count = len(product_reviews[product_reviews['review_rating'] == 3])
    negative_count = len(product_reviews[product_reviews['review_rating'] <= 2])

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

    # Filter by sentiment
    if sentiment == 'positive':
        sentiment_reviews = product_reviews[product_reviews['review_rating'] >= 4]
    elif sentiment == 'neutral':
        sentiment_reviews = product_reviews[product_reviews['review_rating'] == 3]
    elif sentiment == 'negative':
        sentiment_reviews = product_reviews[product_reviews['review_rating'] <= 2]
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


def compare_products(
    product_id_1: str,
    product_id_2: str,
    df: pd.DataFrame,
    new_reviews: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Compare sentiment and distinctive terms between two products.

    Args:
        product_id_1: First product identifier
        product_id_2: Second product identifier
        df: Full reviews DataFrame
        new_reviews: Optional DataFrame of new reviews to include

    Returns:
        Dict containing sentiment breakdowns and unique terms for both products
    """
    # Get sentiment breakdowns
    breakdown_1 = get_sentiment_breakdown(product_id_1, df, new_reviews)
    breakdown_2 = get_sentiment_breakdown(product_id_2, df, new_reviews)

    # Get all reviews for both products
    reviews_1 = df[df['product_id'] == product_id_1].copy()
    reviews_2 = df[df['product_id'] == product_id_2].copy()

    # Add new reviews if provided
    if new_reviews is not None and not new_reviews.empty:
        new_reviews_1 = new_reviews[new_reviews['product_id'] == product_id_1].copy()
        new_reviews_2 = new_reviews[new_reviews['product_id'] == product_id_2].copy()
        reviews_1 = pd.concat([reviews_1, new_reviews_1], ignore_index=True)
        reviews_2 = pd.concat([reviews_2, new_reviews_2], ignore_index=True)

    # Count word frequencies for each product
    counter_1 = Counter()
    counter_2 = Counter()

    for text in reviews_1['processed_review_text']:
        if pd.isna(text) or text == '':
            continue
        counter_1.update(text.split())

    for text in reviews_2['processed_review_text']:
        if pd.isna(text) or text == '':
            continue
        counter_2.update(text.split())

    # Calculate differential terms (words much more common in one product vs another)
    # Use normalized frequency difference to account for different review counts
    total_1 = sum(counter_1.values()) or 1  # Avoid division by zero
    total_2 = sum(counter_2.values()) or 1

    # Words in product 1
    product_1_diff = {}
    for word, count in counter_1.items():
        freq_1 = count / total_1
        freq_2 = counter_2.get(word, 0) / total_2
        diff = freq_1 - freq_2
        if diff > 0:  # More common in product 1
            product_1_diff[word] = diff

    # Words in product 2
    product_2_diff = {}
    for word, count in counter_2.items():
        freq_2 = count / total_2
        freq_1 = counter_1.get(word, 0) / total_1
        diff = freq_2 - freq_1
        if diff > 0:  # More common in product 2
            product_2_diff[word] = diff

    # Get top 15 distinctive terms for each product
    product_1_unique = sorted(product_1_diff.items(), key=lambda x: x[1], reverse=True)[:15]
    product_2_unique = sorted(product_2_diff.items(), key=lambda x: x[1], reverse=True)[:15]

    return {
        'product_1_sentiment': breakdown_1,
        'product_2_sentiment': breakdown_2,
        'product_1_unique_terms': [(word, f"{score:.4f}") for word, score in product_1_unique],
        'product_2_unique_terms': [(word, f"{score:.4f}") for word, score in product_2_unique]
    }


COMPARISON_COLORS = ['#EE4D2D', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']


def create_sentiment_chart(breakdown: Dict) -> go.Figure:
    """
    Create a donut chart showing sentiment distribution.

    Args:
        breakdown: Sentiment breakdown dict from get_sentiment_breakdown()

    Returns:
        Plotly Figure object for Gradio gr.Plot
    """
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        breakdown['positive_count'],
        breakdown['neutral_count'],
        breakdown['negative_count']
    ]
    colors = ['#2ecc71', '#95a5a6', '#EE4D2D']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=13),
        hovertemplate='%{label}: %{value} reviews (%{percent})<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text=f'Sentiment Distribution ({breakdown["total"]} reviews)',
            font=dict(size=16, family='-apple-system, BlinkMacSystemFont, sans-serif'),
            x=0.5
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        margin=dict(t=60, b=40, l=20, r=20),
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif')
    )

    return fig


def create_single_bar_chart(breakdown: Dict, name: str) -> go.Figure:
    """
    Create a bar chart showing sentiment distribution for a single product.
    """
    categories = ['Positive', 'Neutral', 'Negative']
    pcts = [breakdown['positive_pct'], breakdown['neutral_pct'], breakdown['negative_pct']]
    colors = ['#2ecc71', '#95a5a6', '#EE4D2D']

    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=pcts,
        marker_color=colors,
        text=[f'{p:.1f}%' for p in pcts],
        textposition='inside',
        textfont=dict(color='white', size=13)
    )])

    fig.update_layout(
        title=dict(
            text=f'{name} ({breakdown["total"]} reviews)',
            font=dict(size=14, family='-apple-system, BlinkMacSystemFont, sans-serif'),
            x=0.5
        ),
        yaxis=dict(title='Percentage', range=[0, 100], gridcolor='#f0f0f0'),
        xaxis=dict(title=''),
        margin=dict(t=50, b=30, l=50, r=20),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif'),
        showlegend=False
    )

    return fig


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
            name=f'{name} (n={breakdown["total"]})',
            marker_color=COMPARISON_COLORS[i % len(COMPARISON_COLORS)],
            text=[f'{p:.1f}%' for p in pcts],
            textposition='inside',
            textfont=dict(color='white', size=11)
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


def create_comparison_chart(
    breakdown_1: Dict,
    breakdown_2: Dict,
    name_1: str,
    name_2: str
) -> go.Figure:
    """
    Create a grouped bar chart comparing sentiment distributions of two products.

    Args:
        breakdown_1: Sentiment breakdown for first product
        breakdown_2: Sentiment breakdown for second product
        name_1: Display name for first product
        name_2: Display name for second product

    Returns:
        Plotly Figure object for Gradio gr.Plot
    """
    categories = ['Positive', 'Neutral', 'Negative']
    product_1_pcts = [
        breakdown_1['positive_pct'],
        breakdown_1['neutral_pct'],
        breakdown_1['negative_pct']
    ]
    product_2_pcts = [
        breakdown_2['positive_pct'],
        breakdown_2['neutral_pct'],
        breakdown_2['negative_pct']
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=product_1_pcts,
        name=f'{name_1} (n={breakdown_1["total"]})',
        marker_color='#EE4D2D',
        text=[f'{p:.1f}%' for p in product_1_pcts],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=product_2_pcts,
        name=f'{name_2} (n={breakdown_2["total"]})',
        marker_color='#3498db',
        text=[f'{p:.1f}%' for p in product_2_pcts],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    fig.update_layout(
        title=dict(
            text='Sentiment Distribution Comparison',
            font=dict(size=16, family='-apple-system, BlinkMacSystemFont, sans-serif'),
            x=0.5
        ),
        barmode='group',
        yaxis=dict(title='Percentage of Reviews', range=[0, 100], gridcolor='#f0f0f0'),
        xaxis=dict(title=''),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        margin=dict(t=60, b=60, l=50, r=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif'),
        bargap=0.3
    )

    return fig

"""
Image Scraper for APDS-A3 Product Catalog

Generates product images using a dual approach:
1. Attempt to scrape from Nykaa URLs (likely to fail due to React SPA)
2. Generate styled placeholder images with brand-based gradients (primary fallback)

Usage:
    python image_scraper.py
"""

import os
import hashlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import requests
from bs4 import BeautifulSoup
from io import BytesIO

# Brand color palette - pastel gradients for cosmetics aesthetic
BRAND_COLORS = [
    ('#FFB6C1', '#FF69B4'),  # pink gradients
    ('#DDA0DD', '#BA55D3'),  # purple
    ('#87CEEB', '#4169E1'),  # blue
    ('#98FB98', '#3CB371'),  # green
    ('#FFD700', '#FF8C00'),  # gold/orange
    ('#FFA07A', '#FF4500'),  # coral/red
    ('#E6E6FA', '#9370DB'),  # lavender
    ('#F0E68C', '#DAA520'),  # khaki/goldenrod
]

# Cosmetics category icons (Unicode symbols)
COSMETIC_ICONS = ['💄', '💋', '✨', '🌸', '🌺', '💅', '🎀', '🦋']


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def interpolate_color(color1, color2, factor):
    """Interpolate between two RGB colors. Factor is 0.0 to 1.0."""
    return tuple(
        int(color1[i] + (color2[i] - color1[i]) * factor)
        for i in range(3)
    )


def generate_placeholder(product_id, brand_name, product_title, output_dir):
    """Generate a styled placeholder image for a product."""
    width, height = 300, 300
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Pick colors based on brand name hash
    brand_hash = int(hashlib.md5(brand_name.encode()).hexdigest(), 16)
    color_top_hex, color_bottom_hex = BRAND_COLORS[brand_hash % len(BRAND_COLORS)]
    color_top = hex_to_rgb(color_top_hex)
    color_bottom = hex_to_rgb(color_bottom_hex)

    # Draw vertical gradient background
    for y in range(height):
        factor = y / height
        color = interpolate_color(color_top, color_bottom, factor)
        draw.line([(0, y), (width, y)], fill=color)

    # Get brand initial (first letter, uppercase)
    brand_initial = brand_name[0].upper() if brand_name else 'P'

    # Draw large brand initial in center
    try:
        # Try to use a reasonable font size
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
    except:
        # Fallback to default
        font_large = ImageFont.load_default()

    # Draw initial with white text and subtle shadow
    initial_bbox = draw.textbbox((0, 0), brand_initial, font=font_large)
    initial_w = initial_bbox[2] - initial_bbox[0]
    initial_h = initial_bbox[3] - initial_bbox[1]
    initial_x = (width - initial_w) // 2
    initial_y = (height - initial_h) // 2 - 30

    # Shadow
    draw.text((initial_x + 3, initial_y + 3), brand_initial, fill=(0, 0, 0, 100), font=font_large)
    # Main text
    draw.text((initial_x, initial_y), brand_initial, fill='white', font=font_large)

    # Pick cosmetic icon based on product hash
    product_hash = int(hashlib.md5(product_title.encode()).hexdigest(), 16)
    icon = COSMETIC_ICONS[product_hash % len(COSMETIC_ICONS)]

    # Draw icon above initial
    try:
        font_icon = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 40)
    except:
        font_icon = ImageFont.load_default()

    icon_bbox = draw.textbbox((0, 0), icon, font=font_icon)
    icon_w = icon_bbox[2] - icon_bbox[0]
    icon_x = (width - icon_w) // 2
    icon_y = initial_y - 50
    draw.text((icon_x, icon_y), icon, font=font_icon, embedded_color=True)

    # Draw truncated product title at bottom
    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font_small = ImageFont.load_default()

    # Truncate title to fit
    title_display = product_title[:40] + '...' if len(product_title) > 40 else product_title

    # Draw semi-transparent background for text readability
    text_bg_y = height - 50
    draw.rectangle([(0, text_bg_y), (width, height)], fill=(0, 0, 0, 150))

    # Draw title text
    title_lines = []
    words = title_display.split()
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font_small)
        if bbox[2] - bbox[0] < width - 20:
            current_line.append(word)
        else:
            if current_line:
                title_lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        title_lines.append(' '.join(current_line))

    # Draw up to 2 lines
    title_lines = title_lines[:2]
    text_y = text_bg_y + 8
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=font_small)
        line_w = bbox[2] - bbox[0]
        text_x = (width - line_w) // 2
        draw.text((text_x, text_y), line, fill='white', font=font_small)
        text_y += 18

    # Draw brand name at very bottom
    try:
        font_tiny = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        font_tiny = ImageFont.load_default()

    brand_display = brand_name[:30] if len(brand_name) <= 30 else brand_name[:27] + '...'
    brand_bbox = draw.textbbox((0, 0), brand_display, font=font_tiny)
    brand_w = brand_bbox[2] - brand_bbox[0]
    brand_x = (width - brand_w) // 2
    draw.text((brand_x, height - 15), brand_display, fill='#CCCCCC', font=font_tiny)

    # Save
    output_path = os.path.join(output_dir, f"{product_id}.png")
    img.save(output_path, 'PNG')
    print(f"Generated placeholder: {product_id}")


def try_scrape_image(product_url, product_id, output_dir):
    """Attempt to scrape product image from URL. Returns True if successful."""
    if pd.isna(product_url) or not product_url.startswith('http'):
        return False

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(product_url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Try og:image meta tag first
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            img_url = og_image['content']
        else:
            # Try to find product image
            img_tags = soup.find_all('img', class_=lambda x: x and 'product' in x.lower())
            if img_tags:
                img_url = img_tags[0].get('src') or img_tags[0].get('data-src')
            else:
                return False

        if not img_url:
            return False

        # Download image
        if not img_url.startswith('http'):
            img_url = 'https:' + img_url if img_url.startswith('//') else product_url.rsplit('/', 1)[0] + '/' + img_url

        img_response = requests.get(img_url, headers=headers, timeout=5)
        img_response.raise_for_status()

        # Resize and save
        img = Image.open(BytesIO(img_response.content))
        img = img.convert('RGB')
        img = img.resize((300, 300), Image.Resampling.LANCZOS)

        output_path = os.path.join(output_dir, f"{product_id}.png")
        img.save(output_path, 'PNG')
        print(f"Scraped image: {product_id}")
        return True

    except Exception as e:
        # Scraping failed - expected for React SPAs
        return False


def main():
    """Main execution function."""
    print("Starting image generation for product catalog...")

    # Load data
    df = pd.read_csv('notebooks/processed.csv')
    products = df.drop_duplicates('product_id')[['product_id', 'brand_name', 'product_title', 'product_url']]

    print(f"Found {len(products)} unique products")

    # Create output directory
    os.makedirs('images', exist_ok=True)

    success_count = 0
    fallback_count = 0
    skipped_count = 0

    for idx, row in products.iterrows():
        product_id = row['product_id']
        output_path = f"images/{product_id}.png"

        # Skip if already exists
        if os.path.exists(output_path):
            skipped_count += 1
            continue

        # Try scraping first
        scraped = try_scrape_image(row['product_url'], product_id, 'images')
        if scraped:
            success_count += 1
        else:
            # Generate placeholder (primary approach)
            generate_placeholder(
                product_id,
                row['brand_name'],
                row['product_title'],
                'images'
            )
            fallback_count += 1

    print("\n" + "="*60)
    print("Image generation complete!")
    print(f"  Scraped: {success_count}")
    print(f"  Generated (placeholder): {fallback_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Total: {len(products)}")
    print("="*60)


if __name__ == '__main__':
    main()

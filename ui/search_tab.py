"""Tab 1: Search Products — brand/keyword search with pagination and product detail."""

import gradio as gr
from ui.components import render_product_grid
from ui.product_detail import render_product_detail
from core.search_engine import search_products


PRODUCTS_PER_PAGE = 20


def build_search_tab(products_df, product_vectors, embeddings_dict, ctx):
    """Build the Search Products tab UI and wire event handlers."""

    with gr.Tab("🔍 Search Products"):
        gr.Markdown("### Find products by brand or keyword")

        search_query = gr.Textbox(
            placeholder="Search by brand or keyword...",
            label="",
            show_label=False
        )

        search_btn = gr.Button("Search", variant="primary")

        all_products_list = products_df.to_dict('records')
        initial_grid = render_product_grid(all_products_list[:PRODUCTS_PER_PAGE], ctx['images_dir'])
        search_results_count = gr.Markdown(f"**Showing 1–{min(PRODUCTS_PER_PAGE, len(all_products_list))} of {len(all_products_list)} products** — Search to filter")
        search_results_grid = gr.HTML(initial_grid)

        with gr.Row():
            prev_page_btn = gr.Button("← Previous", variant="secondary", size="sm")
            page_info = gr.Markdown(f"Page 1 of {(len(all_products_list) + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE}")
            next_page_btn = gr.Button("Next →", variant="secondary", size="sm")

        current_page = gr.State(1)
        current_results = gr.State(all_products_list)

        selected_product_id_hidden = gr.Textbox(
            elem_id="selected_product_id",
            visible=True,
            elem_classes=["hidden-input"]
        )

        product_detail_panel = gr.HTML(visible=False)
        back_to_results_btn = gr.Button("← Back to Search Results", visible=False)

        def on_search(query):
            if not query or query.strip() == "":
                all_prods = products_df.to_dict('records')
            else:
                all_prods = search_products(query, products_df, product_vectors, embeddings_dict, top_n=40)

            total = len(all_prods)
            page = 1
            start = 0
            end = min(PRODUCTS_PER_PAGE, total)
            total_pages = max(1, (total + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE)

            if total == 0:
                count_msg = f"**No products found** for \"{query}\""
            else:
                count_msg = f"**Showing {start+1}–{end} of {total} products**" + (f" for \"{query}\"" if query and query.strip() else "")
            grid_html = render_product_grid(all_prods[start:end], ctx['images_dir'])
            page_msg = f"Page {page} of {total_pages}"

            return [
                count_msg,
                grid_html,
                gr.update(visible=False),
                gr.update(visible=False),
                page_msg,
                page,
                all_prods
            ]

        search_btn.click(
            fn=on_search,
            inputs=[search_query],
            outputs=[search_results_count, search_results_grid, product_detail_panel, back_to_results_btn, page_info, current_page, current_results],
            show_progress="hidden"
        )

        search_query.submit(
            fn=on_search,
            inputs=[search_query],
            outputs=[search_results_count, search_results_grid, product_detail_panel, back_to_results_btn, page_info, current_page, current_results],
            show_progress="hidden"
        )

        def go_next(page, results):
            total = len(results)
            total_pages = max(1, (total + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE)
            page = min(page + 1, total_pages)
            start = (page - 1) * PRODUCTS_PER_PAGE
            end = min(start + PRODUCTS_PER_PAGE, total)
            count_msg = f"**Showing {start+1}–{end} of {total} products**"
            grid_html = render_product_grid(results[start:end], ctx['images_dir'])
            page_msg = f"Page {page} of {total_pages}"
            return count_msg, grid_html, page_msg, page

        def go_prev(page, results):
            total = len(results)
            total_pages = max(1, (total + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE)
            page = max(page - 1, 1)
            start = (page - 1) * PRODUCTS_PER_PAGE
            end = min(start + PRODUCTS_PER_PAGE, total)
            count_msg = f"**Showing {start+1}–{end} of {total} products**"
            grid_html = render_product_grid(results[start:end], ctx['images_dir'])
            page_msg = f"Page {page} of {total_pages}"
            return count_msg, grid_html, page_msg, page

        next_page_btn.click(
            fn=go_next,
            inputs=[current_page, current_results],
            outputs=[search_results_count, search_results_grid, page_info, current_page],
            show_progress="hidden"
        )

        prev_page_btn.click(
            fn=go_prev,
            inputs=[current_page, current_results],
            outputs=[search_results_count, search_results_grid, page_info, current_page],
            show_progress="hidden"
        )

        def on_product_select(product_id_str):
            if not product_id_str or product_id_str.strip() == "":
                return [gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]

            try:
                raw_id = product_id_str.split("_")[0]
                product_id = int(raw_id)
                detail_html = render_product_detail(product_id, ctx, target_elem_id="selected_product_id")

                return [
                    gr.update(value=detail_html, visible=True),
                    gr.update(visible=True),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                ]
            except Exception as e:
                print(f"Error loading product detail: {e}")
                return [gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]

        selected_product_id_hidden.change(
            fn=on_product_select,
            inputs=[selected_product_id_hidden],
            outputs=[product_detail_panel, back_to_results_btn, search_results_grid, search_results_count, prev_page_btn, page_info, next_page_btn],
            show_progress="hidden"
        )

        def on_back(page, results):
            total = len(results)
            total_pages = max(1, (total + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE)
            start = (page - 1) * PRODUCTS_PER_PAGE
            end = min(start + PRODUCTS_PER_PAGE, total)
            grid_html = render_product_grid(results[start:end], ctx['images_dir'])
            count_msg = f"**Showing {start+1}–{end} of {total} products**"
            page_msg = f"Page {page} of {total_pages}"
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                grid_html,
                gr.update(value=count_msg, visible=True),
                gr.update(visible=True),
                gr.update(value=page_msg, visible=True),
                gr.update(visible=True),
            ]

        back_to_results_btn.click(
            fn=on_back,
            inputs=[current_page, current_results],
            outputs=[product_detail_panel, back_to_results_btn, selected_product_id_hidden, search_results_grid, search_results_count, prev_page_btn, page_info, next_page_btn],
            show_progress="hidden"
        )

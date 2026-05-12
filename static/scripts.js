// Helper: set Gradio textbox value and trigger change detection
function setGradioValue(selector, value) {
    var el = document.querySelector(selector + ' textarea') || document.querySelector(selector + ' input');
    if (!el) return;
    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value') ||
                                  Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
    if (nativeInputValueSetter && nativeInputValueSetter.set) {
        nativeInputValueSetter.set.call(el, value);
    } else {
        el.value = value;
    }
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
}

// Review filter by star rating
function filterReviews(rating, btn) {
    var container = document.getElementById('pdp-reviews-container');
    if (!container) return;
    var items = Array.from(container.querySelectorAll('.shopee-review-item'));
    items.forEach(function(item) {
        if (rating === 'all') { item.style.display = ''; }
        else { item.style.display = (Math.floor(parseFloat(item.getAttribute('data-rating'))) == parseInt(rating)) ? '' : 'none'; }
    });
    document.querySelectorAll('.shopee-filter-btn').forEach(function(b) { b.classList.remove('active'); });
    if (btn) btn.classList.add('active');
}

// Review sort
function sortReviews(mode, btn) {
    var container = document.getElementById('pdp-reviews-container');
    if (!container) return;
    var items = Array.from(container.querySelectorAll('.shopee-review-item'));
    items.sort(function(a, b) {
        if (mode === 'rating-desc') return parseFloat(b.getAttribute('data-rating')) - parseFloat(a.getAttribute('data-rating'));
        if (mode === 'rating-asc') return parseFloat(a.getAttribute('data-rating')) - parseFloat(b.getAttribute('data-rating'));
        if (mode === 'alpha-az') return a.getAttribute('data-title').localeCompare(b.getAttribute('data-title'));
        return 0;
    });
    items.forEach(function(item) { container.appendChild(item); });
    document.querySelectorAll('.shopee-sort-btn').forEach(function(b) { b.classList.remove('active'); });
    if (btn) btn.classList.add('active');
}

// Final label selection
function selectLabel(val, btn) {
    document.querySelectorAll('.label-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    var el = document.querySelector('#final_label_hidden textarea, #final_label_hidden input');
    if (el) { el.value = val; el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); }
}

// Star rating widget init
function initStarRating() {
    var container = document.getElementById('star-rating-stars');
    if (!container || container.dataset.init) return;
    container.dataset.init = '1';
    var stars = container.querySelectorAll('.star-icon');
    var text = document.getElementById('star-rating-text');
    var currentRating = 4;

    function updateStars(rating, permanent) {
        stars.forEach(function(s) {
            var val = parseInt(s.getAttribute('data-value'));
            if (val <= rating) { s.classList.add('active'); }
            else { s.classList.remove('active'); }
        });
        if (text) text.textContent = rating + ' out of 5';
        if (permanent) {
            currentRating = rating;
            var el = document.querySelector('#star_rating_hidden textarea, #star_rating_hidden input');
            if (el) { el.value = rating.toString(); el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); }
        }
    }

    stars.forEach(function(star) {
        star.addEventListener('mouseenter', function() { updateStars(parseInt(this.getAttribute('data-value')), false); });
        star.addEventListener('click', function() { updateStars(parseInt(this.getAttribute('data-value')), true); });
    });
    container.addEventListener('mouseleave', function() { updateStars(currentRating, false); });
}

// Sync label buttons when Gradio updates the hidden textbox (after prediction)
function syncLabelButtons() {
    var el = document.querySelector('#final_label_hidden textarea, #final_label_hidden input');
    if (!el) return;
    var val = el.value;
    document.querySelectorAll('.label-btn').forEach(function(b) {
        if (b.textContent.trim() === val) { b.classList.add('active'); }
        else { b.classList.remove('active'); }
    });
}

// Observe DOM for dynamic Gradio content
new MutationObserver(function() {
    if (document.getElementById('star-rating-stars') && !document.getElementById('star-rating-stars').dataset.init) {
        initStarRating();
    }
    syncLabelButtons();
}).observe(document.documentElement, {childList: true, subtree: true});
document.addEventListener('DOMContentLoaded', function() { initStarRating(); syncLabelButtons(); });

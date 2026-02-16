/* ========================================
   Blue Plumeria — Main JavaScript
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {

  // ── Mobile Nav Toggle ──
  const hamburger = document.querySelector('.hamburger');
  const navLinks = document.querySelector('.nav-links');

  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('open');
      hamburger.classList.toggle('active');
      hamburger.setAttribute('aria-expanded', isOpen);
    });

    // Close nav when clicking a link
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('open');
        hamburger.classList.remove('active');
        hamburger.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // ── Product Filtering (Shop page) ──
  const filterTabs = document.querySelectorAll('.filter-tab');
  const productGrid = document.getElementById('product-grid');
  const productCountEl = document.getElementById('product-count');

  if (filterTabs.length && productGrid) {
    const cards = Array.from(productGrid.querySelectorAll('.product-card'));

    function filterProducts(category) {
      let visibleCount = 0;

      cards.forEach(card => {
        const cardCategory = card.getAttribute('data-category');
        if (category === 'all' || cardCategory === category) {
          card.style.display = '';
          visibleCount++;
        } else {
          card.style.display = 'none';
        }
      });

      if (productCountEl) {
        productCountEl.textContent = visibleCount;
      }

      // Update active tab
      filterTabs.forEach(tab => {
        tab.classList.toggle('active', tab.getAttribute('data-filter') === category);
      });
    }

    // Tab click handlers
    filterTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const filter = tab.getAttribute('data-filter');
        filterProducts(filter);
        // Update URL hash without scrolling
        history.replaceState(null, '', '#' + filter);
      });
    });

    // Check URL hash on load
    const hash = window.location.hash.replace('#', '');
    if (hash && ['earrings', 'necklaces', 'bracelets'].includes(hash)) {
      filterProducts(hash);
    }

    // Listen for hash changes (e.g. from category tiles on homepage)
    window.addEventListener('hashchange', () => {
      const newHash = window.location.hash.replace('#', '');
      if (newHash && ['earrings', 'necklaces', 'bracelets', 'all'].includes(newHash)) {
        filterProducts(newHash);
      }
    });
  }

  // ── Sort Functionality ──
  const sortSelect = document.getElementById('sort-select');

  if (sortSelect && productGrid) {
    const cards = Array.from(productGrid.querySelectorAll('.product-card'));
    const originalOrder = cards.map(card => card);

    sortSelect.addEventListener('change', () => {
      const value = sortSelect.value;
      let sorted;

      if (value === 'price-low') {
        sorted = [...cards].sort((a, b) => {
          return parseFloat(a.dataset.price) - parseFloat(b.dataset.price);
        });
      } else if (value === 'price-high') {
        sorted = [...cards].sort((a, b) => {
          return parseFloat(b.dataset.price) - parseFloat(a.dataset.price);
        });
      } else {
        sorted = originalOrder;
      }

      sorted.forEach(card => productGrid.appendChild(card));
    });
  }

  // ── Product Showcase Sliders ──
  document.querySelectorAll('[data-slider]').forEach(slider => {
    const images = slider.querySelectorAll('img');
    const prevBtn = slider.querySelector('.prev');
    const nextBtn = slider.querySelector('.next');
    const dotsContainer = slider.parentElement.querySelector('.slider-dots');
    const dots = dotsContainer ? dotsContainer.querySelectorAll('.slider-dot') : [];
    let current = 0;

    function showSlide(index) {
      images.forEach(img => img.classList.remove('active'));
      dots.forEach(dot => dot.classList.remove('active'));
      current = (index + images.length) % images.length;
      images[current].classList.add('active');
      if (dots[current]) dots[current].classList.add('active');
    }

    if (prevBtn) prevBtn.addEventListener('click', () => showSlide(current - 1));
    if (nextBtn) nextBtn.addEventListener('click', () => showSlide(current + 1));
    dots.forEach((dot, i) => {
      dot.addEventListener('click', () => showSlide(i));
    });
  });

  // ── Active Nav Link Highlighting ──
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPage) {
      link.classList.add('active');
    }
  });

});

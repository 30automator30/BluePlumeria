/* ========================================
   Blue Plumeria — Main JavaScript
   Interaction layer: navigation, shop filtering,
   showcase sliders, scroll reveals, form validation.
   Progressive enhancement throughout — every block
   guards its own elements so any page can omit them.
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {

  // Respect the visitor's motion preference across all enhancements.
  const prefersReducedMotion =
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ────────────────────────────────────────
     Mobile Nav Toggle
     ──────────────────────────────────────── */
  const hamburger = document.querySelector('.hamburger');
  const navLinks = document.querySelector('.nav-links');

  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('open');
      hamburger.classList.toggle('active');
      hamburger.setAttribute('aria-expanded', isOpen);
    });

    // Close the menu when a destination is chosen.
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('open');
        hamburger.classList.remove('active');
        hamburger.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* ────────────────────────────────────────
     Collection Filtering + Grouping (Shop page)
     Cards carry data-collection (and optional data-tier). On load
     they're ordered tier → collection, and headings are inserted at
     each group boundary: a large tier heading (e.g. "The Signature
     Collection") and an overline collection heading (e.g. "Sea &
     Shore"). Filter tabs then show/hide by collection. Every block
     guards its own elements, so other pages skip this.
     ──────────────────────────────────────── */
  const filterTabs = document.querySelectorAll('.filter-tab[data-filter]');
  const showcaseSets = Array.from(document.querySelectorAll('.showcase-set[data-collection]'));
  const productCountEl = document.getElementById('product-count');

  // Two-level structure: tier (Signature / Everyday) → material collection.
  // A piece without data-tier is treated as Signature.
  const TIERS = [
    { key: 'signature', label: 'The Signature Collection' },
    { key: 'everyday',  label: 'Everyday Collection' }
  ];
  const COLLECTIONS = [
    { key: 'sea-shore',     label: 'Sea & Shore' },
    { key: 'stone-earth',   label: 'Stone & Earth' },
    { key: 'pearl-crystal', label: 'Pearl & Crystal' },
    { key: 'hand-woven',    label: 'Hand-Woven' }
  ];
  const tierOrder = TIERS.map(t => t.key);
  const collectionOrder = COLLECTIONS.map(c => c.key);
  const labelFor = (list, key) => (list.find(x => x.key === key) || {}).label || key;
  const tierOf = set => set.dataset.tier || 'signature';

  if (filterTabs.length && showcaseSets.length) {
    const grid = showcaseSets[0].parentElement;
    const headings = [];

    if (grid) {
      // Order by tier, then by collection.
      const ordered = [...showcaseSets].sort((a, b) => {
        const byTier = tierOrder.indexOf(tierOf(a)) - tierOrder.indexOf(tierOf(b));
        if (byTier !== 0) return byTier;
        return collectionOrder.indexOf(a.dataset.collection) - collectionOrder.indexOf(b.dataset.collection);
      });

      // Re-append in order, dropping a heading at each group boundary.
      let lastTier = null, lastCollection = null;
      ordered.forEach(set => {
        const tier = tierOf(set);
        const collection = set.dataset.collection;

        if (tier !== lastTier) {
          const h = document.createElement('h2');
          h.className = 'section-heading tier-heading';
          h.textContent = labelFor(TIERS, tier);
          grid.appendChild(h);
          headings.push({ el: h, type: 'tier', tier });
          lastTier = tier;
          lastCollection = null;
        }
        if (collection !== lastCollection) {
          const h = document.createElement('h3');
          h.className = 'collection-heading';
          h.textContent = labelFor(COLLECTIONS, collection);
          grid.appendChild(h);
          headings.push({ el: h, type: 'collection', collection });
          lastCollection = collection;
        }
        grid.appendChild(set);
      });
    }

    function filterCollection(collection) {
      let visible = 0;

      showcaseSets.forEach(set => {
        const match = collection === 'all' || set.dataset.collection === collection;
        set.style.display = match ? '' : 'none';
        if (match) {
          set.classList.add('in'); // ensure a filtered-in card is revealed, never blank
          visible++;
        }
      });

      // "All" shows the full labelled structure; a single collection
      // shows only its own heading (the active tab already names it).
      headings.forEach(h => {
        const show = collection === 'all'
          ? true
          : (h.type === 'collection' && h.collection === collection);
        h.el.style.display = show ? '' : 'none';
      });

      if (productCountEl) productCountEl.textContent = visible;

      filterTabs.forEach(tab =>
        tab.classList.toggle('active', tab.dataset.filter === collection));
    }

    filterTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        filterCollection(tab.dataset.filter);
        // Keep the URL shareable without jumping the scroll position.
        history.replaceState(null, '', '#' + tab.dataset.filter);
      });
    });

    // Honor a collection hash on arrival, and react to changes.
    const validFilters = ['all', ...collectionOrder];
    const applyHash = () => {
      const hash = window.location.hash.replace('#', '');
      if (validFilters.includes(hash)) filterCollection(hash);
    };
    applyHash();
    window.addEventListener('hashchange', applyHash);
  }

  /* ────────────────────────────────────────
     Product Showcase Sliders
     Fully user-controlled — no auto-advance. Each slider
     wraps around, keeps its dots in sync, and answers to
     the arrow keys when a control is focused.
     ──────────────────────────────────────── */
  document.querySelectorAll('[data-slider]').forEach(slider => {
    const images = slider.querySelectorAll('img');
    const prevBtn = slider.querySelector('.prev');
    const nextBtn = slider.querySelector('.next');

    // Dots may live beside the slider inside .showcase-media,
    // or elsewhere within the showcase set — find them robustly.
    let dotsContainer = slider.parentElement
      ? slider.parentElement.querySelector('.slider-dots')
      : null;
    if (!dotsContainer) {
      const set = slider.closest('.showcase-set');
      dotsContainer = set ? set.querySelector('.slider-dots') : null;
    }
    const dots = dotsContainer ? dotsContainer.querySelectorAll('.slider-dot') : [];

    if (!images.length) return;

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

    // Keyboard support: left/right arrows while a slider
    // control (arrow button or dot) holds focus.
    function handleArrowKeys(event) {
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        showSlide(current - 1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        showSlide(current + 1);
      }
    }

    [prevBtn, nextBtn, ...dots].forEach(control => {
      if (control) control.addEventListener('keydown', handleArrowKeys);
    });

    // No auto-advance exists to pause, but should one ever be
    // added, gate it behind !prefersReducedMotion.
  });

  /* ────────────────────────────────────────
     Scroll Reveal
     Elements marked .reveal fade in on approach. Siblings
     sharing a parent get a slight stagger (capped at the
     fourth item) so groups arrive as a quiet cascade.
     ──────────────────────────────────────── */
  const revealEls = document.querySelectorAll('.reveal');

  if (revealEls.length) {
    if (prefersReducedMotion || !('IntersectionObserver' in window)) {
      // No motion, or no observer support — show everything at once.
      revealEls.forEach(el => el.classList.add('in'));
    } else {
      const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
          if (!entry.isIntersecting) return;

          const el = entry.target;

          // Stagger siblings revealed from the same parent:
          // each gets ~90ms more than the last, capped at the
          // fourth item so late arrivals never lag behind.
          const parent = el.parentElement;
          if (parent) {
            const count = parseInt(parent.dataset.revealCount || '0', 10);
            const delay = Math.min(count, 3) * 90;
            if (delay > 0) el.style.transitionDelay = delay + 'ms';
            parent.dataset.revealCount = count + 1;
          }

          el.classList.add('in');

          // Clear the inline delay once the entrance finishes so
          // it never affects hover or other later transitions.
          el.addEventListener('transitionend', function clearDelay() {
            el.style.transitionDelay = '';
            el.removeEventListener('transitionend', clearDelay);
          });

          observer.unobserve(el);
        });
      }, {
        // Begin the entrance just before the element scrolls into view.
        rootMargin: '0px 0px -8% 0px',
        threshold: 0.1
      });

      revealEls.forEach(el => revealObserver.observe(el));
    }
  }

  /* ────────────────────────────────────────
     Contact Form Validation
     Inline, unobtrusive messages — never window.alert.
     A valid form submits natively to Formspree.
     ──────────────────────────────────────── */
  const contactForm = document.querySelector('.contact-form form');

  if (contactForm) {
    // Take over from native browser bubbles so our inline
    // messages (styled to the brand) handle all feedback.
    contactForm.setAttribute('novalidate', '');

    // Field definitions: element, validity test, and message copy.
    const fields = [
      {
        input: contactForm.querySelector('#name'),
        message: 'Enter your name',
        isValid: value => value.trim().length > 0
      },
      {
        input: contactForm.querySelector('#email'),
        message: 'Enter a valid email address',
        isValid: value => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.trim())
      },
      {
        input: contactForm.querySelector('#message'),
        message: 'Add a short message (at least 10 characters)',
        isValid: value => value.trim().length >= 10
      }
    ].filter(field => field.input);

    // Lazily create (once) the inline error element for a field,
    // placed directly after its input or textarea.
    function getErrorEl(input) {
      let errorEl = input.nextElementSibling;
      if (!errorEl || !errorEl.classList.contains('field-error')) {
        errorEl = document.createElement('p');
        errorEl.className = 'field-error';
        input.insertAdjacentElement('afterend', errorEl);
      }
      return errorEl;
    }

    function showError(input, message) {
      input.classList.add('invalid');
      const errorEl = getErrorEl(input);
      errorEl.textContent = message;
      errorEl.classList.add('show');
    }

    function clearError(input) {
      input.classList.remove('invalid');
      const errorEl = input.nextElementSibling;
      if (errorEl && errorEl.classList.contains('field-error')) {
        errorEl.classList.remove('show');
      }
    }

    // Errors dissolve as soon as the visitor starts correcting.
    fields.forEach(({ input }) => {
      input.addEventListener('input', () => clearError(input));
    });

    contactForm.addEventListener('submit', event => {
      let firstInvalid = null;

      fields.forEach(({ input, message, isValid }) => {
        if (!isValid(input.value)) {
          showError(input, message);
          if (!firstInvalid) firstInvalid = input;
        } else {
          clearError(input);
        }
      });

      if (firstInvalid) {
        event.preventDefault();
        firstInvalid.focus();
      }
      // Otherwise, let the native Formspree submission proceed.
    });
  }

  /* ────────────────────────────────────────
     Active Nav Link Highlighting
     ──────────────────────────────────────── */
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPage) {
      link.classList.add('active');
    }
  });

});

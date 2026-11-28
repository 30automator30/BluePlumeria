/* Blue Plumeria: analytics consent.
 *
 * Google Analytics 4 (G-3TWSTSRBSR) loads ONLY after the visitor clicks
 * "Accept". Before that, nothing is requested from Google: no gtag.js,
 * no cookieless pings. "Decline", or a browser sending the Global Privacy
 * Control signal, means GA never loads. The choice is kept in
 * localStorage; the footer "Cookie settings" button reopens it.
 *
 * main.js sends Snipcart cart events through window.gtag only when it
 * exists, so they simply don't fire until consent is given.
 */
(function () {
  'use strict';

  var GA_ID = 'G-3TWSTSRBSR';
  var KEY = 'bp-analytics-consent'; // "granted" | "denied"
  var gaLoaded = false;

  function read() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function write(v) {
    try { localStorage.setItem(KEY, v); } catch (e) { /* private mode: the choice lasts this page only */ }
  }

  function loadGA() {
    if (gaLoaded) return;
    gaLoaded = true;
    window.dataLayer = window.dataLayer || [];
    window.gtag = function () { window.dataLayer.push(arguments); };
    window.gtag('js', new Date());
    window.gtag('config', GA_ID);
    var s = document.createElement('script');
    s.async = true;
    s.src = 'https://www.googletagmanager.com/gtag/js?id=' + GA_ID;
    document.head.appendChild(s);
  }

  // Remove GA's first-party cookies (_ga, _ga_<id>) on this host and its parent domain.
  function clearGACookies() {
    var host = location.hostname;
    var domains = ['', host, '.' + host.replace(/^www\./, '')];
    document.cookie.split(';').forEach(function (c) {
      var name = c.split('=')[0].trim();
      if (name.indexOf('_ga') !== 0) return;
      domains.forEach(function (d) {
        document.cookie = name + '=; Max-Age=0; path=/' + (d ? '; domain=' + d : '');
      });
    });
  }

  var gpc = navigator.globalPrivacyControl === true;
  var choice = read();

  if (choice === 'granted' && !gpc) loadGA();
  // Without consent, make sure no GA cookie lingers: GA can re-write one in the
  // moment before the reload that follows a withdrawal.
  else clearGACookies();

  function closeBanner() {
    var b = document.getElementById('bp-consent');
    if (b) b.remove();
  }

  function decide(value) {
    write(value);
    choice = value;
    closeBanner();
    if (value === 'granted') {
      loadGA();
    } else {
      clearGACookies();
      // GA already running on this page can't be unloaded; reload without it.
      if (gaLoaded) location.reload();
    }
  }

  function showBanner() {
    if (document.getElementById('bp-consent')) return;
    var box = document.createElement('section');
    box.id = 'bp-consent';
    box.className = 'consent-banner';
    box.setAttribute('aria-label', 'Cookie choice');
    box.innerHTML =
      '<p class="consent-text">May we use Google Analytics cookies to see which pieces people ' +
      'look at and how the shop is used? Nothing is sent to Google unless you choose Accept. ' +
      'You can change your mind any time with “Cookie settings” at the bottom of the page. ' +
      '<a href="privacy.html">Privacy policy</a></p>' +
      '<div class="consent-actions">' +
      '<button type="button" class="consent-btn" data-consent="granted">Accept</button>' +
      '<button type="button" class="consent-btn" data-consent="denied">Decline</button>' +
      '</div>';
    box.addEventListener('click', function (e) {
      var v = e.target && e.target.getAttribute && e.target.getAttribute('data-consent');
      if (v) decide(v);
    });
    document.body.appendChild(box);
  }

  function init() {
    // The footer "Cookie settings" button reopens the choice.
    document.addEventListener('click', function (e) {
      var t = e.target && e.target.closest && e.target.closest('[data-consent-open]');
      if (!t) return;
      e.preventDefault();
      if (gpc) {
        alert('Your browser is sending a Global Privacy Control signal, so analytics stay off on this site.');
        return;
      }
      showBanner();
      var first = document.querySelector('#bp-consent .consent-btn');
      if (first) first.focus();
    });
    if (!gpc && choice !== 'granted' && choice !== 'denied') showBanner();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

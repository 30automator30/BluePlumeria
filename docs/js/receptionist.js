/* ============================================================
   Blue Plumeria — AI receptionist widget
   A self-contained floating concierge. Injects its own styles + markup,
   keeps the conversation in sessionStorage (survives page navigation),
   and posts to the ai-receptionist Supabase Edge Function.

   To add to a page: <script src="js/receptionist.js" defer></script>
   ============================================================ */
(function () {
  "use strict";
  if (window.__bpReceptionist) return;      // guard against double-inject
  window.__bpReceptionist = true;

  var ENDPOINT =
    "https://ktjxrxchrxtmyvlfsyof.supabase.co/functions/v1/ai-receptionist";
  // Public Supabase key (same one the storefront already uses).
  var KEY = "sb_publishable_8NSnU6lGLIt9GplZ-7hHUw_XvtDmX2s";
  var STORE_KEY = "bp_receptionist_v1";
  var GREETING =
    "Hi! I'm the Blue Plumeria studio assistant. Ask me about a piece, our " +
    "materials, or a custom order — I'm happy to help.";

  /* ---- conversation state ---- */
  var messages = load();          // [{role, content}]
  var busy = false;

  function load() {
    try { return JSON.parse(sessionStorage.getItem(STORE_KEY)) || []; }
    catch (e) { return []; }
  }
  function save() {
    try { sessionStorage.setItem(STORE_KEY, JSON.stringify(messages.slice(-24))); }
    catch (e) { /* private mode / full: ignore */ }
  }

  /* ---- styles (scoped under .bp-rcp) ---- */
  var css =
  '.bp-rcp,.bp-rcp *{box-sizing:border-box}' +
  '.bp-rcp-launch{position:fixed;right:22px;bottom:22px;z-index:9000;display:flex;' +
    'align-items:center;gap:.6rem;padding:.7rem .95rem .7rem .8rem;border:none;cursor:pointer;' +
    'border-radius:999px;background:var(--ink,#1C1712);color:var(--paper,#FCFAF6);' +
    'font-family:var(--font-body,"Outfit",Arial,sans-serif);font-size:.9rem;font-weight:500;' +
    'box-shadow:0 8px 30px rgba(28,23,18,.28);transition:transform .25s,box-shadow .25s}' +
  '.bp-rcp-launch:hover{transform:translateY(-2px);box-shadow:0 12px 38px rgba(28,23,18,.34)}' +
  '.bp-rcp-launch svg{width:22px;height:22px;flex:none;color:var(--gold,#B08D4F)}' +
  '.bp-rcp-launch .bp-rcp-close-x{display:none}' +
  '.bp-rcp.open .bp-rcp-launch .bp-rcp-label,.bp-rcp.open .bp-rcp-launch .bp-rcp-open-i{display:none}' +
  '.bp-rcp.open .bp-rcp-launch .bp-rcp-close-x{display:block}' +
  '.bp-rcp-panel{position:fixed;right:22px;bottom:88px;z-index:9000;width:380px;max-width:calc(100vw - 32px);' +
    'height:560px;max-height:calc(100vh - 130px);display:none;flex-direction:column;overflow:hidden;' +
    'background:var(--paper,#FCFAF6);border:1px solid var(--border,#E4DBCC);border-radius:18px;' +
    'box-shadow:0 24px 70px rgba(28,23,18,.28);font-family:var(--font-body,"Outfit",Arial,sans-serif);' +
    'opacity:0;transform:translateY(10px) scale(.98);transition:opacity .22s,transform .22s}' +
  '.bp-rcp.open .bp-rcp-panel{display:flex}' +
  '.bp-rcp.ready .bp-rcp-panel{opacity:1;transform:none}' +
  '.bp-rcp-head{padding:1rem 1.1rem;background:var(--ink,#1C1712);color:var(--paper,#FCFAF6)}' +
  '.bp-rcp-head h3{margin:0;font-family:var(--font-heading,"Cormorant Garamond",Georgia,serif);' +
    'font-weight:500;font-size:1.4rem;line-height:1.1}' +
  '.bp-rcp-head p{margin:.15rem 0 0;font-size:.78rem;color:rgba(252,250,246,.7);font-weight:300}' +
  '.bp-rcp-msgs{flex:1;overflow-y:auto;padding:1.1rem;display:flex;flex-direction:column;gap:.7rem;' +
    'background:var(--bone,#F5F0E7)}' +
  '.bp-rcp-msg{max-width:85%;padding:.6rem .85rem;border-radius:14px;font-size:.9rem;line-height:1.5;' +
    'font-weight:300;white-space:normal;word-wrap:break-word}' +
  '.bp-rcp-msg a{color:var(--gold-deep,#8C6D37);text-decoration:underline}' +
  '.bp-rcp-msg.bot{align-self:flex-start;background:var(--paper,#FCFAF6);color:var(--ink,#1C1712);' +
    'border:1px solid var(--border,#E4DBCC);border-bottom-left-radius:4px}' +
  '.bp-rcp-msg.me{align-self:flex-end;background:var(--slate,#6E8CA6);color:#fff;border-bottom-right-radius:4px}' +
  '.bp-rcp-msg.typing{display:flex;gap:4px;align-items:center}' +
  '.bp-rcp-msg.typing span{width:6px;height:6px;border-radius:50%;background:var(--muted,#8A7F6E);' +
    'animation:bp-rcp-bounce 1.2s infinite}' +
  '.bp-rcp-msg.typing span:nth-child(2){animation-delay:.15s}' +
  '.bp-rcp-msg.typing span:nth-child(3){animation-delay:.3s}' +
  '@keyframes bp-rcp-bounce{0%,60%,100%{transform:translateY(0);opacity:.5}30%{transform:translateY(-4px);opacity:1}}' +
  '.bp-rcp-foot{padding:.6rem .8rem;border-top:1px solid var(--border,#E4DBCC);background:var(--paper,#FCFAF6)}' +
  '.bp-rcp-form{display:flex;gap:.5rem;align-items:flex-end}' +
  '.bp-rcp-form textarea{flex:1;resize:none;border:1px solid var(--border,#E4DBCC);border-radius:12px;' +
    'padding:.55rem .7rem;font:inherit;font-size:.9rem;font-weight:300;color:var(--ink,#1C1712);' +
    'background:var(--paper,#FCFAF6);max-height:96px;line-height:1.4}' +
  '.bp-rcp-form textarea:focus{outline:none;border-color:var(--gold,#B08D4F)}' +
  '.bp-rcp-send{flex:none;width:40px;height:40px;border:none;border-radius:12px;cursor:pointer;' +
    'background:var(--ink,#1C1712);color:var(--paper,#FCFAF6);display:flex;align-items:center;justify-content:center}' +
  '.bp-rcp-send:disabled{opacity:.4;cursor:default}' +
  '.bp-rcp-send svg{width:18px;height:18px}' +
  '.bp-rcp-note{margin:.45rem 2px 0;font-size:.66rem;color:var(--muted,#8A7F6E);text-align:center;font-weight:300}' +
  '@media(max-width:480px){.bp-rcp-panel{right:8px;left:8px;bottom:82px;width:auto;' +
    'height:calc(100vh - 100px);max-height:none}.bp-rcp-launch{right:14px;bottom:14px}}';

  /* ---- build DOM ---- */
  var root = document.createElement("div");
  root.className = "bp-rcp";
  root.innerHTML =
    '<style>' + css + '</style>' +
    '<button class="bp-rcp-launch" aria-haspopup="dialog" aria-expanded="false" aria-label="Chat with the studio">' +
      '<span class="bp-rcp-open-i" aria-hidden="true">' + chatIcon() + '</span>' +
      '<span class="bp-rcp-close-x" aria-hidden="true">' + closeIcon() + '</span>' +
      '<span class="bp-rcp-label">Ask the studio</span>' +
    '</button>' +
    '<section class="bp-rcp-panel" role="dialog" aria-label="Blue Plumeria studio assistant" aria-modal="false">' +
      '<header class="bp-rcp-head"><h3>Studio Assistant</h3><p>Questions, pieces & custom orders</p></header>' +
      '<div class="bp-rcp-msgs" role="log" aria-live="polite" aria-atomic="false"></div>' +
      '<div class="bp-rcp-foot">' +
        '<form class="bp-rcp-form">' +
          '<textarea rows="1" placeholder="Type your message…" aria-label="Your message"></textarea>' +
          '<button type="submit" class="bp-rcp-send" aria-label="Send">' + sendIcon() + '</button>' +
        '</form>' +
        '<p class="bp-rcp-note">AI assistant — replies may be imperfect. Email hello@blue-plumeria.com anytime.</p>' +
      '</div>' +
    '</section>';
  document.body.appendChild(root);

  var launch = root.querySelector(".bp-rcp-launch");
  var panel = root.querySelector(".bp-rcp-panel");
  var msgsEl = root.querySelector(".bp-rcp-msgs");
  var form = root.querySelector(".bp-rcp-form");
  var input = form.querySelector("textarea");
  var sendBtn = form.querySelector(".bp-rcp-send");

  /* ---- open / close ---- */
  function toggle(open) {
    var isOpen = open != null ? open : !root.classList.contains("open");
    root.classList.toggle("open", isOpen);
    launch.setAttribute("aria-expanded", String(isOpen));
    if (isOpen) {
      requestAnimationFrame(function () { root.classList.add("ready"); });
      renderAll();
      setTimeout(function () { input.focus(); }, 80);
    } else {
      root.classList.remove("ready");
    }
  }
  launch.addEventListener("click", function () { toggle(); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && root.classList.contains("open")) toggle(false);
  });

  /* ---- input behaviour ---- */
  input.addEventListener("input", function () {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 96) + "px";
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = input.value.trim();
    if (!text || busy) return;
    input.value = "";
    input.style.height = "auto";
    send(text);
  });

  /* ---- render ---- */
  function bubble(role, html) {
    var d = document.createElement("div");
    d.className = "bp-rcp-msg " + (role === "user" ? "me" : "bot");
    d.innerHTML = html;
    return d;
  }
  function renderAll() {
    msgsEl.innerHTML = "";
    msgsEl.appendChild(bubble("assistant", fmt(GREETING)));
    messages.forEach(function (m) {
      msgsEl.appendChild(bubble(m.role, m.role === "user" ? esc(m.content) : fmt(m.content)));
    });
    scroll();
  }
  function scroll() { msgsEl.scrollTop = msgsEl.scrollHeight; }

  /* ---- send to the receptionist ---- */
  function send(text) {
    busy = true; sendBtn.disabled = true;
    messages.push({ role: "user", content: text }); save();
    msgsEl.appendChild(bubble("user", esc(text)));
    var typing = document.createElement("div");
    typing.className = "bp-rcp-msg bot typing";
    typing.innerHTML = "<span></span><span></span><span></span>";
    msgsEl.appendChild(typing); scroll();

    fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json", apikey: KEY, Authorization: "Bearer " + KEY },
      body: JSON.stringify({ messages: messages.slice(-24) }),
    })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r); })
      .then(function (data) {
        var reply = (data && data.reply) ||
          "Sorry — I hit a snag. Please email hello@blue-plumeria.com and we'll help.";
        messages.push({ role: "assistant", content: reply }); save();
        typing.remove();
        msgsEl.appendChild(bubble("assistant", fmt(reply))); scroll();
      })
      .catch(function () {
        typing.remove();
        msgsEl.appendChild(bubble("assistant", fmt(
          "Sorry — I couldn't reach the studio just now. Please email " +
          "[hello@blue-plumeria.com](mailto:hello@blue-plumeria.com) and we'll get right back to you."
        ))); scroll();
      })
      .finally(function () { busy = false; sendBtn.disabled = false; input.focus(); });
  }

  /* ---- safe text -> html (escape first, then a tiny bit of markdown) ---- */
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function safeHref(u) {
    // Only allow http(s), root-relative, and mailto links.
    if (/^(https?:\/\/|\/|mailto:)/i.test(u) && !/[<>"']/.test(u)) return u;
    return null;
  }
  function fmt(text) {
    var out = esc(text);
    // [label](url)
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_, label, url) {
      var href = safeHref(url);
      if (!href) return label;
      var ext = /^https?:/i.test(href);
      return '<a href="' + href + '"' + (ext ? ' target="_blank" rel="noopener"' : "") + ">" + label + "</a>";
    });
    // bare mailto address
    out = out.replace(/(^|[\s(])([\w.+-]+@[\w-]+\.[\w.-]+)/g, function (_, pre, addr) {
      return pre + '<a href="mailto:' + addr + '">' + addr + "</a>";
    });
    // bare root-relative page links like /shop.html
    out = out.replace(/(^|[\s(])(\/[a-z0-9\-]+\.html)/gi, function (_, pre, path) {
      return pre + '<a href="' + path + '">' + path + "</a>";
    });
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    return out.replace(/\n/g, "<br>");
  }

  /* ---- icons ---- */
  function chatIcon() {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-8.9 8.4 9 9 0 0 1-4.1-1L3 20l1.1-4a8.4 8.4 0 0 1-1-4A8.38 8.38 0 0 1 12 3.6a8.38 8.38 0 0 1 9 7.9z"/></svg>';
  }
  function closeIcon() {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  }
  function sendIcon() {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
  }
})();

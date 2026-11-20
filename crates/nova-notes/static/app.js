// Nova Notes: the small amount of behaviour the server-rendered pages need.
(function () {
  "use strict";

  // Theme: follows the system unless the user picked one.
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem("nn-theme"); } catch (e) { /* private mode */ }
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

  var sidebarToggle = document.querySelector("[data-sidebar-toggle]");
  var menuToggle = document.querySelector("[data-menu-toggle]");
  function setSidebarCollapsed(collapsed) {
    root.classList.toggle("sidebar-collapsed", collapsed);
    sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
    var label = collapsed ? "Expand sidebar" : "Collapse sidebar";
    sidebarToggle.setAttribute("aria-label", label);
    sidebarToggle.title = label;
  }
  if (sidebarToggle) {
    setSidebarCollapsed(root.classList.contains("sidebar-collapsed"));
    sidebarToggle.addEventListener("click", function () {
      var collapsed = sidebarToggle.getAttribute("aria-expanded") === "true";
      setSidebarCollapsed(collapsed);
      try { localStorage.setItem("nn-sidebar-collapsed", String(collapsed)); } catch (e) { /* private mode */ }
    });
  }
  if (menuToggle) {
    menuToggle.addEventListener("click", function () {
      var open = document.body.classList.toggle("menu-open");
      menuToggle.setAttribute("aria-expanded", String(open));
      if (open) document.querySelector(".sidebar .brand").focus();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape" || !document.activeElement.closest(".sidebar") || !document.body.classList.contains("menu-open")) return;
      menuToggle.click();
      menuToggle.focus();
    });
  }

  function currentTheme() {
    return root.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }

  function toast(message, bad) {
    var holder = document.getElementById("toasts");
    if (!holder) return;
    var el = document.createElement("div");
    el.className = "toast" + (bad ? " bad" : "");
    el.setAttribute("role", bad ? "alert" : "status");
    el.textContent = message;
    holder.appendChild(el);
    setTimeout(function () { el.remove(); }, bad ? 9000 : 4000);
  }

  // Pull a readable message out of an error response (JSON detail or error page).
  function errorMessage(text, status) {
    try {
      var json = JSON.parse(text);
      if (json && typeof json.detail === "string") return json.detail;
    } catch (e) { /* not JSON */ }
    var doc = new DOMParser().parseFromString(text || "", "text/html");
    var detail = doc.querySelector("[data-error-detail]");
    if (detail && detail.textContent.trim()) return detail.textContent.trim();
    return "Request failed (" + status + ")";
  }

  document.addEventListener("click", function (event) {
    var el = event.target.closest("[data-theme-toggle],[data-open],[data-close],[data-copy]");
    if (!el) return;
    if (el.hasAttribute("data-theme-toggle")) {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("nn-theme", next); } catch (e) { /* ignore */ }
    } else if (el.hasAttribute("data-open")) {
      var dialog = document.querySelector(el.getAttribute("data-open"));
      if (dialog && dialog.showModal) dialog.showModal();
    } else if (el.hasAttribute("data-close")) {
      var open = el.closest("dialog");
      if (open) open.close();
    } else if (el.hasAttribute("data-copy")) {
      navigator.clipboard.writeText(el.getAttribute("data-copy")).then(function () { toast("Copied"); });
    }
  });

  // Clicking the backdrop closes a dialog.
  document.addEventListener("mousedown", function (event) {
    if (event.target.tagName === "DIALOG") event.target.close();
  });

  document.body.addEventListener("htmx:responseError", function (event) {
    toast(errorMessage(event.detail.xhr.responseText, event.detail.xhr.status), true);
  });
  document.body.addEventListener("htmx:sendError", function () {
    toast("Could not reach the server", true);
  });
  // Servers can ask for a toast with an HX-Trigger: {"toast": "..."} header.
  document.body.addEventListener("toast", function (event) {
    toast(typeof event.detail === "string" ? event.detail : event.detail.value);
  });
  // A dialog whose form succeeded closes itself.
  document.body.addEventListener("htmx:afterRequest", function (event) {
    if (!event.detail.successful) return;
    var form = event.detail.elt.closest && event.detail.elt.closest("form[data-close-on-success]");
    if (!form) return;
    var dialog = form.closest("dialog");
    if (dialog) dialog.close();
    if (form.hasAttribute("data-reset")) form.reset();
  });

  // POST JSON and read a server-sent-event reply. onEvent gets each parsed
  // `data:` payload; the promise resolves when the stream ends.
  function streamPost(url, body, onEvent) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify(body),
      credentials: "same-origin"
    }).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (text) { throw new Error(errorMessage(text, response.status)); });
      }
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      function handle(block) {
        var data = block.split("\n").filter(function (line) { return line.indexOf("data:") === 0; })
          .map(function (line) { return line.slice(5).replace(/^ /, ""); }).join("\n");
        if (!data) return;
        try { onEvent(JSON.parse(data)); } catch (e) { onEvent({ type: "text", content: data }); }
      }
      function pump() {
        return reader.read().then(function (result) {
          buffer += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
          var parts = buffer.split(/\r?\n\r?\n/);
          buffer = parts.pop();
          parts.forEach(handle);
          if (result.done) { if (buffer.trim()) handle(buffer); return; }
          return pump();
        });
      }
      return pump();
    });
  }

  window.nn = { toast: toast, streamPost: streamPost, errorMessage: errorMessage };
})();

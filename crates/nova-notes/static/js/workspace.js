// Notes workspace (/notes) and the command palette (Ctrl/Cmd+K, every page).
(function () {
  "use strict";

  var root = document.documentElement;
  var ws = document.getElementById("ws");

  function store(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private mode */ } }

  function localDate(d) {
    d = d || new Date();
    return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
  }

  // POST a small form the normal way, so the server's redirect is followed and
  // an open editor gets to save first (notebooks.js flushes [data-note-create]).
  function postForm(action, fields) {
    var form = document.createElement("form");
    form.method = "post";
    form.action = action;
    form.hidden = true;
    form.setAttribute("data-note-create", "");
    Object.keys(fields || {}).forEach(function (name) {
      var input = document.createElement("input");
      input.type = "hidden";
      input.name = name;
      input.value = fields[name];
      form.appendChild(input);
    });
    document.body.appendChild(form);
    form.requestSubmit();
  }

  function currentNotebook() {
    var params = new URLSearchParams(location.search);
    return location.pathname === "/notes" ? (params.get("nb") || "") : "";
  }
  function newNote(title) { postForm("/notes/new", { nb: currentNotebook(), title: title || "" }); }
  function todayNote() { postForm("/notes/daily", { date: localDate() }); }

  // ------------------------------------------------------------------
  // Command palette
  // ------------------------------------------------------------------
  var palette = null;
  var index = null;
  var indexAt = 0;
  var results = [];
  var selected = 0;

  var PAGES = [
    ["All Notes", "/notes"], ["Pinned notes", "/notes?pinned=1"], ["Notebooks", "/notebooks"],
    ["Ask and Search", "/search"], ["Podcasts", "/podcasts"], ["Sources", "/sources"], ["Models", "/settings/models"],
    ["Transformations", "/transformations"], ["Settings", "/settings"], ["Advanced", "/advanced"],
  ];

  function actions() {
    var list = [
      { kind: "+", label: "New note", meta: "Ctrl Alt N", run: function () { newNote(); } },
      { kind: "◷", label: "Open today's note", meta: "Ctrl Alt D", run: todayNote },
      { kind: "◐", label: "Toggle light / dark theme", run: function () {
        var button = document.querySelector("[data-theme-toggle]");
        if (button) button.click();
      } },
    ];
    if (ws) {
      list.push({ kind: "⤢", label: "Toggle focus mode", meta: "Ctrl .", run: function () { toggle("focus"); } });
      list.push({ kind: "▤", label: "Toggle sidebar", meta: "Ctrl \\", run: function () { toggle("sidebar"); } });
    }
    PAGES.forEach(function (page) {
      list.push({ kind: "→", label: "Go to " + page[0], href: page[1] });
    });
    return list;
  }

  // Fuzzy match: contiguous matches and word starts score best. Returns
  // null, or { score, hits: [indices] }.
  function fuzzy(query, text) {
    if (!query) return { score: 0, hits: [] };
    var q = query.toLowerCase();
    var t = text.toLowerCase();
    var at = t.indexOf(q);
    if (at !== -1) {
      var hits = [];
      for (var i = 0; i < q.length; i++) hits.push(at + i);
      return { score: 100 - at + (at === 0 || /\W/.test(t[at - 1]) ? 50 : 0) - t.length / 100, hits: hits };
    }
    var pos = 0, score = 0, found = [], prev = -2;
    for (var j = 0; j < q.length; j++) {
      if (q[j] === " ") continue;
      var k = t.indexOf(q[j], pos);
      if (k === -1) return null;
      score += k === prev + 1 ? 5 : (k === 0 || /\W/.test(t[k - 1]) ? 4 : 1);
      found.push(k);
      prev = k;
      pos = k + 1;
    }
    return { score: score - t.length / 100, hits: found };
  }

  function loadIndex(force) {
    if (index && !force && Date.now() - indexAt < 20000) return Promise.resolve(index);
    return fetch("/notes/palette", { credentials: "same-origin", headers: { Accept: "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("palette " + r.status); return r.json(); })
      .then(function (data) { index = data; indexAt = Date.now(); return data; });
  }

  function build() {
    palette = document.createElement("dialog");
    palette.className = "nn-palette";
    palette.setAttribute("aria-label", "Search notes and commands");
    palette.innerHTML =
      '<div class="nn-palette-input"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3.5 3.5"/></svg>' +
      '<input type="text" placeholder="Search notes, notebooks, #tags and commands" aria-label="Search" autocomplete="off" spellcheck="false" role="combobox" aria-expanded="true" aria-controls="nn-palette-list"></div>' +
      '<ul class="nn-palette-list" id="nn-palette-list" role="listbox"></ul>' +
      '<div class="nn-palette-foot"><span><kbd>↑</kbd> <kbd>↓</kbd> move</span><span><kbd>Enter</kbd> open</span><span><kbd>Esc</kbd> close</span></div>';
    document.body.appendChild(palette);
    var input = palette.querySelector("input");
    input.addEventListener("input", function () { render(input.value); });
    input.addEventListener("keydown", function (event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (!results.length) return;
        selected = (selected + (event.key === "ArrowDown" ? 1 : -1) + results.length) % results.length;
        highlight();
      } else if (event.key === "Enter") {
        event.preventDefault();
        choose(results[selected]);
      }
    });
    palette.addEventListener("click", function (event) {
      var item = event.target.closest(".nn-palette-item");
      if (item) choose(results[Number(item.dataset.index)]);
    });
    palette.addEventListener("mousemove", function (event) {
      var item = event.target.closest(".nn-palette-item");
      if (item && Number(item.dataset.index) !== selected) { selected = Number(item.dataset.index); highlight(); }
    });
  }

  function labelWithHits(text, hits) {
    var span = document.createElement("span");
    span.className = "nn-palette-label";
    var set = {};
    (hits || []).forEach(function (h) { set[h] = true; });
    var run = "", inMark = false;
    function flush() {
      if (!run) return;
      if (inMark) { var m = document.createElement("mark"); m.textContent = run; span.appendChild(m); }
      else span.appendChild(document.createTextNode(run));
      run = "";
    }
    for (var i = 0; i < text.length; i++) {
      var hit = !!set[i];
      if (hit !== inMark) { flush(); inMark = hit; }
      run += text[i];
    }
    flush();
    return span;
  }

  function render(query) {
    query = (query || "").trim();
    var list = palette.querySelector(".nn-palette-list");
    var sections = [];
    var data = index || { notes: [], notebooks: [], tags: [] };
    var tagQuery = query.charAt(0) === "#";

    function ranked(items, text) {
      return items.map(function (item) { var m = fuzzy(query, text(item)); return m && Object.assign({ match: m }, item); })
        .filter(Boolean)
        .sort(function (a, b) { return b.match.score - a.match.score; });
    }

    if (!query) {
      sections.push(["Recent notes", data.notes.slice(0, 7).map(function (n) {
        return { kind: n.pinned ? "★" : "¶", label: n.title, meta: n.notebook || "", href: "/notes?note=" + encodeURIComponent(n.id) };
      })]);
      sections.push(["Commands", actions().slice(0, ws ? 5 : 3)]);
    } else if (tagQuery) {
      var tq = query.slice(1);
      sections.push(["Tags", data.tags.map(function (t) { var m = fuzzy(tq, t); return m && { kind: "#", label: "#" + t, match: { score: m.score, hits: m.hits.map(function (h) { return h + 1; }) }, href: "/notes?tag=" + encodeURIComponent(t) }; })
        .filter(Boolean).sort(function (a, b) { return b.match.score - a.match.score; })]);
    } else {
      sections.push(["Notes", ranked(data.notes, function (n) { return n.title; }).slice(0, 12).map(function (n) {
        return { kind: n.pinned ? "★" : "¶", label: n.title, match: n.match, meta: n.notebook || "", href: "/notes?note=" + encodeURIComponent(n.id) };
      })]);
      sections.push(["Notebooks", ranked(data.notebooks, function (nb) { return nb.name; }).slice(0, 5).map(function (nb) {
        return { kind: "▢", label: nb.name, match: nb.match, meta: "notebook", href: "/notes?nb=" + encodeURIComponent(nb.id) };
      })]);
      sections.push(["Tags", data.tags.map(function (t) { var m = fuzzy(query, t); return m && { kind: "#", label: "#" + t, match: { score: m.score, hits: m.hits.map(function (h) { return h + 1; }) }, href: "/notes?tag=" + encodeURIComponent(t) }; })
        .filter(Boolean).sort(function (a, b) { return b.match.score - a.match.score; }).slice(0, 5)]);
      sections.push(["Commands", ranked(actions(), function (a) { return a.label; }).slice(0, 6)]);
      var exact = data.notes.some(function (n) { return n.title.toLowerCase() === query.toLowerCase(); });
      if (!exact) sections.push(["Create", [{ kind: "+", label: "Create note “" + query + "”", meta: "Enter", run: function () { newNote(query); } }]]);
    }

    results = [];
    list.replaceChildren();
    sections.forEach(function (section) {
      if (!section[1].length) return;
      var head = document.createElement("li");
      head.className = "nn-palette-section";
      head.setAttribute("role", "presentation");
      head.textContent = section[0];
      list.appendChild(head);
      section[1].forEach(function (item) {
        var li = document.createElement("li");
        li.className = "nn-palette-item";
        li.setAttribute("role", "option");
        li.dataset.index = String(results.length);
        var kind = document.createElement("span");
        kind.className = "nn-palette-kind";
        kind.textContent = item.kind || "•";
        li.appendChild(kind);
        li.appendChild(labelWithHits(item.label, item.match && item.match.hits));
        if (item.meta) {
          var meta = document.createElement("span");
          meta.className = "nn-palette-meta";
          meta.textContent = item.meta;
          li.appendChild(meta);
        }
        list.appendChild(li);
        results.push(item);
      });
    });
    if (!results.length) {
      var empty = document.createElement("li");
      empty.className = "nn-palette-empty";
      empty.textContent = index ? "Nothing matches." : "Loading…";
      list.appendChild(empty);
    }
    selected = 0;
    highlight();
  }

  function highlight() {
    palette.querySelectorAll(".nn-palette-item").forEach(function (el) {
      var on = Number(el.dataset.index) === selected;
      el.setAttribute("aria-selected", on ? "true" : "false");
      if (on) el.scrollIntoView({ block: "nearest" });
    });
  }

  function choose(item) {
    if (!item) return;
    palette.close();
    if (item.run) item.run();
    else if (item.href) {
      // A normal link click, so an open editor saves before leaving (notebooks.js).
      var a = document.createElement("a");
      a.href = item.href;
      a.hidden = true;
      document.body.appendChild(a);
      a.click();
    }
  }

  function openPalette() {
    if (!palette) build();
    if (palette.open) { palette.close(); return; }
    var input = palette.querySelector("input");
    input.value = "";
    render("");
    palette.showModal();
    input.focus();
    loadIndex(false).then(function () { if (palette.open) render(input.value); })
      .catch(function () { if (window.nn) window.nn.toast("Could not load the notes index", true); });
  }

  document.addEventListener("click", function (event) {
    if (event.target.closest && event.target.closest("[data-palette-open]")) openPalette();
  });

  // ------------------------------------------------------------------
  // Keyboard shortcuts
  // ------------------------------------------------------------------
  document.addEventListener("keydown", function (event) {
    var mod = event.ctrlKey || event.metaKey;
    if (!mod) return;
    var key = event.key.toLowerCase();
    if (key === "k" && !event.altKey && !event.shiftKey) { event.preventDefault(); openPalette(); return; }
    if (event.altKey && (event.code === "KeyN" || key === "n")) { event.preventDefault(); newNote(); return; }
    if (event.altKey && (event.code === "KeyD" || key === "d")) { event.preventDefault(); todayNote(); return; }
    if (!ws) return;
    if (event.key === "\\") { event.preventDefault(); toggle("sidebar"); }
    else if (event.key === ".") { event.preventDefault(); toggle("focus"); }
  });

  // ------------------------------------------------------------------
  // The workspace page
  // ------------------------------------------------------------------
  var narrow = window.matchMedia("(max-width: 820px)");

  // The sidebar is the app's own: app.js owns its collapse (desktop) and
  // drawer (narrow windows). Focus mode hides the sidebar and the list.
  function toggle(what) {
    if (what === "sidebar") {
      if (root.classList.contains("ws-focus")) { root.classList.remove("ws-focus"); store("nn-ws-focus", "off"); return; }
      var control = document.querySelector(narrow.matches ? "[data-menu-toggle]" : "[data-sidebar-toggle]");
      if (control) control.click();
    } else if (what === "focus") {
      var on = root.classList.toggle("ws-focus");
      store("nn-ws-focus", on ? "on" : "off");
    }
  }

  // Confirmations, and today's date for daily notes. Capture phase, so this
  // runs before notebooks.js saves an open editor and resubmits the form.
  document.addEventListener("submit", function (event) {
    var form = event.target;
    if (form.hasAttribute("data-ws-confirm") && !form._wsConfirmed) {
      if (!window.confirm(form.getAttribute("data-ws-confirm"))) {
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
      form._wsConfirmed = true;
    }
    if (form.hasAttribute("data-ws-daily")) form.querySelector('[name="date"]').value = localDate();
  }, true);

  // The notebook menu saves as soon as a box is ticked or unticked.
  document.addEventListener("change", function (event) {
    var box = event.target;
    if (box.type !== "checkbox" || !box.closest("form[data-nn-autosave]")) return;
    box.closest("form").requestSubmit();
  });

  // Ctrl+\ also collapses the sidebar on Studio pages.
  document.addEventListener("keydown", function (event) {
    if (ws || !(event.ctrlKey || event.metaKey) || event.key !== "\\") return;
    var control = document.querySelector("[data-sidebar-toggle]");
    if (control && !narrow.matches) { event.preventDefault(); control.click(); }
  });

  if (!ws) return;

  document.addEventListener("click", function (event) {
    var el = event.target.closest && event.target.closest("[data-ws-toggle], [data-ws-drawer-close]");
    if (!el) return;
    if (el.hasAttribute("data-ws-drawer-close")) {
      var menu = document.querySelector("[data-menu-toggle]");
      if (menu && document.body.classList.contains("menu-open")) menu.click();
    } else toggle(el.getAttribute("data-ws-toggle"));
  });

  // Drag a note from the list onto a notebook in the sidebar to file it there.
  var dragged = null;
  document.addEventListener("dragstart", function (event) {
    var link = event.target.closest && event.target.closest("[data-ws-drag]");
    if (!link) return;
    dragged = link.getAttribute("data-ws-drag");
    event.dataTransfer.effectAllowed = "copyMove";
    event.dataTransfer.setData("text/plain", link.querySelector("[data-ws-title]").textContent);
    root.classList.add("ws-dragging");
  });
  document.addEventListener("dragend", function () {
    dragged = null;
    root.classList.remove("ws-dragging");
    document.querySelectorAll(".ws-drop-over").forEach(function (el) { el.classList.remove("ws-drop-over"); });
  });
  function dropTarget(event) { return dragged && event.target.closest && event.target.closest("[data-ws-drop]"); }
  document.addEventListener("dragover", function (event) {
    var target = dropTarget(event);
    if (!target) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    target.classList.add("ws-drop-over");
  });
  document.addEventListener("dragleave", function (event) {
    var target = dropTarget(event);
    if (target && !target.contains(event.relatedTarget)) target.classList.remove("ws-drop-over");
  });
  document.addEventListener("drop", function (event) {
    var target = dropTarget(event);
    if (!target) return;
    event.preventDefault();
    var note = dragged;
    var body = new URLSearchParams({ nb: target.getAttribute("data-ws-drop") });
    fetch("/notes/" + encodeURIComponent(note) + "/file", { method: "POST", credentials: "same-origin", body: body })
      .then(function (response) {
        if (response.ok) return response.json();
        return response.text().then(function (text) { throw new Error(window.nn.errorMessage(text, response.status)); });
      })
      .then(function (result) {
        window.nn.toast("Filed in " + result.notebook);
        // Reload through a link click, so an open editor saves first.
        var a = document.createElement("a");
        a.href = location.pathname + location.search;
        a.hidden = true;
        document.body.appendChild(a);
        setTimeout(function () { a.click(); }, 600);
      })
      .catch(function (error) { window.nn.toast(error.message || "Could not file the note", true); });
  });

  // Dates the way Apple Notes shows them: time today, weekday this week, else the date.
  function dayStart(d) { return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime(); }
  var DAY = 86400000;
  function shortDate(d) {
    var diff = Math.round((dayStart(new Date()) - dayStart(d)) / DAY);
    if (diff === 0) return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    if (diff === 1) return "Yesterday";
    if (diff > 1 && diff < 7) return d.toLocaleDateString([], { weekday: "long" });
    return d.toLocaleDateString([], { day: "numeric", month: "short", year: d.getFullYear() === new Date().getFullYear() ? undefined : "numeric" });
  }
  function groupOf(d, pinned) {
    if (pinned) return "Pinned";
    var diff = Math.round((dayStart(new Date()) - dayStart(d)) / DAY);
    if (diff <= 0) return "Today";
    if (diff === 1) return "Yesterday";
    if (diff < 7) return "Previous 7 Days";
    if (diff < 30) return "Previous 30 Days";
    return d.toLocaleDateString([], { month: "long", year: "numeric" });
  }

  var list = ws.querySelector("[data-ws-notes]");

  function regroup() {
    if (!list) return;
    list.querySelectorAll(".ws-group-label").forEach(function (el) { el.remove(); });
    var last = null;
    list.querySelectorAll(".ws-note-item").forEach(function (li) {
      if (li.hidden) return;
      var d = new Date(li.dataset.updated);
      if (isNaN(d)) return;
      var date = li.querySelector("[data-ws-date]");
      if (date) date.textContent = shortDate(d);
      var group = groupOf(d, li.hasAttribute("data-pinned"));
      if (group !== last) {
        var label = document.createElement("li");
        label.className = "ws-group-label";
        label.setAttribute("role", "presentation");
        label.textContent = group;
        list.insertBefore(label, li);
        last = group;
      }
    });
  }

  ws.querySelectorAll(".ws-doc-foot [data-ws-date]").forEach(function (el) {
    var d = new Date(el.getAttribute("datetime"));
    if (!isNaN(d)) el.textContent = d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  });

  // Filter as you type; Enter or Down opens the first match.
  var filter = ws.querySelector("[data-ws-filter]");
  if (filter) {
    filter.addEventListener("input", function () {
      var words = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
      var shown = 0;
      list.querySelectorAll(".ws-note-item").forEach(function (li) {
        var hay = li.dataset.search || "";
        var ok = words.every(function (w) { return hay.indexOf(w) !== -1; });
        li.hidden = !ok;
        if (ok) shown++;
      });
      ws.querySelector("[data-ws-no-match]").hidden = shown > 0 || !words.length;
      regroup();
    });
    filter.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== "ArrowDown") return;
      var first = list.querySelector(".ws-note-item:not([hidden]) a");
      if (!first) return;
      event.preventDefault();
      if (event.key === "Enter") first.click(); else first.focus();
    });
  }

  // Arrow keys move through the list.
  if (list) {
    list.addEventListener("keydown", function (event) {
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      var links = Array.prototype.slice.call(list.querySelectorAll(".ws-note-item:not([hidden]) a"));
      var at = links.indexOf(document.activeElement);
      if (at === -1) return;
      event.preventDefault();
      var next = links[at + (event.key === "ArrowDown" ? 1 : -1)];
      if (next) next.focus(); else if (event.key === "ArrowUp" && filter) filter.focus();
    });
  }

  // Keep the list in step with the editor's saves: title, preview, date, order.
  function wordCount() {
    var surface = ws.querySelector(".nb-rich-surface");
    var out = ws.querySelector("[data-ws-words]");
    if (!surface || !out) return;
    var text = (surface.innerText || "").trim();
    var words = text ? text.split(/\s+/).length : 0;
    out.textContent = words + (words === 1 ? " word" : " words");
  }
  document.addEventListener("nn:note-saved", function (event) {
    var note = event.detail || {};
    wordCount();
    if (!list || !note.id) return;
    var li = list.querySelector('[data-note-id="' + CSS.escape(note.id) + '"]');
    if (!li) return;
    var content = (note.content || "").split("<!-- nova-notes:attachments -->")[0];
    var lines = content.split("\n").map(function (l) {
      return l.replace(/^[\s#>*+-]+/, "").replace(/^\[[ x]\]\s*/, "").replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/[*`]/g, "").trim();
    }).filter(Boolean);
    var title = (note.title || "").trim() || lines[0] || "New Note";
    li.querySelector("[data-ws-title]").textContent = title;
    var snippet = lines.filter(function (l) { return l !== title; }).join(" ").slice(0, 160);
    li.querySelector("[data-ws-snippet]").textContent = snippet || "No additional text";
    li.dataset.search = (title + " " + snippet).toLowerCase();
    if (note.updated) li.dataset.updated = note.updated;
    if (!li.hasAttribute("data-pinned")) {
      var firstUnpinned = list.querySelector(".ws-note-item:not([data-pinned])");
      if (firstUnpinned && firstUnpinned !== li) list.insertBefore(li, firstUnpinned);
    }
    regroup();
  });
  setTimeout(wordCount, 1500);

  regroup();
  var current = list && list.querySelector('[aria-current="true"]');
  if (current) current.scrollIntoView({ block: "nearest" });
})();

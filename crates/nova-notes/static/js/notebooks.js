// Notebooks: chat-context selects, bulk context actions, the add-source type
// picker, note file drops, the shared modal and the chat composer.
(function () {
  "use strict";

  // Context choices the user made by hand, so a pane that re-renders (a
  // source finishing processing, a note being added) keeps them.
  var chosen = {};

  function notifyContext() {
    document.body.dispatchEvent(new CustomEvent("nb-context", { bubbles: true }));
  }

  function restoreChoices(root) {
    (root || document).querySelectorAll("select.nb-ctx").forEach(function (select) {
      var value = chosen[select.name];
      if (value === undefined) return;
      var option = select.querySelector('option[value="' + value + '"]');
      if (option && !option.disabled) select.value = value;
    });
  }

  document.addEventListener("change", function (event) {
    var target = event.target;
    if (target.matches && target.matches("select.nb-ctx")) {
      chosen[target.name] = target.value;
      notifyContext();
    }
    if (target.matches && target.matches(".nb-upload .nb-file-input") && target.files.length) {
      var form = target.closest("form");
      if (window.htmx) window.htmx.trigger(form, "submit"); else form.submit();
    }
    if (target.matches && target.matches('.nb-type-picker input[name="type"]')) {
      showSourceType(target.closest("form"));
    }
  });

  // The add-source dialog shows only the inputs of the chosen type.
  function showSourceType(form) {
    if (!form) return;
    var checked = form.querySelector('.nb-type-picker input[name="type"]:checked');
    var type = checked ? checked.value : "link";
    form.querySelectorAll("[data-nb-type]").forEach(function (el) {
      el.hidden = el.getAttribute("data-nb-type") !== type;
    });
  }
  document.querySelectorAll(".nb-type-picker").forEach(function (picker) { showSourceType(picker.closest("form")); });
  document.addEventListener("reset", function (event) {
    var form = event.target;
    if (form.querySelector && form.querySelector(".nb-type-picker")) setTimeout(function () { showSourceType(form); }, 0);
  });

  document.addEventListener("click", function (event) {
    var bulk = event.target.closest("[data-nb-bulk]");
    if (bulk) {
      var kind = bulk.getAttribute("data-nb-bulk");
      var mode = bulk.getAttribute("data-mode");
      var prefix = kind === "note" ? "ctx_note:" : "ctx_source:";
      document.querySelectorAll("select.nb-ctx").forEach(function (select) {
        if (select.name.indexOf(prefix) !== 0) return;
        var option = select.querySelector('option[value="' + mode + '"]');
        // "Insights only" leaves sources without insights out (as the old app did).
        var value = option && !option.disabled ? mode : "off";
        select.value = value;
        chosen[select.name] = value;
      });
      var menu = bulk.closest("details");
      if (menu) menu.open = false;
      notifyContext();
      return;
    }
    // Close an open pop-over menu when clicking elsewhere.
    document.querySelectorAll("details.nb-menu[open]").forEach(function (menu) {
      if (!menu.contains(event.target)) menu.open = false;
    });
  });

  // Plain forms that need confirming.
  document.addEventListener("submit", function (event) {
    var message = event.target.getAttribute && event.target.getAttribute("data-nb-confirm");
    if (message && !window.confirm(message)) event.preventDefault();
  }, true);

  // The Transport dialog: filter its note and folder rows as you type.
  document.addEventListener("input", function (event) {
    var filter = event.target.closest && event.target.closest("[data-nb-transport-filter]");
    if (!filter) return;
    var dialog = filter.closest("[data-nb-transport]");
    var words = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
    var shown = 0;
    dialog.querySelectorAll(".nb-transport-row").forEach(function (row) {
      var hay = row.getAttribute("data-search") || "";
      var ok = words.every(function (w) { return hay.indexOf(w) !== -1; });
      row.hidden = !ok;
      if (ok) shown++;
    });
    var empty = dialog.querySelector("[data-nb-transport-empty]");
    if (empty) empty.hidden = shown > 0;
  });

  // Content loaded into the shared modal opens it.
  document.body.addEventListener("htmx:afterSwap", function (event) {
    var target = event.detail.target;
    if (target && target.id === "nb-modal-body") {
      var dialog = target.closest("dialog");
      if (dialog && !dialog.open && dialog.showModal) dialog.showModal();
    }
  });
  document.body.addEventListener("htmx:afterSettle", function (event) {
    restoreChoices(event.detail.target);
    var log = document.querySelector(".nb-chat-log");
    if (log && event.detail.target && event.detail.target.id === "nb-chat") log.scrollTop = log.scrollHeight;
  });
  // Expand fills the window with the modal; closing it shrinks it back.
  function setExpanded(dialog, on) {
    dialog.classList.toggle("nb-full", on);
    var button = dialog.querySelector("[data-nb-expand]");
    if (button) {
      button.textContent = on ? "Shrink" : "Expand";
      button.setAttribute("aria-pressed", on ? "true" : "false");
    }
  }
  document.addEventListener("click", function (event) {
    var button = event.target.closest && event.target.closest("[data-nb-expand]");
    if (!button) return;
    var dialog = button.closest("dialog");
    if (dialog) setExpanded(dialog, !dialog.classList.contains("nb-full"));
  });
  var modal = document.getElementById("nb-modal");
  if (modal) modal.addEventListener("close", function () { setExpanded(modal, false); });

  // A modal form that succeeded closes the modal (the shared handler only
  // knows about forms already in the page).
  document.body.addEventListener("htmx:afterRequest", function (event) {
    if (!event.detail.successful) return;
    var elt = event.detail.elt;
    var form = elt.closest && elt.closest("#nb-modal-body form[data-close-on-success]");
    if (form) { var dialog = form.closest("dialog"); if (dialog) dialog.close(); }
  });

  // Dropping files onto the notes pane adds them as notes.
  function dropzone(event) { return event.target.closest && event.target.closest("[data-nb-dropzone]"); }
  function hasFiles(event) { return event.dataTransfer && Array.prototype.indexOf.call(event.dataTransfer.types, "Files") !== -1; }
  document.addEventListener("dragover", function (event) {
    var zone = dropzone(event);
    if (!zone || !hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    zone.classList.add("nb-dragging");
  });
  document.addEventListener("dragleave", function (event) {
    var zone = dropzone(event);
    if (zone && !zone.contains(event.relatedTarget)) zone.classList.remove("nb-dragging");
  });
  document.addEventListener("drop", function (event) {
    var zone = dropzone(event);
    if (!zone || !hasFiles(event)) return;
    event.preventDefault();
    zone.classList.remove("nb-dragging");
    var input = zone.querySelector(".nb-upload .nb-file-input");
    if (!input) return;
    input.files = event.dataTransfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });

  // Chat composer: Enter sends, Shift+Enter is a new line; the question shows
  // at once while the answer is generated.
  document.addEventListener("keydown", function (event) {
    if (!event.target.matches || !event.target.matches("textarea.nb-chat-input")) return;
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      var form = event.target.closest("form");
      if (form.reportValidity()) form.requestSubmit();
    }
  });
  document.body.addEventListener("htmx:beforeRequest", function (event) {
    if (event.detail.elt.id !== "nb-chat-form") return;
    var text = event.detail.elt.querySelector("textarea.nb-chat-input").value.trim();
    var log = document.querySelector(".nb-chat-log");
    if (!log || !text) return;
    var empty = log.querySelector(".nb-chat-empty");
    if (empty) empty.remove();
    var human = document.createElement("div");
    human.className = "msg human nb-pending";
    var p = document.createElement("p");
    p.className = "pre-wrap";
    p.textContent = text;
    human.appendChild(p);
    var thinking = document.createElement("div");
    thinking.className = "msg ai nb-pending";
    thinking.innerHTML = '<span class="spinner"></span>';
    log.appendChild(human);
    log.appendChild(thinking);
    log.scrollTop = log.scrollHeight;
  });
  document.body.addEventListener("htmx:afterRequest", function (event) {
    if (event.detail.elt.id === "nb-chat-form" && !event.detail.successful) {
      document.querySelectorAll(".nb-chat-log .nb-pending").forEach(function (el) { el.remove(); });
    }
  });
})();

// Dragging a notebook card onto a folder card moves it into that folder,
// through the same form post as the card menu's "Move to folder".
(function () {
  "use strict";
  var MIME = "application/x-nova-notebook";
  function folderOf(event) { return event.target.closest && event.target.closest("[data-nb-drop-folder]"); }
  function carriesNotebook(event) {
    return event.dataTransfer && Array.prototype.indexOf.call(event.dataTransfer.types, MIME) !== -1;
  }
  document.addEventListener("dragstart", function (event) {
    var card = event.target.closest && event.target.closest("[data-nb-drag]");
    if (!card) return;
    event.dataTransfer.setData(MIME, card.getAttribute("data-nb-drag"));
    event.dataTransfer.effectAllowed = "move";
    card.classList.add("nb-dragging-card");
  });
  document.addEventListener("dragend", function (event) {
    var card = event.target.closest && event.target.closest("[data-nb-drag]");
    if (card) card.classList.remove("nb-dragging-card");
  });
  document.addEventListener("dragover", function (event) {
    var folder = folderOf(event);
    if (!folder || !carriesNotebook(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    folder.classList.add("nb-drop-target");
  });
  document.addEventListener("dragleave", function (event) {
    var folder = folderOf(event);
    if (folder && !folder.contains(event.relatedTarget)) folder.classList.remove("nb-drop-target");
  });
  document.addEventListener("drop", function (event) {
    var folder = folderOf(event);
    if (!folder || !carriesNotebook(event)) return;
    event.preventDefault();
    folder.classList.remove("nb-drop-target");
    var notebookId = event.dataTransfer.getData(MIME);
    if (!notebookId) return;
    var form = document.createElement("form");
    form.method = "post";
    form.action = "/notebooks/" + encodeURIComponent(notebookId) + "/move";
    [["folder_id", folder.getAttribute("data-nb-drop-folder")], ["back", folder.getAttribute("data-nb-back") || "/notebooks"]]
      .forEach(function (pair) {
        var input = document.createElement("input");
        input.type = "hidden";
        input.name = pair[0];
        input.value = pair[1];
        form.appendChild(input);
      });
    document.body.appendChild(form);
    form.submit();
  });
})();

// The note is already filed when this page opens. Save snapshots in order,
// including any changes made while a save is in progress.
(function () {
  "use strict";

  function requestJson(url, options) {
    return fetch(url, options).then(function (response) {
      if (response.ok) return response.json();
      return response.text().then(function (text) {
        throw new Error(window.nn.errorMessage(text, response.status));
      });
    });
  }

  function postJson(url, body) {
    return requestJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  function noteHref(root, noteId) {
    // Inside the notes workspace, linked notes open in place there.
    var host = root.closest("[data-note-href]");
    if (host) return host.dataset.noteHref.replace("__ID__", encodeURIComponent(noteId));
    return "/notebooks/" + encodeURIComponent(root.dataset.notebook) + "/notes/" + encodeURIComponent(noteId);
  }

  function setStatus(root, text, failed) {
    root.querySelector("[data-nb-status]").textContent = text;
    root.querySelector("[data-nb-retry]").hidden = !failed;
  }

  function showMode(root, mode) {
    root.querySelectorAll("[data-nb-mode]").forEach(function (button) {
      button.disabled = false;
      button.setAttribute("aria-pressed", String(button.dataset.nbMode === mode));
    });
  }

  function showBacklinks(root, backlinks) {
    if (!backlinks.length) return;
    var holder = root.querySelector(".nb-backlinks");
    var label = document.createElement("span");
    label.className = "faint small";
    label.textContent = "Linked from";
    holder.appendChild(label);
    backlinks.forEach(function (link) {
      var anchor = document.createElement("a");
      anchor.className = "badge nb-backlink";
      anchor.href = noteHref(root, link.id);
      anchor.textContent = link.title || "Untitled Note";
      holder.appendChild(anchor);
    });
    holder.hidden = false;
  }

  function pending(root) {
    var state = root._nb;
    return state && (state.dirtySince !== null || state.saving);
  }

  function queueSave(root) {
    var state = root._nb;
    if (!state) return;
    // BlockSuite only notifies once per dirty period. Re-arm it so every
    // document edit can restart the debounce, even during a save.
    state.handle.markClean();
    var now = performance.now();
    state.dirtySince ??= now;
    if (state.saving) return;
    if (state.timer === null) setStatus(root, "Saving…");
    clearTimeout(state.timer);
    // Save on a pause, with a time limit for uninterrupted typing.
    var delay = Math.min(1000, Math.max(0, 10000 - (now - state.dirtySince)));
    state.timer = setTimeout(function () { save(root); }, delay);
  }

  function save(root) {
    var state = root._nb;
    if (!state) return Promise.resolve(true);
    if (state.saving) return state.saving;
    if (state.dirtySince === null) return Promise.resolve(true);
    clearTimeout(state.timer);
    state.timer = null;
    state.dirtySince = null;
    setStatus(root, "Saving…");
    var title = state.title.value.trim();
    state.saving = state.handle.snapshot(title)
      .then(function (body) { return postJson(root.dataset.editor, body); })
      .then(function (note) {
        document.title = (note.title || "Untitled Note") + " · Nova Notes";
        root.dispatchEvent(new CustomEvent("nn:note-saved", { bubbles: true, detail: note }));
        state.saving = null;
        if (state.dirtySince !== null) queueSave(root);
        else setStatus(root, "Saved");
        return true;
      })
      .catch(function (error) {
        state.saving = null;
        state.dirtySince ??= performance.now();
        setStatus(root, error.message || "Could not save the note", true);
        return false;
      });
    return state.saving;
  }

  async function flush(root) {
    while (pending(root)) {
      if (!await save(root)) {
        root.querySelector("[data-nb-retry]").focus();
        return false;
      }
    }
    return true;
  }

  async function mount(root) {
    var surface = root.querySelector(".nb-rich-surface");
    var title = root.querySelector(".nb-rich-title");
    var css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = root.dataset.css;
    document.head.appendChild(css);
    try {
      var [editor, data] = await Promise.all([import(root.dataset.bundle), requestJson(root.dataset.editor)]);
      var handle = await editor.mount(surface, {
        docId: data.note.id,
        state: data.doc_state,
        content: data.note.content || "",
        mode: data.doc_mode,
        notes: data.notes,
        blobBase: "/note-blobs",
        onOpenNote: async function (id) {
          if (await flush(root)) window.location.href = noteHref(root, id);
        },
        createNote: function (name) { return postJson(root.dataset.link, { title: name }); },
        onChange: function () { queueSave(root); },
      });
      handle.setTitle(title.value.trim());
      root._nb = { handle: handle, title: title, dirtySince: null, timer: null, saving: null };
      showMode(root, handle.mode);
      showBacklinks(root, data.backlinks);
      title.disabled = false;
      if (root.hasAttribute("data-nb-fresh")) {
        title.focus({ preventScroll: true });
        // Drop only the "new" flag, so a reload reopens the same note.
        var url = new URL(location.href);
        url.searchParams.delete("new");
        history.replaceState(history.state, "", url.pathname + url.search);
      }
    } catch (error) {
      console.error(error);
      var message = document.createElement("div");
      message.className = "flash bad";
      message.setAttribute("role", "alert");
      message.textContent = "The editor could not be loaded: " + error.message;
      surface.replaceChildren(message);
    }
  }

  var root = document.querySelector("[data-nb-rich]");
  if (!root) return;
  mount(root);
  root.addEventListener("input", function (event) {
    if (event.target !== root._nb?.title) return;
    root._nb.handle.setTitle(event.target.value.trim());
    queueSave(root);
  });
  root.addEventListener("click", function (event) {
    if (event.target.closest("[data-nb-retry]")) { save(root); return; }
    var mode = event.target.closest("[data-nb-mode]")?.dataset.nbMode;
    if (mode && root._nb && mode !== root._nb.handle.mode) {
      root._nb.handle.setMode(mode);
      showMode(root, mode);
      queueSave(root);
    }
  });

  document.addEventListener("keydown", function (event) {
    if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== "s") return;
    event.preventDefault();
    save(root);
  });
  document.addEventListener("click", async function (event) {
    if (!pending(root) || event.defaultPrevented || event.button !== 0 ||
        event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
    var link = event.target.closest("a[href]");
    if (!link || link.target || link.hasAttribute("download") || link.getAttribute("href").startsWith("#")) return;
    event.preventDefault();
    if (await flush(root)) window.location.href = link.href;
  });
  document.addEventListener("submit", async function (event) {
    if (!pending(root) || event.defaultPrevented || !event.target.matches("[data-note-create], .sidebar-foot form")) return;
    event.preventDefault();
    if (await flush(root)) event.target.requestSubmit(event.submitter || undefined);
  });
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden" && pending(root)) save(root);
  });
  window.addEventListener("online", function () { if (pending(root)) save(root); });
  window.addEventListener("beforeunload", function (event) {
    if (pending(root)) { event.preventDefault(); event.returnValue = ""; }
  });
})();


// Pane layout. Sources, Notes and Chat each fold into a rail; Notes takes the free space (see notebooks.css).
// The layout is remembered across notebooks and visits under `nn-nb-cols`. A small inline script on the page
// restores it before first paint. In a narrow window the toolbar's switch picks one pane at a time instead.
(function () {
  "use strict";
  var KEY = "nn-nb-cols";
  var PANES = ["sources", "notes", "chat"];
  var narrow = window.matchMedia("(max-width: 900px)");

  function cols() { return document.getElementById("nb-cols"); }

  function save(root) {
    var state = {};
    PANES.forEach(function (pane) { state[pane] = root.getAttribute("data-" + pane); });
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* private mode */ }
  }

  // Keep every toggle's pressed state and label in step with the layout.
  function sync(root) {
    document.querySelectorAll("[data-nb-toggle]").forEach(function (el) {
      var pane = el.getAttribute("data-nb-toggle");
      var open = narrow.matches ? root.getAttribute("data-tab") === pane : root.getAttribute("data-" + pane) === "open";
      if (el.hasAttribute("aria-pressed")) el.setAttribute("aria-pressed", open ? "true" : "false");
    });
  }

  function toggle(root, pane, origin) {
    if (narrow.matches) {
      root.setAttribute("data-tab", pane);
    } else {
      var open = root.getAttribute("data-" + pane) === "open";
      // Never fold the last open pane away: that would leave three rails and an empty page.
      if (open && PANES.every(function (other) { return other === pane || root.getAttribute("data-" + other) !== "open"; })) return;
      root.setAttribute("data-" + pane, open ? "closed" : "open");
      save(root);
      // Keep keyboard focus somewhere that still exists.
      var col = root.querySelector('[data-col="' + pane + '"]');
      var next = col && col.querySelector(open ? ".nb-rail" : ".nb-collapse");
      if (next && origin !== "toolbar") next.focus();
    }
    sync(root);
  }

  document.addEventListener("click", function (event) {
    var el = event.target.closest && event.target.closest("[data-nb-toggle]");
    var root = cols();
    if (!el || !root) return;
    toggle(root, el.getAttribute("data-nb-toggle"), el.getAttribute("data-nb-from"));
    syncExpand(root);
  });

  // A note open as a page can expand over the other panes: Sources and Chat
  // fold into their rails, and the same button brings them back as they were.
  // This does not change the saved layout.
  var before = null;
  function expanded(root) {
    return root.getAttribute("data-sources") !== "open" && root.getAttribute("data-chat") !== "open";
  }
  function syncExpand(root) {
    document.querySelectorAll("[data-nb-expand-page]").forEach(function (button) {
      button.setAttribute("aria-pressed", expanded(root) ? "true" : "false");
      button.textContent = expanded(root) ? "Collapse" : "Expand";
    });
  }
  document.addEventListener("click", function (event) {
    var button = event.target.closest && event.target.closest("[data-nb-expand-page]");
    var root = cols();
    if (!button || !root) return;
    if (narrow.matches) return;
    if (!expanded(root)) {
      before = { sources: root.getAttribute("data-sources"), chat: root.getAttribute("data-chat") };
      root.setAttribute("data-sources", "closed");
      root.setAttribute("data-chat", "closed");
      root.setAttribute("data-notes", "open");
    } else {
      var back = before || { sources: "closed", chat: "open" };
      root.setAttribute("data-sources", back.sources);
      root.setAttribute("data-chat", back.chat);
      before = null;
    }
    sync(root);
    syncExpand(root);
  });

  function init() {
    var root = cols();
    if (!root) return;
    sync(root);
    syncExpand(root);
    if (narrow.addEventListener) narrow.addEventListener("change", function () { sync(root); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();

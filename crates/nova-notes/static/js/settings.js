// Behaviour for the settings pages (Models, Settings, Advanced).
(function () {
  "use strict";

  var LOADING = '<div class="dialog-body"><span class="spinner"></span></div>';

  document.addEventListener("click", function (event) {
    var el = event.target.closest("[data-set-dialog],[data-set-reveal],[data-set-toggle-all]");
    if (!el) return;
    if (el.hasAttribute("data-set-dialog")) {
      // Clear the shared dialog before htmx loads the new contents into it.
      var content = document.getElementById("set-dialog-content");
      if (content) content.innerHTML = LOADING;
    } else if (el.hasAttribute("data-set-reveal")) {
      var input = document.querySelector(el.getAttribute("data-set-reveal"));
      if (!input) return;
      var hidden = input.type === "password";
      input.type = hidden ? "text" : "password";
      el.textContent = hidden ? "Hide" : "Show";
    } else if (el.hasAttribute("data-set-toggle-all")) {
      var form = el.closest("form");
      if (!form) return;
      var boxes = Array.prototype.filter.call(
        form.querySelectorAll(".set-discover-item"),
        function (item) { return !item.hidden; }
      ).map(function (item) { return item.querySelector("input[type=checkbox]"); });
      var allOn = boxes.every(function (box) { return box.checked; });
      boxes.forEach(function (box) { box.checked = !allOn; });
      el.textContent = allOn ? "Select all" : "Clear selection";
    }
  });

  // Filter the discovered-model list as the user types.
  document.addEventListener("input", function (event) {
    var input = event.target.closest("[data-set-filter]");
    if (!input) return;
    var query = input.value.trim().toLowerCase();
    var form = input.closest("form");
    if (!form) return;
    form.querySelectorAll(".set-discover-item").forEach(function (item) {
      item.hidden = query !== "" && (item.getAttribute("data-name") || "").indexOf(query) === -1;
    });
  });

  // Changing the embedding model invalidates existing embeddings: confirm
  // first (capture phase, so a cancel never reaches htmx's change trigger).
  document.addEventListener("change", function (event) {
    var select = event.target.closest && event.target.closest("select[data-set-embedding]");
    if (!select) return;
    var current = select.getAttribute("data-current") || "";
    if (!current || select.value === current) return;
    var ok = window.confirm(
      "Changing your embedding model requires rebuilding all existing embeddings. " +
      "Existing embeddings stay unchanged until you rebuild, so searches may return incomplete results.\n\n" +
      "Change the embedding model now?"
    );
    if (!ok) {
      select.value = current;
      event.stopPropagation();
      event.preventDefault();
    }
  }, true);
})();

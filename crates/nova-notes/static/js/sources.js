// Behaviour for the sources pages.
(function () {
  "use strict";

  // Add-source dialog: show only the inputs of the chosen type.
  function syncType(form) {
    var checked = form.querySelector("input[name=type]:checked");
    form.setAttribute("data-src-type", checked ? checked.value : "link");
  }
  function initForms(root) {
    (root || document).querySelectorAll(".src-add-form").forEach(syncType);
  }
  document.addEventListener("change", function (event) {
    if (event.target.name === "type" && event.target.closest(".src-add-form")) {
      syncType(event.target.closest(".src-add-form"));
    }
  });
  document.addEventListener("DOMContentLoaded", function () { initForms(); });

  // Source chat: Enter sends, Shift+Enter makes a new line.
  document.addEventListener("keydown", function (event) {
    var area = event.target;
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    if (!area.matches || !area.matches(".src-chat-form textarea")) return;
    event.preventDefault();
    if (area.value.trim()) area.form.requestSubmit();
  });

  // Show the question at once while the answer is being written.
  document.addEventListener("htmx:beforeRequest", function (event) {
    var form = event.detail.elt;
    if (!form.matches || !form.matches(".src-chat-form")) return;
    var area = form.querySelector("textarea");
    var log = form.closest(".src-chat") && form.closest(".src-chat").querySelector(".chat-log");
    if (!area || !log || !area.value.trim()) return;
    var empty = log.querySelector(".empty");
    if (empty) empty.remove();
    var bubble = document.createElement("div");
    bubble.className = "msg human pre-wrap src-pending";
    bubble.textContent = area.value;
    var thinking = document.createElement("div");
    thinking.className = "msg ai src-pending";
    thinking.innerHTML = '<span class="spinner" aria-hidden="true"></span> Thinking…';
    log.appendChild(bubble);
    log.appendChild(thinking);
    log.scrollTop = log.scrollHeight;
  });
  document.addEventListener("htmx:afterRequest", function (event) {
    var form = event.detail.elt;
    if (!form.matches || !form.matches(".src-chat-form") || event.detail.successful) return;
    var panel = form.closest(".src-chat");
    if (panel) panel.querySelectorAll(".src-pending").forEach(function (el) { el.remove(); });
  });

  // Keep the newest chat message in view.
  function scrollChat(root) {
    (root || document).querySelectorAll(".src-chat .chat-log").forEach(function (log) { log.scrollTop = log.scrollHeight; });
  }
  document.addEventListener("DOMContentLoaded", function () { scrollChat(); });
  document.addEventListener("htmx:afterSettle", function (event) {
    initForms(event.target);
    scrollChat(event.target.parentNode || document);
  });
})();

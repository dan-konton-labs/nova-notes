// Behaviour for the podcasts pages: dialog links, confirmations, the
// script drop zone, the "uses speaker profile" hint, artwork checks, and
// keeping the Completed group open across list refreshes.
(function () {
  "use strict";

  var SCRIPT_FILE = /\.(txt|md|markdown)$/i;
  var ARTWORK_TYPES = ["image/png", "image/jpeg"];
  var ARTWORK_MAX_BYTES = 10 * 1024 * 1024;
  var completedOpen = false;

  function toast(message, bad) {
    if (window.nn && window.nn.toast) window.nn.toast(message, bad);
  }

  // Links that also open a dialog (they work as plain links without JS).
  document.addEventListener("click", function (event) {
    var link = event.target.closest("a.pod-dialog-link[data-open]");
    if (link && document.querySelector(link.getAttribute("data-open"))) event.preventDefault();
  });

  // Plain (non-htmx) forms that need a confirmation.
  document.addEventListener("submit", function (event) {
    var form = event.target.closest("form[data-pod-confirm]");
    if (form && !window.confirm(form.getAttribute("data-pod-confirm"))) event.preventDefault();
  });

  // The feed URL selects itself on focus, ready to copy by hand.
  document.addEventListener("focusin", function (event) {
    if (event.target.matches && event.target.matches("input[data-pod-select]")) event.target.select();
  });

  // Remember whether the Completed group is open; the list is re-rendered by htmx.
  document.addEventListener("toggle", function (event) {
    if (event.target.id === "pod-completed") completedOpen = event.target.open;
  }, true);
  document.addEventListener("htmx:load", function () {
    var completed = document.getElementById("pod-completed");
    if (completed && completedOpen && !completed.open) completed.open = true;
  });

  // Episode profile → "Uses speaker profile ...".
  document.addEventListener("change", function (event) {
    var select = event.target;
    if (select.matches("select[data-pod-profile]")) {
      var hint = select.closest(".field").querySelector("[data-pod-uses-speaker]");
      var option = select.options[select.selectedIndex];
      if (!hint) return;
      if (!option || !option.value) hint.textContent = "";
      else if (option.getAttribute("data-speaker")) hint.textContent = "Uses speaker profile " + option.getAttribute("data-speaker");
      else hint.textContent = "This episode profile references a speaker profile that no longer exists. Edit the profile and select a speaker profile.";
    } else if (select.matches("input[data-pod-script-file]")) {
      loadScript(select.closest("[data-pod-drop]"), select.files && select.files[0]);
    } else if (select.matches("input[data-pod-artwork]")) {
      var file = select.files && select.files[0];
      if (!file) return;
      if (ARTWORK_TYPES.indexOf(file.type) < 0) {
        toast("Artwork must be a PNG or JPEG image.", true);
        select.value = "";
      } else if (file.size > ARTWORK_MAX_BYTES) {
        toast("Artwork must be 10 MB or smaller.", true);
        select.value = "";
      }
    }
  });

  function updateCount(zone) {
    var area = zone.querySelector("textarea[name=script]");
    var counter = zone.querySelector("[data-pod-chars]");
    if (!area || !counter) return;
    var max = parseInt(counter.getAttribute("data-max"), 10) || 0;
    counter.textContent = area.value.length.toLocaleString() + " characters";
    counter.classList.toggle("pod-too-long", max > 0 && area.value.length > max);
  }

  function loadScript(zone, file) {
    if (!zone || !file) return;
    if (!SCRIPT_FILE.test(file.name) && file.type !== "text/plain" && file.type !== "text/markdown") {
      toast("Only .txt and .md script files are supported", true);
      return;
    }
    file.text().then(function (text) {
      var area = zone.querySelector("textarea[name=script]");
      if (area) area.value = text;
      var form = zone.closest("form");
      var name = form && form.querySelector("input[name=episode_name]");
      if (name && !name.value) name.value = file.name.replace(SCRIPT_FILE, "");
      var label = zone.querySelector("[data-pod-filename]");
      if (label) label.textContent = file.name;
      updateCount(zone);
    });
  }

  document.addEventListener("input", function (event) {
    var zone = event.target.closest && event.target.closest("[data-pod-drop]");
    if (zone && event.target.name === "script") updateCount(zone);
  });

  document.addEventListener("dragover", function (event) {
    var zone = event.target.closest && event.target.closest("[data-pod-drop]");
    if (!zone) return;
    event.preventDefault();
    zone.classList.add("pod-dragging");
  });
  document.addEventListener("dragleave", function (event) {
    var zone = event.target.closest && event.target.closest("[data-pod-drop]");
    if (zone && !zone.contains(event.relatedTarget)) zone.classList.remove("pod-dragging");
  });
  document.addEventListener("drop", function (event) {
    var zone = event.target.closest && event.target.closest("[data-pod-drop]");
    if (!zone) return;
    event.preventDefault();
    zone.classList.remove("pod-dragging");
    loadScript(zone, event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]);
  });
})();

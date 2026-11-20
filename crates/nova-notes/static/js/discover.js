// Discover: open the dialog in the chosen mode and show rejected forms in place.
(function () {
  "use strict";

  var TITLES = { notebook: "Discover a notebook", podcast: "Discover a notebook and podcast" };

  function syncTitle(form) {
    var dialog = form && form.closest("dialog");
    var title = dialog && dialog.querySelector("#disc-title");
    var checked = form && form.querySelector('input[name="mode"]:checked');
    if (title && checked) title.textContent = TITLES[checked.value] || TITLES.notebook;
  }

  // The header buttons pick the mode before app.js opens the dialog.
  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-disc-mode]");
    if (!button) return;
    var form = document.getElementById("disc-form");
    if (!form) return;
    var radio = form.querySelector('input[name="mode"][value="' + button.getAttribute("data-disc-mode") + '"]');
    if (radio) radio.checked = true;
    syncTitle(form);
    var prompt = form.querySelector("#disc-prompt");
    if (prompt) setTimeout(function () { prompt.focus(); }, 0);
  });

  document.addEventListener("change", function (event) {
    if (event.target.name === "mode" && event.target.closest(".disc-form")) syncTitle(event.target.closest(".disc-form"));
  });

  // A 4xx answer to the form is the form again, with the reason: swap it in
  // instead of showing an error toast.
  document.body.addEventListener("htmx:beforeSwap", function (event) {
    var elt = event.detail.elt;
    var status = event.detail.xhr.status;
    if (elt && elt.classList && elt.classList.contains("disc-form") && status >= 400 && status < 500) {
      event.detail.shouldSwap = true;
      event.detail.isError = false;
    }
  });
  document.body.addEventListener("htmx:afterSwap", function () {
    var form = document.getElementById("disc-form");
    if (form) syncTitle(form);
  });
})();

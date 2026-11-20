// Ask: stream the three stages into the page instead of waiting for the
// whole answer. Without JavaScript the form posts to /search/ask, which
// renders every stage at once. Event HTML is rendered and sanitised by the
// server; this script only places it.
(function () {
  "use strict";

  function slot(out, name) { return out.querySelector('[data-sr-slot="' + name + '"]'); }

  function place(el, html) {
    el.innerHTML = html;
    if (window.htmx) window.htmx.process(el);
  }

  function reset(out) {
    ["error", "strategy", "final"].forEach(function (name) { slot(out, name).innerHTML = ""; });
    var answers = slot(out, "answers");
    answers.hidden = true;
    answers.querySelector("[data-sr-answer-list]").innerHTML = "";
    answers.querySelector("[data-sr-answer-count]").textContent = "0";
  }

  function askBody(form) {
    var data = new FormData(form);
    var body = {
      question: String(data.get("question") || ""),
      strategy_model: String(data.get("strategy_model") || ""),
      answer_model: String(data.get("answer_model") || ""),
      final_answer_model: String(data.get("final_answer_model") || "")
    };
    var ids = data.getAll("notebook_ids").map(String).filter(Boolean);
    if (ids.length) body.notebook_ids = ids;
    return body;
  }

  function run(form) {
    var out = document.getElementById("sr-ask-out");
    if (!out || !window.nn || form.dataset.srBusy) return;
    var button = form.querySelector("[data-sr-ask-button]");
    var question = form.querySelector('[name="question"]');
    var progress = out.querySelector("[data-sr-progress]");
    var finished = false;
    reset(out);
    form.dataset.srBusy = "1";
    out.setAttribute("aria-busy", "true");
    progress.hidden = false;
    if (button) { button.disabled = true; button.textContent = "Processing..."; }
    if (question) question.readOnly = true;

    function done() {
      delete form.dataset.srBusy;
      out.removeAttribute("aria-busy");
      progress.hidden = true;
      if (button) { button.disabled = false; button.textContent = "Ask"; }
      if (question) question.readOnly = false;
    }

    function onEvent(event) {
      if (event.type === "strategy") {
        place(slot(out, "strategy"), event.html);
      } else if (event.type === "answer") {
        var answers = slot(out, "answers");
        var holder = document.createElement("div");
        holder.innerHTML = event.html;
        while (holder.firstChild) answers.querySelector("[data-sr-answer-list]").appendChild(holder.firstChild);
        var count = answers.querySelector("[data-sr-answer-count]");
        count.textContent = String(Number(count.textContent || "0") + 1);
        answers.hidden = false;
      } else if (event.type === "final_answer") {
        place(slot(out, "final"), event.html);
        progress.hidden = true;
      } else if (event.type === "error") {
        finished = true;
        place(slot(out, "error"), event.html || "");
        window.nn.toast(event.message || "Ask failed", true);
      } else if (event.type === "complete") {
        finished = true;
      }
    }

    window.nn.streamPost("/search/ask/stream", askBody(form), onEvent)
      .then(function () {
        if (!finished) window.nn.toast("The answer stream ended early. Try again.", true);
      })
      .catch(function (error) { window.nn.toast(error.message || "Ask failed", true); })
      .then(done);
  }

  document.addEventListener("submit", function (event) {
    var form = event.target.closest && event.target.closest("form[data-sr-ask]");
    if (!form || !window.fetch || !window.ReadableStream) return;
    event.preventDefault();
    run(form);
  });

  // Cmd/Ctrl+Enter submits the question.
  document.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" || !(event.metaKey || event.ctrlKey)) return;
    var form = event.target.closest && event.target.closest("form[data-sr-ask]");
    if (!form || event.target.name !== "question") return;
    event.preventDefault();
    if (form.reportValidity()) form.requestSubmit ? form.requestSubmit() : run(form);
  });
})();

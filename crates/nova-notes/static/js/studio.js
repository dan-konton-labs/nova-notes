// Behaviour for the studio panel: the in-browser meeting recorder and the
// explainer / infographic job list.
(function () {
  "use strict";

  var MAX_CHUNKS = 720;          // four hours of 20 s chunks
  var MAX_PENDING = 6;           // chunks waiting for upload before we stop
  var CHUNK_SECONDS = 20;
  var MAX_SECONDS = 4 * 60 * 60;

  function toast(message, bad) {
    if (window.nn && window.nn.toast) window.nn.toast(message, bad);
  }

  function errorText(text, status) {
    return window.nn && window.nn.errorMessage ? window.nn.errorMessage(text, status) : "Request failed (" + status + ")";
  }

  // POST and return the response text; throws Error(detail) on failure.
  function post(url, body) {
    return fetch(url, {
      method: "POST",
      body: body,
      credentials: "same-origin",
      headers: { "HX-Request": "true" }
    }).then(function (response) {
      return response.text().then(function (text) {
        if (!response.ok) throw new Error(errorText(text, response.status));
        return text;
      });
    });
  }

  function fragment(html) {
    var template = document.createElement("template");
    template.innerHTML = html.trim();
    return template.content.firstElementChild;
  }

  function cardFor(id) {
    return document.querySelector('.st-meeting[data-meeting-id="' + CSS.escape(id) + '"]');
  }

  // ---------------------------------------------------------------------
  // Audio capture
  // ---------------------------------------------------------------------

  // Bounded mono PCM capture in an AudioWorklet. No audio is written to
  // browser storage; each chunk is posted as soon as it is complete.
  var WORKLET = [
    "class MeetingPCMRecorder extends AudioWorkletProcessor {",
    "  constructor() {",
    "    super();",
    "    this.buffer = new Float32Array(sampleRate * " + CHUNK_SECONDS + ");",
    "    this.length = 0; this.active = true;",
    "    this.port.onmessage = (event) => {",
    "      if (event.data === 'stop') { this.flush(); this.active = false; this.port.postMessage({ stopped: true }); }",
    "    };",
    "  }",
    "  flush() {",
    "    if (!this.length) return;",
    "    const pcm = this.buffer.slice(0, this.length);",
    "    this.port.postMessage({ pcm }, [pcm.buffer]);",
    "    this.length = 0;",
    "  }",
    "  process(inputs) {",
    "    if (!this.active) return false;",
    "    const channels = inputs[0];",
    "    if (channels && channels.length) {",
    "      for (let i = 0; i < channels[0].length; i++) {",
    "        let sample = 0;",
    "        for (const channel of channels) sample += channel[i] / channels.length;",
    "        this.buffer[this.length++] = sample;",
    "        if (this.length === this.buffer.length) this.flush();",
    "      }",
    "    }",
    "    return true;",
    "  }",
    "}",
    "registerProcessor('meeting-pcm-recorder', MeetingPCMRecorder);"
  ].join("\n");

  // Standalone mono 16-bit PCM WAV (what the server accepts).
  function encodeWav(samples, rate) {
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);
    function text(offset, value) {
      for (var i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
    }
    text(0, "RIFF"); view.setUint32(4, buffer.byteLength - 8, true);
    text(8, "WAVE"); text(12, "fmt "); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    text(36, "data"); view.setUint32(40, samples.length * 2, true);
    for (var i = 0; i < samples.length; i++) {
      var value = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, value < 0 ? value * 32768 : value * 32767, true);
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  var MESSAGES = {
    noSharedAudio: "No shared audio was provided. Choose a source with audio and enable Share audio, or use microphone only.",
    unsupported: "Recording is unavailable in this browser. Use a supported browser over HTTPS.",
    startFailed: "Could not start. Check microphone/sharing permissions and the default speech-to-text model in Settings.",
    finishFailed: "Could not finish. Your completed transcript chunks are retained; use Finish interrupted recording to recover.",
    chunkFailed: "Some audio could not be transcribed. Missing sections will be marked in the transcript.",
    backlog: "Transcription cannot keep up. Recording stopped to keep memory bounded; queued sections are being processed.",
    sourceEnded: "An audio source disconnected. Recording stopped; check the transcript for gaps.",
    limitReached: "The four-hour recording limit has been reached."
  };

  function codedError(key) {
    var error = new Error(MESSAGES[key]);
    error.code = key;
    return error;
  }

  // Ask for permissions on the click, before a meeting exists on the server.
  function prepareCapture(shared) {
    var streams = [];
    var context = null;
    var node = null;
    var stopping = null;
    var workletUrl = null;

    function cleanup() {
      streams.forEach(function (stream) { stream.getTracks().forEach(function (track) { track.stop(); }); });
      if (node) node.disconnect();
      if (context && context.state !== "closed") context.close();
      if (workletUrl) URL.revokeObjectURL(workletUrl);
    }

    var ready = Promise.resolve();
    if (shared) {
      ready = navigator.mediaDevices.getDisplayMedia({ video: true, audio: true }).then(function (display) {
        streams.push(display);
        if (!display.getAudioTracks().length) throw codedError("noSharedAudio");
      });
    }
    return ready.then(function () {
      return navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }
      });
    }).then(function (mic) {
      streams.push(mic);
      context = new AudioContext({ sampleRate: 16000 });
      workletUrl = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      return context.audioWorklet.addModule(workletUrl);
    }).then(function () {
      node = new AudioWorkletNode(context, "meeting-pcm-recorder");
      return {
        start: function (onChunk, onEnded) {
          node.port.onmessage = function (event) {
            if (event.data.pcm) onChunk(encodeWav(event.data.pcm, context.sampleRate));
          };
          streams.forEach(function (stream) {
            var audioOnly = new MediaStream(stream.getAudioTracks());
            context.createMediaStreamSource(audioOnly).connect(node);
            stream.getTracks().forEach(function (track) { track.onended = onEnded; });
          });
          var silent = context.createGain();
          silent.gain.value = 0;
          node.connect(silent).connect(context.destination);
          return context.resume();
        },
        // Flush the last partial chunk; resolves false when it timed out.
        stop: function () {
          if (stopping) return stopping;
          stopping = new Promise(function (resolve) {
            var previous = node.port.onmessage;
            var timeout = setTimeout(function () { cleanup(); resolve(false); }, 2000);
            node.port.onmessage = function (event) {
              if (event.data.stopped) {
                clearTimeout(timeout);
                cleanup();
                resolve(true);
              } else if (previous) {
                previous.call(node.port, event);
              }
            };
            node.port.postMessage("stop");
          });
          return stopping;
        }
      };
    }).catch(function (error) {
      cleanup();
      throw error;
    });
  }

  // ---------------------------------------------------------------------
  // Recorder controls
  // ---------------------------------------------------------------------

  var session = null;   // the recording in progress in this window
  var starting = false;

  function ui(root) {
    return {
      title: root.querySelector("[data-st-title]"),
      shared: root.querySelector("[data-st-shared]"),
      record: root.querySelector("[data-st-record]"),
      stop: root.querySelector("[data-st-stop]"),
      status: root.querySelector("[data-st-status]"),
      statusText: root.querySelector("[data-st-status-text]"),
      timer: root.querySelector("[data-st-timer]"),
      busy: root.querySelector("[data-st-busy]"),
      error: root.querySelector("[data-st-error]")
    };
  }

  function setState(root, state, seconds) {
    var el = ui(root);
    var idle = state === "idle";
    el.record.hidden = !idle;
    el.title.disabled = !idle;
    el.shared.disabled = !idle;
    el.status.hidden = idle;
    el.stop.hidden = state !== "recording";
    el.busy.hidden = state === "idle" || state === "recording";
    el.statusText.textContent = { starting: "Preparing recording", recording: "Recording", finishing: "Finishing transcription" }[state] || "";
    root.classList.toggle("st-live", state === "recording");
    if (typeof seconds === "number") {
      el.timer.textContent = Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
    }
  }

  function report(root, message) {
    var el = ui(root).error;
    el.textContent = message;
    el.hidden = !message;
  }

  function markActive() {
    document.querySelectorAll(".st-meeting").forEach(function (card) {
      var active = !!session && card.getAttribute("data-meeting-id") === session.id;
      card.querySelectorAll("[data-st-inactive-only]").forEach(function (el) { el.hidden = active; });
    });
  }

  function refreshCard(id) {
    var card = cardFor(id);
    if (!card || !window.htmx) return Promise.resolve();
    return window.htmx.ajax("GET", card.getAttribute("data-card-url"), { target: card, swap: "outerHTML" });
  }

  function insertCard(html) {
    var card = fragment(html);
    if (!card) return null;
    var list = document.querySelector("[data-st-meetings]");
    if (!list) return card;
    var empty = list.querySelector(".st-meetings-empty");
    if (empty) empty.remove();
    var existing = cardFor(card.getAttribute("data-meeting-id"));
    if (existing) existing.replaceWith(card); else list.prepend(card);
    if (window.htmx) window.htmx.process(card);
    return card;
  }

  function stop(root) {
    var current = session;
    if (!current) return Promise.resolve();
    if (current.stopping) return current.stopping;
    setState(root, "finishing");
    current.stopping = (function () {
      clearInterval(current.timer);
      return current.capture.stop().then(function (flushed) {
        if (flushed === false) {
          // The last partial chunk was lost; count it so it shows as a gap.
          current.total = Math.min(MAX_CHUNKS, current.total + 1);
          report(root, MESSAGES.chunkFailed);
        }
        return current.queue;
      }).then(function () {
        var body = new URLSearchParams();
        body.set("total_chunks", String(current.total));
        return post(current.cardUrl + "/finish", body);
      }).then(function (html) {
        session = null;
        insertCard(html);
      }, function () {
        session = null;
        report(root, MESSAGES.finishFailed);
        return refreshCard(current.id);
      }).then(function () {
        setState(root, "idle");
        markActive();
      });
    })();
    return current.stopping;
  }

  function start(root) {
    if (starting || session) return;
    var el = ui(root);
    starting = true;
    report(root, "");
    setState(root, "starting", 0);
    var shared = el.shared.checked;
    var title = el.title.value.trim() || "Meeting";
    var capture = null;

    var ready = (navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.AudioWorkletNode)
      ? prepareCapture(shared)
      : Promise.reject(codedError("unsupported"));

    ready.then(function (prepared) {
      capture = prepared;
      var body = new URLSearchParams();
      body.set("title", title);
      body.set("capture_mode", shared ? "shared" : "microphone");
      return post(root.getAttribute("data-start-url"), body);
    }).then(function (html) {
      var card = insertCard(html);
      if (!card) throw new Error(MESSAGES.startFailed);
      var current = {
        id: card.getAttribute("data-meeting-id"),
        cardUrl: card.getAttribute("data-card-url"),
        capture: capture,
        total: 0,
        pending: 0,
        queue: Promise.resolve()
      };
      session = current;
      markActive();
      var began = Date.now();
      return capture.start(function (audio) {
        var sequence = current.total++;
        if (sequence >= MAX_CHUNKS) {
          current.total = MAX_CHUNKS;
          report(root, MESSAGES.limitReached);
          stop(root);
          return;
        }
        if (current.pending >= MAX_PENDING) {
          report(root, MESSAGES.backlog);
          stop(root);
          return;
        }
        current.pending++;
        // Upload one chunk at a time, in order.
        current.queue = current.queue.then(function () {
          var form = new FormData();
          form.append("audio", audio, "chunk.wav");
          return post(current.cardUrl + "/chunks/" + sequence, form).catch(function () {
            report(root, MESSAGES.chunkFailed);
          }).then(function () {
            current.pending--;
            if (session === current && !current.stopping) return refreshCard(current.id);
          });
        });
      }, function () {
        report(root, MESSAGES.sourceEnded);
        stop(root);
      }).then(function () {
        setState(root, "recording", 0);
        current.timer = setInterval(function () {
          var elapsed = Math.floor((Date.now() - began) / 1000);
          setState(root, "recording", elapsed);
          if (elapsed >= MAX_SECONDS) {
            report(root, MESSAGES.limitReached);
            stop(root);
          }
        }, 1000);
      });
    }).catch(function (error) {
      var failed = session;
      session = null;
      if (capture) capture.stop();
      if (failed) clearInterval(failed.timer);
      setState(root, "idle");
      markActive();
      // Our own messages are shown as they are; anything else (permission
      // denied, the server's reason) follows the generic hint.
      var detail = error && error.message ? error.message : "";
      report(root, error && error.code ? detail : MESSAGES.startFailed + (detail ? " (" + detail + ")" : ""));
    }).then(function () {
      starting = false;
    });
  }

  document.addEventListener("click", function (event) {
    var record = event.target.closest("[data-st-record]");
    if (record) { start(record.closest("[data-st-recorder]")); return; }
    var stopButton = event.target.closest("[data-st-stop]");
    if (stopButton) { stop(stopButton.closest("[data-st-recorder]")); return; }
    var dismiss = event.target.closest("[data-st-dismiss]");
    if (dismiss) {
      var row = dismiss.closest("[data-job-id]");
      if (row) { rememberDismissed(row.getAttribute("data-job-id")); row.remove(); }
    }
  });

  document.addEventListener("change", function (event) {
    if (!event.target.matches("[data-st-shared]")) return;
    var root = event.target.closest("[data-st-recorder]");
    root.querySelector("[data-st-mic-help]").hidden = event.target.checked;
    root.querySelector("[data-st-shared-help]").hidden = !event.target.checked;
  });

  window.addEventListener("beforeunload", function (event) {
    if (session || starting) { event.preventDefault(); event.returnValue = ""; }
  });

  // ---------------------------------------------------------------------
  // Generator jobs
  // ---------------------------------------------------------------------

  var DISMISSED_KEY = "st-dismissed-jobs";
  var seenJobs = null; // job id -> status, from the last render

  function readDismissed() {
    try {
      var ids = JSON.parse(localStorage.getItem(DISMISSED_KEY) || "[]");
      return Array.isArray(ids) ? ids : [];
    } catch (e) { return []; }
  }

  function rememberDismissed(id) {
    var ids = readDismissed().filter(function (x) { return x !== id; });
    ids.push(id);
    try { localStorage.setItem(DISMISSED_KEY, JSON.stringify(ids.slice(-50))); } catch (e) { /* ignore */ }
  }

  var LABELS = {
    explainer: { ready: "Explainer ready: ", failed: "Could not generate the explainer" },
    infographic: { ready: "Infographic ready: ", failed: "Could not generate the infographic" }
  };

  // Toast when a job seen running finishes, and tell the notes list.
  function scanJobs(list) {
    var dismissed = readDismissed();
    var now = {};
    var finished = false;
    list.querySelectorAll("[data-job-id]").forEach(function (el) {
      var id = el.getAttribute("data-job-id");
      var status = el.getAttribute("data-job-status");
      now[id] = status;
      if (status === "failed" && dismissed.indexOf(id) !== -1) el.remove();
      if (!seenJobs) return;
      var before = seenJobs[id];
      var wasActive = before === undefined || before === "queued" || before === "running";
      if (!wasActive || status === "queued" || status === "running") return;
      var labels = LABELS[el.getAttribute("data-job-kind")] || LABELS.explainer;
      if (status === "completed") {
        finished = true;
        toast(labels.ready + (el.getAttribute("data-job-title") || ""));
      } else if (status === "failed") {
        toast(labels.failed + ": " + (el.getAttribute("data-job-error") || "Something went wrong."), true);
      }
    });
    seenJobs = now;
    if (finished) document.body.dispatchEvent(new CustomEvent("notes-changed", { bubbles: true }));
  }

  function onLoad(root) {
    var lists = root.matches && root.matches("[data-st-jobs]") ? [root] : root.querySelectorAll("[data-st-jobs]");
    Array.prototype.forEach.call(lists, scanJobs);
    if (root.querySelector && (root.matches(".st-meeting") || root.querySelector(".st-meeting"))) markActive();
  }

  document.addEventListener("htmx:load", function (event) { onLoad(event.target); });
  document.addEventListener("DOMContentLoaded", function () { onLoad(document.body); });
})();

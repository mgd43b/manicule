/* manicule browser surface: streaming, the command palette, the theme, and the buttons.
 *
 * Three rules this file is written under, each of which has a test against what the route
 * actually served rather than against this source:
 *
 * 1. **It builds DOM, never markup.** Every piece of answer text, every citation label and
 *    every error message reaches the page through `textContent`. No markup-writing property or
 *    method is used anywhere below, and a test asserts that against what this route served.
 *    An answer is model output over documents somebody else wrote; treating it as markup would
 *    make any document in the corpus a script into this page.
 * 2. **It adds no operation.** Every request below is a route the HTTP API already publishes.
 *    There is no upload, no configuration write and no plugin install, because there is none
 *    there.
 * 3. **Every request carries a JSON content type.** That is not decoration: a JSON body is not
 *    a "simple request", so a cross-origin attempt at one is preflighted and refused rather
 *    than sent. The server refuses cross-origin unsafe methods as well; this is the half a
 *    browser can enforce on its own.
 *
 * No framework, no bundle, no build step. `uv sync` is the whole install.
 */
(function () {
  "use strict";

  var THEME_KEY = "manicule.theme";

  function json(method, path, body) {
    var options = {
      method: method,
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
    };
    options.body = JSON.stringify(body || {});
    return fetch(path, options).then(function (response) {
      return response.json().then(function (envelope) {
        return { status: response.status, envelope: envelope };
      }).catch(function () {
        return { status: response.status, envelope: null };
      });
    });
  }

  function say(node, message) {
    if (!node) { return; }
    node.textContent = message;
  }

  function failure(result) {
    var envelope = result && result.envelope;
    if (envelope && envelope.error && envelope.error.message) {
      return envelope.error.type + ": " + envelope.error.message;
    }
    return "The request was refused (" + (result ? result.status : "no response") + ").";
  }

  function actionState(node, state, message) {
    if (!node) { return; }
    node.setAttribute("role", state === "error" ? "alert" : "status");
    node.setAttribute("aria-live", state === "error" ? "assertive" : "polite");
    node.setAttribute("data-action-state", state);
    say(node, message);
  }

  function nearbyStatus(control, node) {
    if (node) { return node; }
    var created = document.createElement("span");
    created.className = "muted";
    created.setAttribute("data-generated-status", "");
    control.parentNode.insertBefore(created, control.nextSibling);
    return created;
  }

  function busy(control, value) {
    control.disabled = value;
    if (value) {
      control.setAttribute("aria-busy", "true");
    } else {
      control.removeAttribute("aria-busy");
    }
  }

  function runAction(control, node, pending, request, success) {
    node = nearbyStatus(control, node);
    if (node.getAttribute("data-action-running") === "true") { return Promise.resolve(); }
    node.setAttribute("data-action-running", "true");
    busy(control, true);
    actionState(node, "pending", pending);
    return Promise.resolve().then(request).then(function (result) {
      if (!result.envelope || !result.envelope.ok) {
        node.removeAttribute("data-action-running");
        busy(control, false);
        actionState(node, "error", failure(result));
        return;
      }
      success(result, node);
    }).catch(function () {
      node.removeAttribute("data-action-running");
      busy(control, false);
      actionState(node, "error", "The service could not be reached. Try again.");
    });
  }

  /* Reload after a mutation rather than patching the table in place. The page is rendered by
   * the server from one envelope; re-rendering it there keeps one description of what a
   * listing is, instead of a second one written in this file that can disagree. */
  function reloadAfterChange(result, node, message) {
    actionState(node, "success", message + " Reloading…");
    window.setTimeout(function () { window.location.reload(); }, 250);
  }

  function act(selector, node, pending, success, request) {
    document.querySelectorAll(selector).forEach(function (element) {
      element.addEventListener("click", function () {
        runAction(element, node, pending, function () {
          return request(element);
        }, function (result, status) {
          reloadAfterChange(result, status, success);
        });
      });
    });
  }

  function readEventStream(response, onEvent) {
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var finalEnvelope = null;
    var finalCount = 0;

    function readFrame(raw) {
      var name = "";
      var payload = "";
      raw.split(/\r?\n/).forEach(function (line) {
        if (line.indexOf("event: ") === 0) { name = line.slice(7); }
        if (line.indexOf("data: ") === 0) { payload = line.slice(6); }
      });
      if (!payload) { return; }
      var parsed;
      try { parsed = JSON.parse(payload); } catch (error) { return; }
      if (finalCount && name !== "final") {
        throw new Error("The answer stream continued after its final frame.");
      }
      if (name === "final") {
        finalCount += 1;
        finalEnvelope = parsed;
      } else {
        onEvent(name, parsed);
      }
    }

    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) {
          buffer += decoder.decode();
          if (buffer.trim()) { readFrame(buffer); }
          if (finalCount !== 1) {
            throw new Error("The answer stream did not contain exactly one final frame.");
          }
          return finalEnvelope;
        }
        buffer += decoder.decode(chunk.value, { stream: true });
        var frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop();
        frames.forEach(readFrame);
        return pump();
      });
    }
    return pump();
  }

  /* --- theme ---------------------------------------------------------------------------- */

  function applyTheme(value) {
    if (value === "light" || value === "dark") {
      document.documentElement.setAttribute("data-theme", value);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }

  /* What the page currently *looks* like, which is not the same as what it was told to be.
   * With no stored preference there is no `data-theme` attribute and the stylesheet follows
   * `prefers-color-scheme`, so reading the attribute alone reports "light" for a page that is
   * plainly dark. The toggle did exactly that: on a machine set to dark, the first press
   * computed "not dark, therefore go dark" and changed nothing a person could see. */
  function effectiveTheme() {
    var explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "light" || explicit === "dark") { return explicit; }
    try {
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return "dark";
      }
    } catch (error) { /* matchMedia is absent in some embedded views */ }
    return "light";
  }

  function startTheme() {
    var stored = null;
    try { stored = window.localStorage.getItem(THEME_KEY); } catch (error) { stored = null; }
    applyTheme(stored);
    var button = document.querySelector("[data-theme-toggle]");
    if (!button) { return; }
    button.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
      /* The only thing this page stores. A theme is not a credential; the widget stores
       * nothing at all precisely because what it holds is one. */
      try { window.localStorage.setItem(THEME_KEY, next); } catch (error) { /* private mode */ }
    });
  }

  /* --- command palette and keyboard navigation ------------------------------------------ */

  function startPalette() {
    var palette = document.querySelector("[data-palette]");
    var input = document.querySelector("[data-palette-input]");
    var list = document.querySelector("[data-palette-list]");
    var source = document.querySelector("[data-navigation]");
    if (!palette || !input || !list || !source) { return; }

    var entries = [];
    source.querySelectorAll("a").forEach(function (link) {
      entries.push({ label: link.textContent.trim(), href: link.getAttribute("href") });
    });

    var selected = 0;

    function draw() {
      var needle = input.value.trim().toLowerCase();
      var shown = entries.filter(function (entry) {
        return !needle || entry.label.toLowerCase().indexOf(needle) >= 0;
      });
      if (selected >= shown.length) { selected = Math.max(shown.length - 1, 0); }
      list.textContent = "";
      shown.forEach(function (entry, index) {
        var item = document.createElement("li");
        item.setAttribute("aria-selected", index === selected ? "true" : "false");
        var link = document.createElement("a");
        link.href = entry.href;
        link.textContent = entry.label;
        item.appendChild(link);
        list.appendChild(item);
      });
      return shown;
    }

    /* Where focus was before the palette took it. Closing a dialog without putting focus back
     * leaves it on an element that is now `hidden`, so the browser drops it to <body> and the
     * next Tab restarts from the top of the page — a keyboard user loses their place every
     * time they press Escape. */
    var returnFocusTo = null;

    function open() {
      if (palette.hidden) {
        returnFocusTo = document.activeElement;
      }
      palette.hidden = false;
      input.value = "";
      selected = 0;
      draw();
      input.focus();
    }

    function close() {
      if (palette.hidden) { return; }
      palette.hidden = true;
      if (returnFocusTo && typeof returnFocusTo.focus === "function") {
        returnFocusTo.focus();
      }
      returnFocusTo = null;
    }

    document.querySelectorAll("[data-palette-open]").forEach(function (button) {
      button.addEventListener("click", open);
    });

    palette.addEventListener("click", function (event) {
      if (event.target === palette) { close(); }
    });

    input.addEventListener("input", draw);

    input.addEventListener("keydown", function (event) {
      var shown = draw();
      if (event.key === "ArrowDown") {
        selected = Math.min(selected + 1, shown.length - 1);
        draw();
        event.preventDefault();
      } else if (event.key === "ArrowUp") {
        selected = Math.max(selected - 1, 0);
        draw();
        event.preventDefault();
      } else if (event.key === "Enter" && shown[selected]) {
        window.location.href = shown[selected].href;
        event.preventDefault();
      } else if (event.key === "Escape") {
        close();
      }
    });

    document.addEventListener("keydown", function (event) {
      var focused = document.activeElement;
      var typing = !!focused && /^(INPUT|TEXTAREA|SELECT)$/.test(focused.tagName);
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        open();
        event.preventDefault();
        return;
      }
      if (event.key === "Escape") { close(); return; }
      if (typing) { return; }
      if (event.key === "/") {
        var search = document.getElementById("q");
        if (search) { search.focus(); event.preventDefault(); }
      } else if (event.key === "?") {
        open();
        event.preventDefault();
      }
    });
  }

  /* --- chat ------------------------------------------------------------------------------ */

  function startChat() {
    var thread = document.querySelector("[data-chat]");
    if (!thread) { return; }
    var form = thread.querySelector("[data-ask]");
    var turns = thread.querySelector("[data-turns]");
    var live = thread.querySelector("[data-live]");
    var answer = thread.querySelector("[data-answer]");
    var citations = thread.querySelector("[data-citations]");
    var verdict = thread.querySelector("[data-verdict]");
    var rate = thread.querySelector("[data-rate]");
    var rated = thread.querySelector("[data-rated]");
    var status = thread.querySelector("[data-status]");
    var profile = thread.querySelector("[data-profile]");
    var question = thread.querySelector("#question");
    var submit = form.querySelector('[type="submit"]');
    var stop = form.querySelector("[data-stop]");
    var retry = form.querySelector("[data-retry]");
    var conversation = thread.getAttribute("data-conversation") || "";
    var lastMessage = "";
    var requestState = "idle";
    var requestNumber = 0;
    var currentRequest = null;
    var retryRequest = null;
    var liveComplete = false;

    function setRequestState(state) {
      requestState = state;
      busy(submit, state !== "idle");
      stop.hidden = state !== "streaming";
      if (state === "idle") {
        form.removeAttribute("aria-busy");
      } else {
        form.setAttribute("aria-busy", "true");
      }
    }

    function addAsked(text) {
      var article = document.createElement("article");
      article.className = "turn turn-user";
      var role = document.createElement("p");
      role.className = "role";
      role.textContent = "you";
      var body = document.createElement("div");
      body.className = "body";
      body.textContent = text;
      article.appendChild(role);
      article.appendChild(body);
      turns.appendChild(article);
    }

    function addCitation(citation) {
      var item = document.createElement("li");
      item.className = "cite";
      var slot = document.createElement("span");
      slot.className = "slot";
      slot.textContent = "[" + citation.slot + "]";
      var title = document.createElement("span");
      title.className = "title";
      title.textContent = citation.title || "";
      item.appendChild(slot);
      item.appendChild(title);
      var path = citation.heading_path || [];
      if (path.length) {
        var trail = document.createElement("span");
        trail.className = "trail";
        trail.textContent = path.join(" › ");
        item.appendChild(trail);
      }
      var mark = document.createElement("span");
      mark.className = "verdict verdict-" + (citation.verification || "");
      mark.textContent = citation.verification || "";
      item.appendChild(mark);
      citations.appendChild(item);
    }

    function settle(envelope) {
      if (!envelope.ok) {
        say(status, failure({ status: 0, envelope: envelope }));
        return;
      }
      var data = envelope.data || {};
      var parts = [];
      if (data.confidence !== null && data.confidence !== undefined) {
        /* Two decimals, matching `manicule ask`. The raw value arrives as a double and read
         * as "confidence 0.5232638788223267", which claims sixteen digits of precision for an
         * uncalibrated number and disagrees with the command line about the same answer. */
        parts.push("confidence " + Number(data.confidence).toFixed(2) +
                   " (" + (data.confidence_band || "") + ")");
      } else {
        parts.push("the corpus was not consulted, so there is no confidence to report");
      }
      if (data.dropped) { parts.push(data.dropped + " citation(s) dropped as unverifiable"); }
      if (data.ungrounded) { parts.push("nothing survived verification"); }
      if (data.context_truncated) { parts.push("context truncated"); }
      if (data.model) { parts.push(data.model); }
      say(verdict, parts.join(" · "));
      lastMessage = data.message_id || "";
      if (lastMessage) {
        rate.hidden = false;
        say(rated, "");
      }
      if (data.conversation_id && !conversation) {
        conversation = data.conversation_id;
        thread.setAttribute("data-conversation", conversation);
      }
      liveComplete = true;
      actionState(status, "success", "Answer complete.");
    }

    function isCurrent(request) {
      return currentRequest && currentRequest.number === request.number;
    }

    function readFrames(response, request) {
      return readEventStream(response, function (name, parsed) {
        if (!isCurrent(request)) { return; }
        if (name === "delta") {
          answer.textContent += parsed.text || "";
        } else if (name === "citation" && parsed.citation) {
          addCitation(parsed.citation);
        }
      }).then(function (finalEnvelope) {
        if (!isCurrent(request)) { return; }
        if (!finalEnvelope.ok) {
          var error = new Error(failure({ status: 0, envelope: finalEnvelope }));
          error.terminal = true;
          throw error;
        }
        settle(finalEnvelope);
      });
    }

    function failRequest(request, message) {
      if (!isCurrent(request)) { return; }
      currentRequest = null;
      setRequestState("idle");
      retry.hidden = false;
      actionState(status, "error", message);
    }

    function finishRequest(request) {
      if (!isCurrent(request)) { return; }
      currentRequest = null;
      retryRequest = null;
      retry.hidden = true;
      setRequestState("idle");
    }

    /* The answer that was on screen, moved into the thread before the next one overwrites the
     * live block. Without this, asking a second question makes the first answer vanish — the
     * live block is one element and the turns above it come from the server. The rating widget
     * is dropped from the copy: cloning does not carry its listener, so a button that looked
     * live and did nothing is worse than no button. */
    function keepPreviousAnswer() {
      if (!liveComplete || live.hidden || !answer.textContent) { return; }
      var finished = live.firstElementChild;
      if (!finished) { return; }
      var copy = finished.cloneNode(true);
      var stale = copy.querySelector("[data-rate]");
      if (stale) { stale.parentNode.removeChild(stale); }
      turns.appendChild(copy);
    }

    function ask(saved, addTurn) {
      if (requestState !== "idle") { return; }
      if (addTurn) {
        keepPreviousAnswer();
        addAsked(saved.text);
      }
      live.hidden = false;
      liveComplete = false;
      rate.hidden = true;
      answer.textContent = "";
      citations.textContent = "";
      say(verdict, "");
      setRequestState("streaming");
      actionState(status, "pending", "Asking…");
      retry.hidden = true;
      retryRequest = saved;
      var request = {
        number: requestNumber += 1,
        controller: new AbortController(),
        responseArrived: false,
        responseStarted: false,
      };
      currentRequest = request;
      fetch("/api/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(saved.body),
        signal: request.controller.signal,
      }).then(function (response) {
        if (!isCurrent(request)) { return; }
        request.responseArrived = true;
        if (!response.ok || !response.body) {
          return response.json().catch(function () { return null; }).then(function (envelope) {
            throw new Error(failure({ status: response.status, envelope: envelope }));
          });
        }
        request.responseStarted = true;
        return readFrames(response, request);
      }).then(function () {
        finishRequest(request);
      }).catch(function (error) {
        if (!isCurrent(request)) { return; }
        failRequest(
          request,
          request.responseStarted
            ? error.terminal
              ? error.message
              : "The answer stream ended before completion. Retry the question."
            : request.responseArrived
              ? (error.message || "The request was refused.")
              : "The corpus could not be reached. Try again."
        );
      });
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      if (requestState !== "idle") { return; }
      var text = question.value.trim();
      if (!text) { return; }
      var body = { question: text };
      if (profile.value) { body.profile = profile.value; }
      if (conversation) { body.conversation_id = conversation; }
      question.value = "";
      ask({ text: text, body: body }, true);
    });

    stop.addEventListener("click", function () {
      if (!currentRequest) { return; }
      var request = currentRequest;
      currentRequest = null;
      request.controller.abort();
      setRequestState("idle");
      retry.hidden = false;
      actionState(status, "stopped", "Stopped. The same question can be retried.");
    });

    retry.addEventListener("click", function () {
      if (retryRequest) { ask(retryRequest, false); }
    });

    question.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    thread.querySelectorAll("[data-feedback]").forEach(function (button) {
      button.addEventListener("click", function () {
        if (!lastMessage) {
          actionState(rated, "error", "There is no answer to rate yet.");
          return;
        }
        runAction(button, rated, "Recording…", function () {
          return json("POST", "/api/v1/chat/feedback", {
            message_id: lastMessage,
            feedback: button.getAttribute("data-feedback"),
          });
        }, function (result, node) {
          node.removeAttribute("data-action-running");
          busy(button, false);
          actionState(node, "success", "Recorded.");
        });
      });
    });
  }

  /* --- documents, collections, connectors, plugins, keys --------------------------------- */

  function startActions() {
    var documentStatus = document.querySelector("[data-document-status]");

    act("[data-reindex]", documentStatus, "Reindexing…", "Reindexed.", function (element) {
      return json("POST", "/api/v1/documents/" + encodeURIComponent(element.getAttribute("data-reindex")) + "/reindex")
    });

    act("[data-trash]", documentStatus, "Moving to trash…", "Moved to trash.", function (element) {
      return json("DELETE", "/api/v1/documents/" + encodeURIComponent(element.getAttribute("data-trash")))
    });

    act("[data-restore]", documentStatus, "Restoring…", "Restored.", function (element) {
      return json("POST", "/api/v1/documents/" + encodeURIComponent(element.getAttribute("data-restore")) + "/restore")
    });

    var syncStatus = document.querySelector("[data-sync-status]");
    document.querySelectorAll("[data-sync]").forEach(function (element) {
      element.addEventListener("click", function () {
        runAction(element, syncStatus, "Syncing…", function () {
          return json("POST", "/api/v1/admin/connectors/" + encodeURIComponent(element.getAttribute("data-sync")) + "/sync", {});
        }, function (result, node) {
          var report = result.envelope.data || {};
          node.removeAttribute("data-action-running");
          busy(element, false);
          actionState(node, "success", "Ingested " + report.ingested + ", skipped " + report.skipped +
                      ", failed " + report.failed + (report.error ? " — " + report.error : ""));
        });
      });
    });

    var pluginStatus = document.querySelector("[data-plugin-status]");
    act("[data-plugin-enable]", pluginStatus, "Enabling…", "Enabled.", function (element) {
      return json("POST", "/api/v1/plugins/" + encodeURIComponent(element.getAttribute("data-plugin-enable")))
    });
    act("[data-plugin-disable]", pluginStatus, "Disabling…", "Disabled.", function (element) {
      return json("DELETE", "/api/v1/plugins/" + encodeURIComponent(element.getAttribute("data-plugin-disable")))
    });

    var collectionStatus = document.querySelector("[data-collection-status]");
    var collectionForm = document.querySelector("[data-create-collection]");
    if (collectionForm) {
      collectionForm.addEventListener("submit", function (event) {
        event.preventDefault();
        var submit = collectionForm.querySelector('[type="submit"]');
        runAction(submit, collectionStatus, "Creating…", function () {
          return json("POST", "/api/v1/collections", { name: collectionForm.elements.name.value });
        }, function (result, node) { reloadAfterChange(result, node, "Created."); });
      });
    }
    act("[data-delete-collection]", collectionStatus, "Deleting…", "Deleted.", function (element) {
      return json("DELETE", "/api/v1/collections/" + encodeURIComponent(element.getAttribute("data-delete-collection")))
    });

    var tagStatus = document.querySelector("[data-tag-status]");
    var tagForm = document.querySelector("[data-create-tag]");
    if (tagForm) {
      tagForm.addEventListener("submit", function (event) {
        event.preventDefault();
        var submit = tagForm.querySelector('[type="submit"]');
        runAction(submit, tagStatus, "Creating…", function () {
          return json("POST", "/api/v1/tags", { name: tagForm.elements.name.value });
        }, function (result, node) { reloadAfterChange(result, node, "Created."); });
      });
    }
    act("[data-delete-tag]", tagStatus, "Deleting…", "Deleted.", function (element) {
      return json("DELETE", "/api/v1/tags/" + encodeURIComponent(element.getAttribute("data-delete-tag")))
    });

    var keyStatus = document.querySelector("[data-key-status]");
    var secret = document.querySelector("[data-secret]");
    var keyForm = document.querySelector("[data-create-key]");
    if (keyForm) {
      keyForm.addEventListener("submit", function (event) {
        event.preventDefault();
        var days = keyForm.elements.expires_days.value;
        var body = { name: keyForm.elements.name.value, role: keyForm.elements.role.value };
        if (days) { body.expires_days = Number(days); }
        var submit = keyForm.querySelector('[type="submit"]');
        runAction(submit, keyStatus, "Minting…", function () {
          return json("POST", "/api/v1/auth/keys", body);
        }, function (result, node) {
          /* The only copy of the secret. Written into the page and nowhere else: not stored,
           * not logged, and gone as soon as this page is left. */
          secret.hidden = false;
          say(secret, result.envelope.data.secret);
          node.removeAttribute("data-action-running");
          busy(submit, false);
          actionState(node, "success", "Copy this now — only a digest is kept, so it cannot be shown again.");
        });
      });
    }
    act("[data-revoke]", keyStatus, "Revoking…", "Revoked.", function (element) {
      return json("DELETE", "/api/v1/auth/keys/" + encodeURIComponent(element.getAttribute("data-revoke")))
    });
  }

  startTheme();
  startPalette();
  startChat();
  startActions();
})();

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

  /* Reload after a mutation rather than patching the table in place. The page is rendered by
   * the server from one envelope; re-rendering it there keeps one description of what a
   * listing is, instead of a second one written in this file that can disagree. */
  function afterChange(element, node, result) {
    if (result.envelope && result.envelope.ok) {
      window.location.reload();
      return;
    }
    /* Re-enabled on a refusal. `json()` resolves for a 4xx as well as a 2xx — an envelope is
     * an envelope — so a failure that only re-enabled on a rejected promise would leave the
     * button disabled for good, with the reason printed beside a control nobody can retry. */
    if (element) { element.disabled = false; }
    say(node, failure(result));
  }

  function act(selector, run) {
    document.querySelectorAll(selector).forEach(function (element) {
      element.addEventListener("click", function () {
        element.disabled = true;
        run(element).catch(function () {
          element.disabled = false;
        });
      });
    });
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
    var conversation = thread.getAttribute("data-conversation") || "";
    var lastMessage = "";

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
      say(status, "");
    }

    function readFrames(response) {
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) { return; }
          buffer += decoder.decode(chunk.value, { stream: true });
          var frames = buffer.split("\n\n");
          buffer = frames.pop();
          frames.forEach(function (raw) {
            var name = "";
            var payload = "";
            raw.split("\n").forEach(function (line) {
              if (line.indexOf("event: ") === 0) { name = line.slice(7); }
              if (line.indexOf("data: ") === 0) { payload = line.slice(6); }
            });
            if (!payload) { return; }
            var parsed;
            try { parsed = JSON.parse(payload); } catch (error) { return; }
            if (name === "delta") {
              answer.textContent += parsed.text || "";
            } else if (name === "citation" && parsed.citation) {
              addCitation(parsed.citation);
            } else if (name === "final") {
              settle(parsed);
            }
          });
          return pump();
        });
      }
      return pump();
    }

    /* The answer that was on screen, moved into the thread before the next one overwrites the
     * live block. Without this, asking a second question makes the first answer vanish — the
     * live block is one element and the turns above it come from the server. The rating widget
     * is dropped from the copy: cloning does not carry its listener, so a button that looked
     * live and did nothing is worse than no button. */
    function keepPreviousAnswer() {
      if (live.hidden || !answer.textContent) { return; }
      var finished = live.firstElementChild;
      if (!finished) { return; }
      var copy = finished.cloneNode(true);
      var stale = copy.querySelector("[data-rate]");
      if (stale) { stale.parentNode.removeChild(stale); }
      turns.appendChild(copy);
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var text = question.value.trim();
      if (!text) { return; }
      question.value = "";
      keepPreviousAnswer();
      addAsked(text);
      live.hidden = false;
      rate.hidden = true;
      answer.textContent = "";
      citations.textContent = "";
      say(verdict, "");
      say(status, "asking…");
      var body = { question: text };
      if (profile.value) { body.profile = profile.value; }
      if (conversation) { body.conversation_id = conversation; }
      fetch("/api/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(body),
      }).then(function (response) {
        if (!response.ok || !response.body) {
          return response.json().then(function (envelope) {
            say(status, failure({ status: response.status, envelope: envelope }));
          }).catch(function () {
            say(status, "The request was refused (" + response.status + ").");
          });
        }
        return readFrames(response);
      }).catch(function () {
        say(status, "The corpus could not be reached.");
      });
    });

    question.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    thread.querySelectorAll("[data-feedback]").forEach(function (button) {
      button.addEventListener("click", function () {
        if (!lastMessage) { say(rated, "There is no answer to rate yet."); return; }
        json("POST", "/api/v1/chat/feedback", {
          message_id: lastMessage,
          feedback: button.getAttribute("data-feedback"),
        }).then(function (result) {
          say(rated, result.envelope.ok ? "Recorded." : failure(result));
        });
      });
    });
  }

  /* --- documents, collections, connectors, plugins, keys --------------------------------- */

  function startActions() {
    var documentStatus = document.querySelector("[data-document-status]");

    act("[data-reindex]", function (element) {
      return json("POST", "/api/v1/documents/" + encodeURIComponent(element.getAttribute("data-reindex")) + "/reindex")
        .then(function (result) { afterChange(element, documentStatus, result); });
    });

    act("[data-trash]", function (element) {
      return json("DELETE", "/api/v1/documents/" + encodeURIComponent(element.getAttribute("data-trash")))
        .then(function (result) { afterChange(element, documentStatus, result); });
    });

    act("[data-restore]", function (element) {
      return json("POST", "/api/v1/documents/" + encodeURIComponent(element.getAttribute("data-restore")) + "/restore")
        .then(function (result) { afterChange(element, documentStatus, result); });
    });

    var syncStatus = document.querySelector("[data-sync-status]");
    act("[data-sync]", function (element) {
      say(syncStatus, "syncing…");
      return json("POST", "/api/v1/admin/connectors/" + encodeURIComponent(element.getAttribute("data-sync")) + "/sync", {})
        .then(function (result) {
          element.disabled = false;
          if (!result.envelope.ok) { say(syncStatus, failure(result)); return; }
          var report = result.envelope.data || {};
          say(syncStatus, "ingested " + report.ingested + ", skipped " + report.skipped +
                          ", failed " + report.failed + (report.error ? " — " + report.error : ""));
        });
    });

    var pluginStatus = document.querySelector("[data-plugin-status]");
    act("[data-plugin-enable]", function (element) {
      return json("POST", "/api/v1/plugins/" + encodeURIComponent(element.getAttribute("data-plugin-enable")))
        .then(function (result) { afterChange(element, pluginStatus, result); });
    });
    act("[data-plugin-disable]", function (element) {
      return json("DELETE", "/api/v1/plugins/" + encodeURIComponent(element.getAttribute("data-plugin-disable")))
        .then(function (result) { afterChange(element, pluginStatus, result); });
    });

    var collectionStatus = document.querySelector("[data-collection-status]");
    var collectionForm = document.querySelector("[data-create-collection]");
    if (collectionForm) {
      collectionForm.addEventListener("submit", function (event) {
        event.preventDefault();
        json("POST", "/api/v1/collections", { name: collectionForm.elements.name.value })
          .then(function (result) { afterChange(null, collectionStatus, result); });
      });
    }
    act("[data-delete-collection]", function (element) {
      return json("DELETE", "/api/v1/collections/" + encodeURIComponent(element.getAttribute("data-delete-collection")))
        .then(function (result) { afterChange(element, collectionStatus, result); });
    });

    var tagStatus = document.querySelector("[data-tag-status]");
    var tagForm = document.querySelector("[data-create-tag]");
    if (tagForm) {
      tagForm.addEventListener("submit", function (event) {
        event.preventDefault();
        json("POST", "/api/v1/tags", { name: tagForm.elements.name.value })
          .then(function (result) { afterChange(null, tagStatus, result); });
      });
    }
    act("[data-delete-tag]", function (element) {
      return json("DELETE", "/api/v1/tags/" + encodeURIComponent(element.getAttribute("data-delete-tag")))
        .then(function (result) { afterChange(element, tagStatus, result); });
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
        json("POST", "/api/v1/auth/keys", body).then(function (result) {
          if (!result.envelope.ok) { say(keyStatus, failure(result)); return; }
          /* The only copy of the secret. Written into the page and nowhere else: not stored,
           * not logged, and gone as soon as this page is left. */
          secret.hidden = false;
          say(secret, result.envelope.data.secret);
          say(keyStatus, "Copy this now — only a digest is kept, so it cannot be shown again.");
        });
      });
    }
    act("[data-revoke]", function (element) {
      return json("DELETE", "/api/v1/auth/keys/" + encodeURIComponent(element.getAttribute("data-revoke")))
        .then(function (result) { afterChange(element, keyStatus, result); });
    });
  }

  startTheme();
  startPalette();
  startChat();
  startActions();
})();

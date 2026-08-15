"""The browser's asynchronous request states, executed rather than inferred from source."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from manicule.web import rendering

_NODE = shutil.which("node")
if _NODE is None:
    pytest.skip("the JavaScript runtime tests require node", allow_module_level=True)
NODE: str = _NODE
SCRIPT = Path(rendering.HERE) / "static" / "manicule.js"
BOOT = r"""
const fs = require("fs");
const vm = require("vm");
globalThis.window = globalThis;
globalThis.location = {reload() {}};
let source = fs.readFileSync(process.argv[1], "utf8");
const start = "  startTheme();\n  startPalette();\n  startChat();\n  startActions();";
const expose = "  globalThis.hooks = { json: json, runAction: runAction, " +
               "reloadAfterChange: reloadAfterChange, " +
               "readEventStream: readEventStream, startChat: startChat };";
if (!source.includes(start)) { throw new Error("script startup marker moved"); }
source = source.replace(start, expose);
vm.runInThisContext(source);
"""


def run_javascript(body: str) -> dict[str, object]:
    """Run one dependency-free browser-state scenario against the served script's source."""
    completed = subprocess.run(  # noqa: S603 - fixed runtime and repository-owned script
        [
            NODE,
            "-e",
            BOOT + "\n(async () => {\n" + body + "\n})().catch(e => { throw e; });",
            str(SCRIPT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast("dict[str, object]", json.loads(completed.stdout))


def test_mutations_report_every_failure_and_queue_shared_status_actions() -> None:
    """Shared status actions run in order, remain visible, and can be retried after failure."""
    result = run_javascript(
        r"""
function element() {
  return {
    attrs: {}, disabled: false, textContent: "",
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k] || null; },
    removeAttribute(k) { delete this.attrs[k]; },
  };
}
const status = element();
const control = element();
await hooks.runAction(control, status, "Working…", () => Promise.resolve({
  status: 409, envelope: {ok: false, error: {type: "conflict", message: "already exists"}}
}), () => {});
const envelope = {message: status.textContent, role: status.attrs.role, disabled: control.disabled};

globalThis.fetch = () => Promise.resolve({
  status: 502, json: () => Promise.reject(new Error("html"))
});
await hooks.runAction(control, status, "Working…", () => hooks.json("POST", "/x", {}), () => {});
const nonJson = status.textContent;

globalThis.fetch = () => Promise.reject(new Error("offline"));
await hooks.runAction(control, status, "Working…", () => hooks.json("POST", "/x", {}), () => {});
const network = status.textContent;

const queueStatus = element();
const history = [];
let visible = "";
Object.defineProperty(queueStatus, "textContent", {
  get() { return visible; },
  set(value) { visible = value; history.push(value); }
});
const firstControl = element();
const secondControl = element();
let release;
const order = [];
const held = new Promise(resolve => { release = resolve; });
const first = hooks.runAction(firstControl, queueStatus, "First…", () => {
  order.push("first");
  return held;
}, () => {});
const second = hooks.runAction(secondControl, queueStatus, "Second…", () => {
  order.push("second");
  return Promise.resolve({
    status: 409, envelope: {ok: false, error: {type: "bad", message: "second"}}
  });
}, () => {});
await hooks.runAction(secondControl, queueStatus, "Duplicate…", () => {
  order.push("duplicate");
  return Promise.resolve({status: 200, envelope: {ok: true}});
}, () => {});
const queuedDisabled = secondControl.disabled;
release({
  status: 409, envelope: {ok: false, error: {type: "bad", message: "first"}}
});
await Promise.all([first, second]);
const retryable = !firstControl.disabled && !secondControl.disabled;
await hooks.runAction(secondControl, queueStatus, "Retry…", () => {
  order.push("retry");
  return Promise.resolve({status: 200, envelope: {ok: true}});
}, (result, node) => {
  node.setAttribute("role", "status");
  node.setAttribute("data-action-state", "success");
  node.textContent = "Retried.";
});
console.log(JSON.stringify({
  envelope, nonJson, network, queuedDisabled, retryable, order, history,
  retryDisabled: secondControl.disabled
}));
"""
    )
    assert result == {
        "envelope": {"message": "conflict: already exists", "role": "alert", "disabled": False},
        "nonJson": "The request was refused (502).",
        "network": "The service could not be reached. Try again.",
        "queuedDisabled": True,
        "retryable": True,
        "order": ["first", "second", "retry"],
        "history": ["First…", "bad: first", "Second…", "bad: second", "Retry…", "Retried."],
        "retryDisabled": False,
    }


def test_a_later_queued_failure_preserves_an_earlier_successful_reload() -> None:
    """A committed mutation is refreshed only after the later refusal has remained visible."""
    result = run_javascript(
        r"""
const delays = [];
let reloads = 0;
window.location.reload = () => { reloads += 1; };
window.setTimeout = (callback, delay) => {
  delays.push(delay);
  queueMicrotask(callback);
  return delays.length;
};
window.clearTimeout = () => {};
function element() {
  return {
    attrs: {}, disabled: false, textContent: "",
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k] || null; },
    removeAttribute(k) { delete this.attrs[k]; },
  };
}
const status = element();
const first = element();
const second = element();
const history = [];
let visible = "";
Object.defineProperty(status, "textContent", {
  get() { return visible; },
  set(value) { visible = value; history.push(value); }
});
const succeeded = hooks.runAction(first, status, "First…", () => Promise.resolve({
  status: 200, envelope: {ok: true}
}), (response, node) => hooks.reloadAfterChange(response, node, "First changed."));
const failed = hooks.runAction(second, status, "Second…", () => Promise.resolve({
  status: 409, envelope: {ok: false, error: {type: "conflict", message: "second refused"}}
}), () => {});
await Promise.all([succeeded, failed]);
await new Promise(resolve => setImmediate(resolve));
console.log(JSON.stringify({delays, reloads, history}));
"""
    )
    assert result == {
        "delays": [250, 250, 4000],
        "reloads": 1,
        "history": [
            "First…",
            "First changed. Reloading…",
            "Second…",
            (
                "conflict: second refused An earlier change succeeded; "
                "reloading in a few seconds to show it."
            ),
        ],
    }


def test_stream_requires_one_terminal_frame_and_flushes_an_undelimited_tail() -> None:
    """A clean socket close is not success unless exactly one final envelope arrived."""
    result = run_javascript(
        r"""
function response(parts) {
  return {body: new ReadableStream({start(controller) {
    parts.forEach(part => controller.enqueue(new TextEncoder().encode(part)));
    controller.close();
  }})};
}
const events = [];
const final = await hooks.readEventStream(response([
  "event: delta\r\ndata: {\"text\":\"hel",
  "lo\"}\r\n\r\nevent: final\r\ndata: {\"ok\":true}"
]), (name, data) => events.push([name, data.text]));
async function refusal(text) {
  try { await hooks.readEventStream(response([text]), () => {}); return "accepted"; }
  catch (error) { return error.message; }
}
const missing = await refusal("event: delta\ndata: {\"text\":\"partial\"}\n\n");
const duplicate = await refusal(
  "event: final\ndata: {\"ok\":true}\n\nevent: final\ndata: {\"ok\":true}\n\n"
);
const after = await refusal(
  "event: final\ndata: {\"ok\":true}\n\nevent: delta\ndata: {\"text\":\"late\"}\n\n"
);
const malformed = await refusal("event: delta\ndata: {not-json}\n\n");
const wrongShape = await refusal("event: final\ndata: null\n\n");
console.log(JSON.stringify({events, final, missing, duplicate, after, malformed, wrongShape}));
"""
    )
    assert result["events"] == [["delta", "hello"]]
    assert result["final"] == {"ok": True}
    assert "exactly one final frame" in str(result["missing"])
    assert "exactly one final frame" in str(result["duplicate"])
    assert "continued after its final frame" in str(result["after"])
    assert "malformed JSON" in str(result["malformed"])
    assert "non-object event" in str(result["wrongShape"])


def test_chat_stops_overlaps_retries_exactly_and_ignores_the_old_request() -> None:
    """Cancellation reaches fetch, while a late callback cannot replace the retry's state."""
    result = run_javascript(
        r"""
class Element {
  constructor() { this.attrs = {}; this.listeners = {}; this.children = []; this.hidden = false;
    this.disabled = false; this.textContent = ""; this.value = ""; this.parentNode = this; }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k] || ""; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  appendChild(child) { this.children.push(child); this.firstElementChild ||= child; }
  insertBefore(child) { this.appendChild(child); }
  removeChild(child) { this.children = this.children.filter(item => item !== child); }
  querySelector(selector) { return (this.queries || {})[selector] || null; }
  querySelectorAll() { return []; }
  cloneNode() { return new Element(); }
}
const thread = new Element(); const form = new Element(); const turns = new Element();
const live = new Element(); live.hidden = true;
live.firstElementChild = new Element();
const answer = new Element(); const citations = new Element(); const verdict = new Element();
const rate = new Element(); const rated = new Element(); const status = new Element();
const profile = new Element(); profile.value = "precise"; const question = new Element();
const submit = new Element(); const stop = new Element(); stop.hidden = true;
const retry = new Element(); retry.hidden = true;
form.queries = {'[type="submit"]': submit, '[data-stop]': stop, '[data-retry]': retry};
thread.queries = {'[data-ask]': form, '[data-turns]': turns, '[data-live]': live,
  '[data-answer]': answer, '[data-citations]': citations, '[data-verdict]': verdict,
  '[data-rate]': rate, '[data-rated]': rated, '[data-status]': status,
  '[data-profile]': profile, '#question': question};
globalThis.document = {querySelector: selector => selector === '[data-chat]' ? thread : null,
  createElement: () => new Element()};
const pending = [];
globalThis.fetch = (path, options) => new Promise((resolve, reject) => {
  pending.push({path, options, resolve, reject});
});
hooks.startChat();
const submitEvent = {preventDefault() {}};
question.value = "exact original";
form.listeners.submit(submitEvent);
question.value = "must not overlap";
form.listeners.submit(submitEvent);
const callsBeforeStop = pending.length;
stop.listeners.click();
const aborted = pending[0].options.signal.aborted;
retry.listeners.click();
const sameBody = pending[0].options.body === pending[1].options.body;
pending[0].resolve({
  ok: false, status: 500, body: null, json: () => Promise.reject(new Error("late"))
});
await new Promise(resolve => setImmediate(resolve));
const staleIgnored = status.textContent === "Asking…" && submit.disabled;
const stream = new ReadableStream({start(controller) {
  const frame = 'event: final\ndata: ' +
    '{"ok":true,"data":{"confidence":null,"message_id":"m1"}}';
  controller.enqueue(new TextEncoder().encode(frame));
  controller.close();
}});
pending[1].resolve({ok: true, status: 200, body: stream, json: () => Promise.resolve({})});
await new Promise(resolve => setImmediate(resolve));
await new Promise(resolve => setImmediate(resolve));
const completed = status.textContent;
const turnsBeforeFailure = turns.children.length;
const preservedDraft = question.value;
question.value = "terminal failure";
form.listeners.submit(submitEvent);
const failed = new ReadableStream({start(controller) {
  const frame = 'event: final\ndata: ' +
    '{"ok":false,"error":{"type":"generation","message":"model refused"}}';
  controller.enqueue(new TextEncoder().encode(frame));
  controller.close();
}});
pending[2].resolve({ok: true, status: 200, body: failed, json: () => Promise.resolve({})});
await new Promise(resolve => setImmediate(resolve));
await new Promise(resolve => setImmediate(resolve));
const terminalStatus = status.textContent;
const retryOffered = !retry.hidden;
const disabled = submit.disabled;
question.value = "partial request";
form.listeners.submit(submitEvent);
const partial = new ReadableStream({start(controller) {
  controller.enqueue(new TextEncoder().encode(
    'event: delta\ndata: {"text":"unfinished words"}\n\n'
  ));
  controller.close();
}});
pending[3].resolve({ok: true, status: 200, body: partial, json: () => Promise.resolve({})});
await new Promise(resolve => setImmediate(resolve));
await new Promise(resolve => setImmediate(resolve));
const turnsBeforeFresh = turns.children.length;
question.value = "a fresh question";
form.listeners.submit(submitEvent);
const freshTurnDelta = turns.children.length - turnsBeforeFresh;
const partialWasDiscarded = answer.textContent === "";
const malformed = new ReadableStream({start(controller) {
  controller.enqueue(new TextEncoder().encode('event: delta\ndata: {not-json}\n\n'));
  controller.close();
}});
pending[4].resolve({ok: true, status: 200, body: malformed, json: () => Promise.resolve({})});
await new Promise(resolve => setImmediate(resolve));
await new Promise(resolve => setImmediate(resolve));
console.log(JSON.stringify({callsBeforeStop, aborted, sameBody, staleIgnored,
  callsAfterRetry: 2, completed, disabled,
  turnsBeforeFailure, preservedDraft,
  terminalStatus, retryOffered,
  freshTurnDelta, partialWasDiscarded,
  protocolStatus: status.textContent, protocolRetry: !retry.hidden}));
"""
    )
    assert result == {
        "callsBeforeStop": 1,
        "aborted": True,
        "sameBody": True,
        "staleIgnored": True,
        "callsAfterRetry": 2,
        "completed": "Answer complete.",
        "disabled": False,
        "turnsBeforeFailure": 1,
        "preservedDraft": "must not overlap",
        "terminalStatus": "generation: model refused",
        "retryOffered": True,
        "freshTurnDelta": 1,
        "partialWasDiscarded": True,
        "protocolStatus": "The answer stream contained malformed JSON. Retry the question.",
        "protocolRetry": True,
    }

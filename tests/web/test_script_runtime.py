"""The browser's asynchronous request states, executed rather than inferred from source."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

from manicule.web import rendering

_NODE = shutil.which("node")
if _NODE is None:
    msg = "the JavaScript runtime tests require node"
    raise RuntimeError(msg)
NODE: str = _NODE
SCRIPT = Path(rendering.HERE) / "static" / "manicule.js"
BOOT = r"""
const fs = require("fs");
const vm = require("vm");
let source = fs.readFileSync(process.argv[1], "utf8");
const start = "  startTheme();\n  startPalette();\n  startChat();\n  startActions();";
const expose = "  globalThis.hooks = { json: json, runAction: runAction, " +
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


def test_mutations_report_envelope_non_json_and_network_failures_and_serialize() -> None:
    """Every rejection is visible, and one shared status cannot host two racing actions."""
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

let release;
let calls = 0;
const held = new Promise(resolve => { release = resolve; });
const first = hooks.runAction(control, status, "Working…", () => {
  calls += 1;
  return held;
}, () => {});
await hooks.runAction(control, status, "Working…", () => {
  calls += 1;
  return Promise.resolve({status: 200, envelope: {ok: true}});
}, () => {});
release({status: 400, envelope: {ok: false, error: {type: "bad", message: "no"}}});
await first;
console.log(JSON.stringify({envelope, nonJson, network, calls}));
"""
    )
    assert result == {
        "envelope": {"message": "conflict: already exists", "role": "alert", "disabled": False},
        "nonJson": "The request was refused (502).",
        "network": "The service could not be reached. Try again.",
        "calls": 1,
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
console.log(JSON.stringify({events, final, missing, duplicate, after}));
"""
    )
    assert result["events"] == [["delta", "hello"]]
    assert result["final"] == {"ok": True}
    assert "exactly one final frame" in str(result["missing"])
    assert "exactly one final frame" in str(result["duplicate"])
    assert "continued after its final frame" in str(result["after"])


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
console.log(JSON.stringify({callsBeforeStop, aborted, sameBody, staleIgnored,
  callsAfterRetry: 2, completed, disabled: submit.disabled,
  turnsBeforeFailure, preservedDraft,
  terminalStatus: status.textContent, retryOffered: !retry.hidden}));
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
    }

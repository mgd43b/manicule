// The whole of what this extension does: read the cookie jar for one configured site and hand
// it to a manicule running on this machine.
//
// There are no content scripts in this extension and the manifest has no permission that would
// let one exist. It never sees a page, a form, a keystroke or a password — only cookies, for one
// origin the person named, through an API Chrome supports for exactly this. That is a stronger
// statement than manicule's driven-browser path can make, and it is the reason this exists.
//
// It talks to `chrome.runtime.sendNativeMessage`, which starts a short-lived local process that
// Chrome finds through a manifest naming this extension's id. There is no port, no localhost
// server and no token: the pairing is mutual and the operating system enforces it.

const HOST = "com.manicule.session_handoff";

// Cookies change constantly on a signed-in site — a session cookie is rewritten on most
// requests — so a send on every change would be a stream of hand-offs. This collapses a burst
// into one, and the delay is short enough that finishing a sign-in feels immediate.
const SETTLE_MS = 2000;

let pending = null;

async function configured() {
  const { baseUrl } = await chrome.storage.local.get("baseUrl");
  return baseUrl || "";
}

// The one call that reads anything. `chrome.cookies.getAll({url})` returns the cookies a request
// to that URL would carry, including `httpOnly` ones — which is why this works at all, since a
// Confluence session cookie is httpOnly and unreachable from page script.
//
// The jar is sent as-is rather than filtered here. The host re-filters to the configured
// authority with the same function manicule's other login paths use, and a filter in the
// extension would be a second implementation of a security rule that could drift from the first.
// What is sent is already scoped to one URL, so this is not the whole browser's cookies.
async function jarFor(baseUrl) {
  const cookies = await chrome.cookies.getAll({ url: baseUrl });
  return cookies.map((c) => ({
    name: c.name,
    value: c.value,
    domain: c.domain,
    path: c.path,
    secure: c.secure,
    expirationDate: c.expirationDate,
  }));
}

async function handOff(baseUrl) {
  const cookies = await jarFor(baseUrl);
  if (cookies.length === 0) {
    return { ok: false, error: "no cookies for that site yet — sign in to it in this browser" };
  }
  try {
    // `sendNativeMessage` starts the host, sends one message and reads one reply. A refusal
    // comes back as `{ok: false, error}` rather than as a dead port, because the host answers
    // instead of exiting — see its module docstring.
    return await chrome.runtime.sendNativeMessage(HOST, { base_url: baseUrl, cookies });
  } catch (e) {
    // The usual cause by a wide margin: the native host manifest is not installed, so Chrome
    // has nothing to start. Say the command rather than the exception.
    //
    // `e` is not necessarily an Error — a rejected port can surface as a string, and reading
    // `.message` off null throws inside the handler, which would replace a useful refusal with
    // a stack trace nobody sees.
    const reason = e instanceof Error ? e.message : String(e);
    return {
      ok: false,
      error: `could not reach manicule on this machine (${reason}). Run: manicule browser-auth install`,
    };
  }
}

async function remember(result) {
  await chrome.storage.local.set({
    lastResult: { ...result, at: new Date().toISOString() },
  });
}

// Re-authentication, which is the point of watching rather than only offering a button. When the
// person signs in to the wiki in their own browser — because their session expired, or because
// they signed in for their own reasons — the new cookies reach manicule without anybody running
// a command. That is the whole difference between this and every other login path.
chrome.cookies.onChanged.addListener(async (change) => {
  const baseUrl = await configured();
  if (!baseUrl) return;
  // `chrome.cookies.getAll({url})` decides relevance properly; this is only a cheap filter to
  // avoid waking on every cookie in the browser.
  let host;
  try {
    host = new URL(baseUrl).hostname;
  } catch {
    return;
  }
  const domain = change.cookie.domain.replace(/^\./, "");
  if (host !== domain && !host.endsWith(`.${domain}`)) return;

  if (pending) clearTimeout(pending);
  pending = setTimeout(async () => {
    pending = null;
    // A removal storm during sign-out should not be reported as a failure the person has to
    // read; only a jar that produces a held session is worth recording.
    const result = await handOff(baseUrl);
    if (result.ok) await remember(result);
  }, SETTLE_MS);
});

// The popup asks for these; nothing else can, because an extension's message port is not
// reachable from a page.
chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (message?.kind === "send") {
    configured()
      .then((baseUrl) =>
        baseUrl
          ? handOff(baseUrl)
          : { ok: false, error: "no site configured yet" }
      )
      .then(async (result) => {
        await remember(result);
        respond(result);
      });
    return true; // keeps the port open for the async reply
  }
  return false;
});

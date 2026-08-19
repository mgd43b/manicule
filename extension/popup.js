// The popup does three things: remember which site, ask Chrome for permission to read that
// site's cookies, and let the person send on demand. Everything else happens in the worker.

const site = document.getElementById("site");
const status = document.getElementById("status");

function show(text, kind) {
  status.textContent = text;
  status.className = kind || "muted";
}

chrome.storage.local.get(["baseUrl", "lastResult"]).then(({ baseUrl, lastResult }) => {
  if (baseUrl) site.value = baseUrl;
  if (lastResult) {
    show(
      lastResult.ok
        ? `Last handoff: ${lastResult.connector} as ${lastResult.account}`
        : `Last attempt failed: ${lastResult.error}`,
      lastResult.ok ? "ok" : "bad"
    );
  }
});

document.getElementById("save").addEventListener("click", async () => {
  let url;
  try {
    url = new URL(site.value.trim());
  } catch {
    return show("That is not a URL.", "bad");
  }
  // Requested at runtime for one origin rather than declared in the manifest, so Chrome's own
  // dialog names the site and the extension holds no standing access to anything else.
  const origin = `${url.origin}/*`;
  const granted = await chrome.permissions.request({ origins: [origin] });
  if (!granted) return show("Chrome declined access to that site.", "bad");
  await chrome.storage.local.set({ baseUrl: site.value.trim() });
  show(`Watching ${url.origin}. Sign in there and manicule will follow.`, "ok");
});

document.getElementById("send").addEventListener("click", () => {
  show("Sending…");
  chrome.runtime.sendMessage({ kind: "send" }, (result) => {
    if (!result) return show("The extension did not answer.", "bad");
    show(
      result.ok ? `Held for ${result.connector} as ${result.account}` : result.error,
      result.ok ? "ok" : "bad"
    );
  });
});

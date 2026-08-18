# Release notes

## Unreleased

### Rule-driven collection management

Collection rules are now available through the application service, CLI, HTTP API, control
socket, and writable MCP server. A collection can select documents by source, media type, tag,
or update bounds when it is created, and its rule can later be shown, replaced, or cleared.

Existing indexes can adopt these rules immediately. Membership remains evaluated at read time,
so matching documents already in the workspace and matching documents ingested later appear
without reconciliation. Rule management does not fetch sources, enumerate the corpus, ingest
documents, rebuild chunks, or re-embed content. Manual membership remains unioned with the rule,
and clearing a rule preserves those manual members.

"""Root-page scoping: which pages a configured page tree contains, and how that is decided.

A space is often far more than anybody wants indexed. This is the narrowing:
``root_page_ids`` names one or more pages, and the source becomes those pages and everything
currently beneath them. Everything else in the space is not merely filtered out — it is not
asked for.

Four decisions carry the design, and each is here rather than in the connector because each is
about *scope* rather than about syncing.

**The source does the traversal.** ``ancestor`` is Confluence's own descendant predicate and it
matches at any depth, so there is no client-side walk: no queue of page ids, no cycle
detection, no depth ceiling, and no second enumeration that a moved page can fall between. A
tree deeper than any bound this connector could have chosen is one query, and a cycle — which
Confluence does not permit but a client walking ``child/page`` would still have to survive —
cannot arise, because nothing here follows a link.

**What came back is checked against what was asked for.** Every page carries its own ancestor
ids in the same response that returned it, so membership is re-derived from the page rather
than inferred from the fact that the source returned it. That check is not paranoia about
Confluence; it is what stops the one failure that would be invisible. A deployment that did not
honor the predicate would answer with the whole space, and a connector that trusted the query
would index all of it while reporting a subtree — the exact outcome that makes a scoped sync
worth having. Here it is a refusal naming the page.

**An empty answer is not an empty subtree.** Reconciliation deletes what it does not see, so
"this tree has no descendants" is the most dangerous sentence this module can say. It is
therefore never said on the strength of one query: an enumeration that finds no descendants is
cross-checked against ``child/page`` on each root, and a root that has children the enumeration
did not find stops the run instead of emptying the index.

**Attachments are scoped through their page, not through their own ancestry.** Confluence
exposes an attachment's container and this module trusts nothing else about its position, so an
attachment is in scope exactly when the page holding it is. The cost of that is stated where an
operator can read it (``docs/connectors/confluence.md`` §2.1): the attachment enumeration stays
space-wide, because there is no descendant predicate for attachments worth relying on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from manicule.connectors import cql
from manicule.connectors.client import ConfluenceClient
from manicule.connectors.config import ConfluenceConfig
from manicule.connectors.errors import ConnectorError, NotFoundError

__all__ = [
    "CONTENT_PATH",
    "SEARCH_PATH",
    "RootPage",
    "Subtree",
    "ancestor_ids",
    "resolve",
]

SEARCH_PATH = "/rest/api/content/search"
"""CQL search. One home for the path, shared with the connector that enumerates through it."""

CONTENT_PATH = "/rest/api/content"

_CURRENT = "current"
_PAGE = "page"


@dataclass(frozen=True, slots=True)
class RootPage:
    """One configured root, as the source describes it rather than as configuration spelled it.

    The id is echoed back from the source rather than reused from configuration, because
    validation's job is to establish that this page exists and where it lives; carrying the
    configured string forward would mean the rest of the run trusted a number nobody confirmed.
    """

    id: str
    space: str
    title: str


class Subtree:
    """The pages one run may see, and which configured root put each of them there.

    Built once per enumeration and thrown away, so that discovery and reconciliation each ask
    the source what is in scope *now* rather than sharing a snapshot taken at some other time.
    """

    def __init__(
        self,
        client: ConfluenceClient,
        config: ConfluenceConfig,
        roots: Sequence[RootPage],
        *,
        source: str,
    ) -> None:
        self._client = client
        self._config = config
        self._source = source
        self._roots = tuple(roots)
        by_space: dict[str, list[RootPage]] = {}
        for root in self._roots:
            by_space.setdefault(root.space, []).append(root)
        self._by_space = {space: tuple(found) for space, found in by_space.items()}
        self._members: dict[str, Mapping[str, tuple[str, ...]]] = {}

    # --- what the queries need -----------------------------------------------------------

    def spaces(self) -> tuple[str, ...]:
        """The spaces holding a configured root, in the order the roots were configured."""
        return tuple(self._by_space)

    def roots_in(self, space: str) -> tuple[str, ...]:
        return tuple(root.id for root in self._by_space.get(space, ()))

    def clause(self, space: str) -> str:
        """The CQL narrowing one space's page query to the roots configured in it."""
        return cql.subtree_clause(
            self.roots_in(space), include_roots=self._config.include_root_pages
        )

    # --- membership ----------------------------------------------------------------------

    def covering_roots(
        self, space: str, page_id: str, ancestor_ids: Sequence[str]
    ) -> tuple[str, ...]:
        """Which configured roots put this page in scope, or ``()`` for a page that is not.

        The whole of "is this page in scope", in the order somebody debugging it would ask:

        1. Is it one of the configured roots, and are roots included?
        2. Is a configured root among its ancestors?

        Nothing else. In particular there is no third rule about how the page was reached,
        because the answer must not depend on which query returned it — discovery's watermarked
        query and reconciliation's full one have to agree, and they agree by asking this.
        """
        roots = self.roots_in(space)
        if not roots:
            return ()
        if page_id in roots:
            return (page_id,) if self._config.include_root_pages else ()
        seen = set(ancestor_ids)
        return tuple(root for root in roots if root in seen)

    async def members(self, space: str) -> Mapping[str, tuple[str, ...]]:
        """Every in-scope page id in ``space``, mapped to the roots that put it there.

        A full enumeration, ids and ancestor ids only, bounded by the subtree rather than by
        the space — which is the premise of the whole feature and the reason this is affordable
        where a space-wide equivalent would not be.

        Cached for the lifetime of this object, which is one enumeration. It is what an
        attachment's scope is decided against, and what reconciliation reports as still
        existing.

        Raises:
            ConnectorError: The source returned a page outside the configured trees, or
                returned no descendants for a root that demonstrably has children.
        """
        if space in self._members:
            return self._members[space]

        found: dict[str, tuple[str, ...]] = {}
        params = [
            ("cql", cql.content_query(space, types=(_PAGE,), subtree=self.clause(space))),
            ("limit", str(self._config.page_size)),
            ("expand", "ancestors"),
        ]
        async for payload in self._client.paginate(self._client.url(SEARCH_PATH), params):
            for result in _results(payload):
                page_id = _str(result.get("id"))
                if not page_id:
                    continue
                covering = self.covering_roots(space, page_id, ancestor_ids(result))
                if not covering:
                    raise ConnectorError(self.out_of_scope(space, page_id))
                found[page_id] = covering

        await self._check_not_falsely_empty(space, found)
        self._members[space] = found
        return found

    async def _check_not_falsely_empty(
        self, space: str, found: Mapping[str, tuple[str, ...]]
    ) -> None:
        """Refuse an empty descendant set that a second, independent question contradicts.

        This is the guard between "the tree really is one page" and "the descendant predicate
        did nothing", and the two are indistinguishable from the enumeration alone — both are a
        query that succeeded and returned no children. They are not indistinguishable in
        consequence: the second reports the entire subtree as gone, and reconciliation deletes
        what it does not see.

        So when nothing but the roots came back, each root is asked whether it has a child
        through a different endpoint. One extra request per root, on the one run where the
        answer decides whether an index is emptied.
        """
        descendants = set(found) - {root.id for root in self._by_space.get(space, ())}
        if descendants:
            return
        for root in self._by_space.get(space, ()):
            if not await self._has_child_page(root.id):
                continue
            msg = (
                f"source {self._source!r}: root page {root.id} in space {space} has child "
                f"pages, and the descendant query for it returned none. The subtree is not "
                f"empty, so something about `ancestor` on this deployment is not doing what "
                f"this connector asked. Stopping here: an empty answer taken at face value "
                f"would report the whole subtree as deleted at the next reconciliation."
            )
            raise ConnectorError(msg)

    async def _has_child_page(self, page_id: str) -> bool:
        """Whether a page has at least one child, asked without the predicate under suspicion."""
        url = f"{self._client.url(CONTENT_PATH)}/{page_id}/child/page"
        try:
            payload = await self._client.get_json(url, [("limit", "1")])
        except NotFoundError:
            # The root was validated at the start of this run, so a 404 here is the endpoint
            # rather than the page. Nothing is concluded from it: the guard exists to turn a
            # positive answer into a refusal, and it has not got one.
            return False
        return bool(_results(payload))

    def out_of_scope(self, space: str, page_id: str) -> str:
        """Why a page the source returned is being refused rather than filtered away.

        Takes the id rather than the result it came in. A refusal is read in logs and a page
        title is content, so the response is not a thing this message should be holding at all.
        """
        roots = ", ".join(self.roots_in(space))
        excluded = (
            " That page is a configured root, and include_root_pages is false, so this run did "
            "not ask for it either."
            if page_id in self.roots_in(space)
            else ""
        )
        return (
            f"source {self._source!r}: the search for the page tree(s) {roots} in space "
            f"{space} returned page {page_id}, which is not in them.{excluded} The query "
            f"narrows at the source, so a page it did not ask for means this deployment did "
            f"not apply the narrowing — and filtering the difference away here would index a "
            f"subtree while paying for, and claiming, the whole space."
        )


async def resolve(
    client: ConfluenceClient,
    config: ConfluenceConfig,
    allowed_spaces: Sequence[str],
    *,
    source: str,
) -> Subtree:
    """Validate every configured root and build the scope for this run.

    Runs before anything is enumerated, because every failure here is one that reconciliation
    would otherwise turn into deletions: a root that is missing, that this account cannot see,
    or that sits outside the space allowlist all produce an empty or truncated subtree, and an
    empty subtree diffed against the index is the index.

    ``allowed_spaces`` is the already-checked space allowlist, or every visible space when
    there is none. Passing it in rather than re-deriving it is what makes the two settings one
    scope: a root is checked against the same list the rest of the run uses.

    Raises:
        ConnectorError: A root is missing, invisible, not a current page, outside the
            allowlist, or the source could not say which space it is in. Also when a listed
            space contains no configured root, which is a configuration that cannot be honored
            as written rather than one with an obvious reading.
    """
    roots: list[RootPage] = []
    for configured in config.root_page_ids:
        roots.append(await _validated(client, configured, source=source))

    allowed = {space.casefold(): space for space in allowed_spaces}
    # Always, allowlist or not. With one, this is the allowlist being an allowlist; without
    # one, `allowed_spaces` is every space the account can see, and a root in a space that is
    # not among them is a root whose space this account has lost — which would otherwise be a
    # query against a space nobody checked, returning nothing, for the same reason a mistyped
    # space key would.
    _refuse_roots_outside(roots, allowed, listed=bool(config.spaces), source=source)
    if config.spaces:
        # Only with an allowlist. "Every visible space must contain a configured root" would
        # be a rule about somebody else's Confluence.
        _refuse_spaces_without_roots(roots, allowed, source=source)
    return Subtree(client, config, roots, source=source)


async def _validated(client: ConfluenceClient, page_id: str, *, source: str) -> RootPage:
    """One root, as the source describes it.

    Asked with no body expansion: this establishes existence, status and space, and a root page
    is fetched later like any other page if it is in scope.
    """
    url = f"{client.url(CONTENT_PATH)}/{page_id}"
    try:
        payload = await client.get_json(url, [("expand", "space")])
    except NotFoundError as exc:
        msg = (
            f"source {source!r}: configured root page {page_id} is not there. Either no page "
            f"has that id, or the account this source syncs as cannot see it — Confluence "
            f"answers both the same way, and neither is something to sync around. Nothing was "
            f"enumerated, so nothing will be proposed for deletion."
        )
        raise ConnectorError(msg) from exc

    status = _str(payload.get("status"))
    if status and status != _CURRENT:
        msg = (
            f"source {source!r}: configured root page {page_id} has status {status!r} rather "
            f"than {_CURRENT!r}. A trashed or draft page has no current descendants to sync, "
            f"and treating it as a root would enumerate an empty subtree."
        )
        raise ConnectorError(msg)

    kind = _str(payload.get("type"))
    if kind and kind != _PAGE:
        msg = (
            f"source {source!r}: configured root page {page_id} is a {kind!r} rather than a "
            f"page. Only a page has descendants, so this is a root that can never contain "
            f"anything."
        )
        raise ConnectorError(msg)

    space = _str(_obj(payload.get("space")).get("key"))
    if not space:
        msg = (
            f"source {source!r}: the source did not say which space configured root page "
            f"{page_id} is in, so this run cannot check it against the configured spaces or "
            f"scope a query to it."
        )
        raise ConnectorError(msg)

    resolved = _str(payload.get("id")) or page_id
    return RootPage(id=resolved, space=space, title=_str(payload.get("title")))


def _refuse_roots_outside(
    roots: Iterable[RootPage], allowed: Mapping[str, str], *, listed: bool, source: str
) -> None:
    """A root outside the spaces this source may read is a refusal, not a quiet widening.

    Two readings, one check. With an allowlist, ``spaces`` says which spaces this source may
    read, and honoring a root outside it would mean the allowlist had stopped being one the
    moment somebody added a page id. Without one, the same list is every space the account can
    see, and a root outside it sits in a space this account cannot read at all — a query
    against which returns nothing, indistinguishable from a subtree that has been emptied.
    """
    outside = [root for root in roots if root.space.casefold() not in allowed]
    if not outside:
        return
    named = ", ".join(f"{root.id} (in {root.space})" for root in sorted(outside, key=_by_id))
    available = ", ".join(sorted(allowed.values())) or "none"
    why = (
        (
            f"the `spaces` allowlist does not include. Allowed: {available}. `spaces` and "
            f"`root_page_ids` narrow one scope between them rather than adding to each other, "
            f"so this is refused instead of resolved: either add the space to `spaces`, or "
            f"remove the root."
        )
        if listed
        else (
            f"this account cannot see. Visible: {available}. A query against a space that is "
            f"not there returns nothing rather than an error, so this would be a sync that "
            f"appears to work and a reconciliation that proposes emptying the subtree."
        )
    )
    msg = f"source {source!r}: configured root page(s) {named} are in space(s) {why}"
    raise ConnectorError(msg)


def _refuse_spaces_without_roots(
    roots: Iterable[RootPage], allowed: Mapping[str, str], *, source: str
) -> None:
    """A listed space with no configured root would contribute nothing, in silence.

    The same failure the space allowlist check already refuses, arriving by a different route:
    a space that is configured and syncs nothing looks exactly like a space that is empty, and
    reconciliation then proposes deleting everything it ever contributed.
    """
    covered = {root.space.casefold() for root in roots}
    barren = sorted(name for key, name in allowed.items() if key not in covered)
    if not barren:
        return
    msg = (
        f"source {source!r}: space(s) {', '.join(barren)} are in `spaces` but contain none of "
        f"the configured `root_page_ids`, so this source would enumerate nothing from them "
        f"while appearing to sync them. Give each listed space a root, or remove it from "
        f"`spaces`."
    )
    raise ConnectorError(msg)


def _by_id(root: RootPage) -> str:
    return root.id


def _results(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    entries = cast("Sequence[object]", results)
    return [cast("Mapping[str, object]", entry) for entry in entries if isinstance(entry, dict)]


def ancestor_ids(result: Mapping[str, object]) -> tuple[str, ...]:
    """Ancestor content ids, outermost first. The page's own id is not among them."""
    ancestors = result.get("ancestors")
    if not isinstance(ancestors, list):
        return ()
    entries = cast("Sequence[object]", ancestors)
    ids = [_str(_obj(entry).get("id")) for entry in entries]
    return tuple(found for found in ids if found)


def _obj(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, dict) else {}


def _str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""

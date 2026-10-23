"""Synthetic mail threads with realistic quoting, and the ground truth of what each one repeats.

The corpus :mod:`tests.benchmarks.thread_duplication` measures. Everything is generated from a
seed: invented people at ``example.org``, invented projects, invented decisions. Nothing here is
or resembles real mail.

**Every message authors exactly one passage**, a sentence or two stating one decision, and the
generator records which message wrote it. That record is the ground truth the measurement scores
against. It is never inferred from what retrieval returned, because a duplicate is a fact about
the corpus, not about a ranking.

Three quoting styles, because they repeat text differently:

``top``
    Top-posted, full history. Each reply is new text above ``On …, … wrote:`` and the whole
    previous message, every line prefixed with ``>``, so nesting deepens by one level per reply
    and the first message is repeated in every later one. What most clients do by default.

``inline``
    Interleaved and trimmed. Each reply quotes only the passage it answers and the one before
    it, and writes beneath them. Each passage appears in at most three messages.

``html``
    HTML-only, in the shape a webmail client writes: the reply in a ``div``, the history in
    nested ``blockquote class="gmail_quote"``. The mail parser renders an HTML-only body through
    the HTML converter, which keeps the quoted text and drops the blockquote — so a quoted copy
    arrives with no ``>`` marker at all and is textually the original.

Beside the threads: a **companion** note for every passage a query asks about — a separate
message, from someone else, restating the same decision in other words. It is a second relevant
document outside the thread, so a page filled with quotes of the original has a measurable
victim. And **distractor** messages on other projects, so a top ten is not filled by one thread
merely because nothing else exists.
"""

from __future__ import annotations

import html
import random
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from email.policy import SMTP
from email.utils import format_datetime

STYLES = ("top", "inline", "html")
"""The quoting styles, in report order."""

LENGTHS = (2, 5, 10, 20, 30)
"""Messages per thread. Thirty is long for mail and still ordinary for a busy project thread."""

CODENAMES = (
    "aurora", "basalt", "cinder", "delta", "ember", "fathom", "glacier", "harbor", "isotope",
    "juniper", "krypton", "lantern", "mistral", "nimbus", "obsidian", "pumice", "quartz",
    "ravine", "sierra", "tundra", "umber", "vellum", "willow", "xenon", "yarrow", "zephyr",
    "anvil", "bramble", "cobalt", "dune", "eskar", "fennel", "garnet", "heron", "indigo",
    "jasper", "kestrel", "lichen", "marlin", "nettle", "onyx", "plover", "quill", "russet",
    "saffron", "thistle", "upland", "vireo", "wren", "yucca",
)  # fmt: skip
"""Fifty project codenames, no two sharing a word."""

SYSTEMS = (
    "ledger", "gateway", "scheduler", "warehouse", "indexer", "router", "archive", "notifier",
    "planner", "cache", "balancer", "collector", "compactor", "resolver", "broker", "migrator",
    "allocator", "sequencer", "transcoder", "reconciler", "dashboard", "exporter", "importer",
    "renderer", "crawler", "mailer", "tracker", "registry", "uploader", "verifier", "auditor",
    "billing service", "search portal", "payroll run", "inventory feed", "license server",
    "ticket queue", "build farm", "data lake", "voice bridge", "print spooler", "kiosk fleet",
    "badge system", "survey tool", "help center", "parts catalog", "booking engine",
    "fleet telemetry", "pricing engine", "loyalty program",
)  # fmt: skip
"""What each codename is the name of. Paired by position, so every topic is distinct."""

PEOPLE = (
    ("Pat Rivera", "pat"),
    ("Sam Okafor", "sam"),
    ("Lee Brandt", "lee"),
    ("Ana Cho", "ana"),
    ("Ravi Mehta", "ravi"),
)

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
MONTHS = ("March", "April", "May", "June", "September", "October")
TEAMS = ("platform", "finance", "support", "security", "data", "facilities")
VENDORS = ("Northwind", "Contoso", "Fabrikam", "Tailspin", "Litware", "Adventure Works")

ASPECTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "We agreed the retry budget for the {topic} is {n} attempts with a {m} second backoff "
        "between them.",
        "Note for the runbook: {topic} retries stop after {n} tries, waiting {m} seconds each "
        "time.",
        "how many retries does the {topic} make before giving up",
        "{topic} retry budget",
    ),
    (
        "The {topic} cutover moves to {day} the {d}th of {month} because the {vendor} audit "
        "slipped.",
        "Calendar update: switching over to the new {topic} now happens on {month} {d}, a "
        "{day}, after the {vendor} review was delayed.",
        "when is the {topic} cutover scheduled",
        "{topic} cutover",
    ),
    (
        "Ownership of the {topic} passes to the {team} team from next quarter, with {name} as "
        "the escalation contact.",
        "From next quarter the {team} group owns the {topic}; escalations go to {name}.",
        "which team will own the {topic}",
        "{topic} ownership escalation",
    ),
    (
        "The {topic} keeps logs for {n} days and then archives them to cold storage for {m} "
        "months.",
        "Retention for the {topic}: {n} days hot, then {m} months archived.",
        "how long does the {topic} retain its logs",
        "{topic} logs archives",
    ),
    (
        "We will cap the {topic} at {n} thousand requests per minute until the {vendor} "
        "contract is renewed.",
        "Rate limit reminder: the {topic} stays at {n}k requests a minute pending the {vendor} "
        "renewal.",
        "what is the rate limit on the {topic}",
        "{topic} requests per minute",
    ),
    (
        "Budget for the {topic} this year is {n} hundred thousand dollars, of which {m} percent "
        "is hardware.",
        "Finance summary: {topic} spend is set at {n}00k for the year, {m}% of it on equipment.",
        "what is the annual budget for the {topic}",
        "{topic} budget hardware",
    ),
    (
        "The rollback window for the {topic} is {n} hours, after which we roll forward instead "
        "of back.",
        "If a {topic} release goes wrong we have {n} hours to revert; past that we fix forward.",
        "how long is the rollback window for the {topic}",
        "{topic} rollback window",
    ),
    (
        "On call for the {topic} rotates weekly on {day} mornings, and the {team} team covers "
        "holidays.",
        "Pager handoff for the {topic} is every {day} morning; {team} takes the holiday shifts.",
        "when does the {topic} on call rotation change over",
        "{topic} call rotates",
    ),
    (
        "The {topic} needs {n} replicas in each region to survive losing a whole availability "
        "zone.",
        "Capacity note: run {n} copies of the {topic} per region so a zone outage is survivable.",
        "how many replicas does the {topic} run per region",
        "{topic} replicas region",
    ),
    (
        "Backups of the {topic} run every {n} hours and a restore drill happens on the first "
        "{day} of each month.",
        "The {topic} is backed up each {n} hours, with a monthly restore test on the first {day}.",
        "how often is the {topic} backed up",
        "{topic} restore drill",
    ),
    (
        "Support for the old {topic} client ends in {month}, and {name} is writing the "
        "deprecation notice.",
        "The legacy {topic} client is retired in {month}; {name} owns the announcement.",
        "when does support for the old {topic} client end",
        "{topic} old client",
    ),
    (
        "The {topic} will be hosted by {vendor} under a {n} year agreement with a {m} percent "
        "discount.",
        "Hosting decision: {vendor} runs the {topic} on a {n} year deal at {m}% off list.",
        "which vendor hosts the {topic}",
        "{topic} hosted agreement",
    ),
    (
        "Latency target for the {topic} is {n} milliseconds at the ninety ninth percentile, "
        "measured at the load balancer.",
        "The {topic} must answer within {n} ms at p99, timed at the balancer.",
        "what is the latency target for the {topic}",
        "{topic} latency target",
    ),
    (
        "Access to the {topic} admin console is limited to the {team} team and requires a "
        "hardware key.",
        "Only {team} may use the {topic} admin console, and only with a security key.",
        "who can access the {topic} admin console",
        "{topic} admin console",
    ),
    (
        "The {topic} migration is split into {n} waves, the first covering the {team} accounts.",
        "Plan: move the {topic} in {n} batches, starting with {team}.",
        "how many waves is the {topic} migration split into",
        "{topic} migration waves",
    ),
    (
        "Error reports from the {topic} go to a shared inbox that {name} triages every {day}.",
        "{name} reviews the {topic} error inbox each {day}.",
        "who triages error reports from the {topic}",
        "{topic} error reports inbox",
    ),
    (
        "The {topic} disaster recovery site is in the {vendor} facility and must be live within "
        "{n} hours.",
        "Recovery for the {topic} fails over to {vendor} with a {n} hour objective.",
        "where is the disaster recovery site for the {topic}",
        "{topic} disaster recovery",
    ),
    (
        "Maintenance on the {topic} is allowed on {day} nights between {n} and {m} in the morning.",
        "The {topic} maintenance window is {day} night, {n} to {m} a.m.",
        "when is the maintenance window for the {topic}",
        "{topic} maintenance nights",
    ),
    (
        "The {topic} encryption keys rotate every {n} days and the old key stays valid for {m} "
        "days after.",
        "Key rotation for the {topic}: new key each {n} days, previous key honored {m} more days.",
        "how often do the {topic} encryption keys rotate",
        "{topic} encryption keys",
    ),
    (
        "Training on the new {topic} is mandatory for the {team} team before the end of {month}.",
        "Everyone in {team} has to finish {topic} training by the close of {month}.",
        "who must complete training on the {topic}",
        "{topic} training mandatory",
    ),
    (
        "The {topic} license covers {n} hundred seats and renews each {month}.",
        "Seat count for the {topic} is {n}00, renewing in {month}.",
        "how many seats does the {topic} license cover",
        "{topic} license seats",
    ),
    (
        "We are dropping the nightly {topic} report in favor of a weekly summary sent on {day}.",
        "The {topic} report is weekly now, out every {day}, replacing the nightly one.",
        "how often is the {topic} report sent",
        "{topic} weekly summary",
    ),
    (
        "Incidents on the {topic} are reviewed within {n} business days, and the review is "
        "chaired by {name}.",
        "{name} chairs {topic} incident reviews, held within {n} working days.",
        "who chairs incident reviews for the {topic}",
        "{topic} incidents chaired",
    ),
    (
        "The {topic} test environment is rebuilt from scratch every {day} so drift cannot "
        "accumulate.",
        "Every {day} the {topic} staging environment is torn down and recreated.",
        "how often is the {topic} test environment rebuilt",
        "{topic} test environment rebuilt",
    ),
    (
        "Data in the {topic} is classified as confidential, so exports need sign off from {name}.",
        "Exports from the {topic} require {name} to approve, since the data is confidential.",
        "who approves data exports from the {topic}",
        "{topic} exports sign off",
    ),
    (
        "The {topic} alert threshold is {n} percent error rate sustained for {m} minutes.",
        "Alerting on the {topic} fires at {n}% errors held for {m} minutes.",
        "what error rate triggers an alert on the {topic}",
        "{topic} alert threshold",
    ),
    (
        "Customer notices about {topic} downtime go out {n} days ahead through the {team} team.",
        "Planned {topic} outages are announced to customers {n} days in advance by {team}.",
        "how far ahead are customers told about {topic} downtime",
        "{topic} downtime notices",
    ),
    (
        "The {topic} dashboard refreshes every {n} minutes and keeps {m} weeks of history.",
        "Dashboards for the {topic} update each {n} minutes with {m} weeks retained.",
        "how often does the {topic} dashboard refresh",
        "{topic} dashboard refreshes",
    ),
    (
        "Procurement approved {n} new servers for the {topic}, arriving from {vendor} in {month}.",
        "{vendor} ships the {n} additional {topic} servers in {month}.",
        "when do the new servers for the {topic} arrive",
        "{topic} procurement servers",
    ),
    (
        "The {topic} code freeze starts on the {d}th of {month} and lasts {n} days.",
        "No {topic} changes from {month} {d} for {n} days.",
        "when does the {topic} code freeze start",
        "{topic} code freeze",
    ),
    (
        "Load testing of the {topic} showed it saturates at {n} thousand concurrent sessions.",
        "The {topic} tops out around {n}k simultaneous sessions under load test.",
        "how many concurrent sessions can the {topic} handle",
        "{topic} load testing saturates",
    ),
    (
        "The {topic} audit trail is append only and is reviewed by {vendor} every {month}.",
        "{vendor} reviews the immutable {topic} audit log each {month}.",
        "who reviews the {topic} audit trail",
        "{topic} audit trail",
    ),
    (
        "Feature flags for the {topic} are cleaned up after {n} weeks, and {name} owns the list.",
        "{name} prunes {topic} feature flags once they are {n} weeks old.",
        "who owns the feature flag cleanup for the {topic}",
        "{topic} feature flags",
    ),
    (
        "The {topic} sends its metrics to the {team} observability stack at {n} second resolution.",
        "{topic} metrics flow into the {team} monitoring system every {n} seconds.",
        "where does the {topic} send its metrics",
        "{topic} metrics observability",
    ),
)
"""Per kind of decision: the statement, a companion's rewording, a question and a keyword lookup.

Thirty-four of them, so a thirty-message thread never states two of the same kind. Every word
of the keyword lookup appears in the statement, because the lexical leg matches a chunk only when
it holds every query term — a natural question rarely reaches it at all. Plain words only — no
apostrophes, emphasis or list markers — so the HTML converter cannot change a passage's
characters and the ground truth can locate every copy exactly.
"""

GREETINGS = ("Hi {to}", "Thanks {to}", "{to}", "Hello {to}")
SIGNOFFS = ("Thanks", "Cheers", "Best", "Regards")
EPOCH = datetime(2026, 2, 2, 9, 0, tzinfo=UTC)
WIDTH = 72


@dataclass(frozen=True)
class Passage:
    """One authored statement, the message that wrote it, and a question it answers."""

    id: str
    text: str
    query: str
    """A natural question, phrased as somebody would ask it."""
    keywords: str
    """A few words, every one of them in :attr:`text`."""
    companion: str


@dataclass
class Message:
    """One message as the generator built it, and every passage its body carries."""

    key: str
    raw: bytes
    own: Passage
    carries: tuple[Passage, ...]
    """Its own passage first, then every quoted one, as the generator put them there."""
    thread: Thread | None = None
    gmail_id: str = ""


@dataclass
class Thread:
    """A conversation: one topic, one quoting style, ``len(messages)`` replies."""

    key: str
    style: str
    topic: str
    messages: list[Message] = field(default_factory=list[Message])
    queried: tuple[int, ...] = ()
    """Positions whose passage a query asks about: the first, the middle and the last."""


@dataclass
class Corpus:
    """Everything indexed, and the questions asked of it."""

    threads: list[Thread]
    companions: dict[str, Message]
    """Keyed by the passage id they restate."""
    distractors: list[Message]

    @property
    def messages(self) -> list[Message]:
        found = [message for thread in self.threads for message in thread.messages]
        return [*found, *self.companions.values(), *self.distractors]


def _values(rng: random.Random) -> dict[str, object]:
    name, _ = rng.choice(PEOPLE)
    return {
        "n": rng.randint(3, 48),
        "m": rng.randint(5, 90),
        "d": rng.randint(4, 28),
        "day": rng.choice(DAYS),
        "month": rng.choice(MONTHS),
        "team": rng.choice(TEAMS),
        "vendor": rng.choice(VENDORS),
        "name": name,
    }


def _passage(
    key: str, template: tuple[str, str, str, str], topic: str, rng: random.Random
) -> Passage:
    statement, companion, query, keywords = template
    values = {"topic": topic, **_values(rng)}
    return Passage(
        id=key,
        text=statement.format(**values),
        query=query.format(**values),
        keywords=keywords.format(**values),
        companion=companion.format(**values),
    )


def quote(text: str) -> str:
    """Prefix every line with ``>`` the way a webmail client's plain-text part does.

    A line already quoted gains a bare ``>``, so nesting reads ``>>``, ``>>>``; an empty line
    becomes a bare ``>`` so the quoted block stays one block.
    """
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            lines.append(f">{line}")
        elif line:
            lines.append(f"> {line}")
        else:
            lines.append(">")
    return "\n".join(lines)


def _wrap(text: str) -> str:
    return textwrap.fill(text, WIDTH)


def _attribution(when: datetime, name: str, user: str) -> str:
    stamp = when.strftime("%a, %b %d, %Y at %H:%M")
    return f"On {stamp} {name} <{user}@example.org> wrote:"


def _letter(greeting: str, passage: Passage, signoff: str, sender: str) -> str:
    return f"{greeting},\n\n{_wrap(passage.text)}\n\n{signoff},\n{sender.split(maxsplit=1)[0]}"


def _html_letter(greeting: str, passage: Passage, signoff: str, sender: str) -> str:
    return (
        f'<div dir="ltr"><div>{html.escape(greeting)},</div><div><br></div>'
        f"<div>{html.escape(passage.text)}</div><div><br></div>"
        f"<div>{signoff},<br>{html.escape(sender.split(maxsplit=1)[0])}</div></div>"
    )


def _envelope(
    *,
    subject: str,
    sender: tuple[str, str],
    to: tuple[str, str],
    when: datetime,
    message_id: str,
    parent: str | None,
    references: list[str],
) -> EmailMessage:
    built = EmailMessage()
    built["From"] = f"{sender[0]} <{sender[1]}@example.org>"
    built["To"] = f"{to[0]} <{to[1]}@example.org>"
    built["Subject"] = subject
    built["Date"] = format_datetime(when)
    built["Message-ID"] = message_id
    if parent is not None:
        built["In-Reply-To"] = parent
        built["References"] = " ".join(references)
    return built


def build_thread(
    key: str, *, style: str, length: int, topic: str, rng: random.Random, epoch: datetime
) -> Thread:
    """One thread of ``length`` messages on ``topic``, quoted in ``style``."""
    thread = Thread(key=key, style=style, topic=topic)
    templates = rng.sample(ASPECTS, length)
    passages = [
        _passage(f"{key}.m{position:02d}", template, topic, rng)
        for position, template in enumerate(templates)
    ]
    subject = f"{topic.capitalize()} planning"
    cast = rng.sample(PEOPLE, 3)
    bodies: list[str] = []
    ids: list[str] = []
    for position, passage in enumerate(passages):
        sender = cast[position % len(cast)]
        to = cast[(position + 1) % len(cast)]
        when = epoch + timedelta(hours=3 * position)
        greeting = rng.choice(GREETINGS).format(to=to[0].split(maxsplit=1)[0])
        signoff = rng.choice(SIGNOFFS)
        parent = ids[-1] if ids else None
        message_id = f"<{key}.{position:02d}@example.org>"
        envelope = _envelope(
            subject=subject if position == 0 else f"Re: {subject}",
            sender=sender,
            to=to,
            when=when,
            message_id=message_id,
            parent=parent,
            references=list(ids),
        )
        previous = cast[(position - 1) % len(cast)]
        attribution = _attribution(when - timedelta(hours=3), *previous)
        if style == "top":
            body = _letter(greeting, passage, signoff, sender[0])
            if position:
                body = f"{body}\n\n{attribution}\n\n{quote(bodies[-1])}"
            carries = passages[position::-1]
            envelope.set_content(body + "\n")
        elif style == "inline":
            core = _wrap(passages[position - 1].text) if position else ""
            if position >= 2:
                core = f"{quote(_wrap(passages[position - 2].text))}\n\n{core}"
            body = (
                f"{greeting},\n\n{attribution}\n\n{quote(core)}\n\n{_wrap(passage.text)}\n\n"
                f"{signoff},\n{sender[0].split()[0]}"
                if position
                else _letter(greeting, passage, signoff, sender[0])
            )
            carries = tuple(passages[max(0, position - 2) : position + 1][::-1])
            envelope.set_content(body + "\n")
        else:
            body = _html_letter(greeting, passage, signoff, sender[0])
            if position:
                body = (
                    f'{body}<br><div class="gmail_quote"><div dir="ltr" class="gmail_attr">'
                    f"{html.escape(attribution)}<br></div>"
                    '<blockquote class="gmail_quote" style="margin:0px 0px 0px 0.8ex;'
                    'border-left:1px solid rgb(204,204,204);padding-left:1ex">'
                    f"{bodies[-1]}</blockquote></div>"
                )
            carries = passages[position::-1]
            envelope.set_content(body + "\n", subtype="html")
        bodies.append(body)
        ids.append(message_id)
        thread.messages.append(
            Message(
                key=message_id,
                raw=envelope.as_bytes(policy=SMTP),
                own=passage,
                carries=tuple(carries),
                thread=thread,
            )
        )
    thread.queried = tuple(sorted({0, length // 2, length - 1}))
    return thread


def _single(
    key: str, subject: str, passage: Passage, *, companion: bool, rng: random.Random, when: datetime
) -> Message:
    """A message outside any thread: a companion restating ``passage``, or a distractor.

    A companion carries no passage in the ground-truth sense, because its text is a rewording:
    it is found by its identity, never by locating the statement inside it.
    """
    text = passage.companion if companion else passage.text
    sender, to = rng.sample(PEOPLE, 2)
    envelope = _envelope(
        subject=subject,
        sender=sender,
        to=to,
        when=when,
        message_id=f"<{key}@example.org>",
        parent=None,
        references=[],
    )
    greeting, signature = to[0].split(maxsplit=1)[0], sender[0].split(maxsplit=1)[0]
    envelope.set_content(f"Hi {greeting},\n\n{_wrap(text)}\n\n{signature}\n")
    return Message(
        key=f"<{key}@example.org>",
        raw=envelope.as_bytes(policy=SMTP),
        own=passage,
        carries=() if companion else (passage,),
    )


def build_corpus(
    *,
    seed: int = 463,
    lengths: tuple[int, ...] = LENGTHS,
    styles: tuple[str, ...] = STYLES,
    copies: int = 2,
    distractors: int = 20,
) -> Corpus:
    """``copies`` threads per length and style, a companion per queried passage, distractors."""
    if max(lengths) > len(ASPECTS):
        msg = f"a thread of {max(lengths)} would repeat a decision; at most {len(ASPECTS)}"
        raise ValueError(msg)
    topics = [f"{code} {system}" for code, system in zip(CODENAMES, SYSTEMS, strict=True)]
    needed = len(lengths) * len(styles) * copies + distractors
    if needed > len(topics):
        msg = f"{needed} topics needed and {len(topics)} exist"
        raise ValueError(msg)
    rng = random.Random(seed)  # noqa: S311 - a seeded corpus generator, not a key
    rng.shuffle(topics)
    threads: list[Thread] = []
    companions: dict[str, Message] = {}
    epoch = EPOCH
    for copy in range(copies):
        for style in styles:
            for length in lengths:
                key = f"{style}-{length:02d}-{copy}"
                thread = build_thread(
                    key, style=style, length=length, topic=topics.pop(), rng=rng, epoch=epoch
                )
                threads.append(thread)
                epoch += timedelta(days=2)
                for position in thread.queried:
                    passage = thread.messages[position].own
                    companions[passage.id] = _single(
                        f"note.{passage.id}",
                        f"Notes on the {thread.topic}",
                        passage,
                        companion=True,
                        rng=rng,
                        when=epoch,
                    )
    others: list[Message] = []
    for number in range(distractors):
        topic = topics.pop()
        passage = _passage(f"other.{number:02d}", rng.choice(ASPECTS), topic, rng)
        others.append(
            _single(
                f"other.{number:02d}",
                f"{topic.capitalize()} update",
                passage,
                companion=False,
                rng=rng,
                when=epoch + timedelta(hours=number),
            )
        )
    return Corpus(threads=threads, companions=companions, distractors=others)


__all__ = [
    "ASPECTS",
    "LENGTHS",
    "STYLES",
    "Corpus",
    "Message",
    "Passage",
    "Thread",
    "build_corpus",
    "build_thread",
    "quote",
]

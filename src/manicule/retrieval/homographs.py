"""Ordinary English words a glossary term might collide with.

The list exists to answer one question and no other: *is this token a word people write
without meaning the glossary term?* ``NOW`` is; ``ISOTOPE`` is not. The answer decides whether
a lower-case occurrence needs corroborating evidence before it may expand, and nothing else in
manicule consults it.

**It is deliberately short, and short in a particular direction.** Every entry is a word common
enough that a corpus will contain dozens of innocent uses, which is the only population where
over-expansion is expensive. Adding rare words would not make the feature more correct — an
acronym colliding with ``quoin`` expands, and the outcome is one extra cited candidate for a
query that named the term — while a long list would make the rule harder to review and would
start refusing legitimate terms.

**It is not a stopword list and must not become one.** Stopwords are removed from text;
these are kept, and merely require a second reason before they expand. Nor is it a spell
checker: it is not consulted for anything but a glossary key, and a key is at most twelve
characters.

Operators extend it — ``rag.glossary.homographs`` — rather than editing it, because the words
that collide with one corpus's terms are a property of that corpus.
"""

from __future__ import annotations

from typing import Final

_WORDS = (
    # Written as prose and split, rather than as a list literal of nine hundred quoted
    # strings. This is a word list, and the only edit anybody will ever make to it is adding
    # or removing a word; one word per line with quotes and a comma would make that edit
    # nine hundred lines long and no easier to review.
    # Function words and the commonest verbs, which is where short acronyms land.
    "a an and are as at be been being but by can could did do does doing done for from"
    " get gets got had has have having he her here hers him his how i if in into is it"
    " its me more most must my no nor not now of off on once one only or other others"
    " ought our ours out over own same she should so some such than that the their theirs"
    " them then there these they this those through to too under until up very was we"
    " were what when where which while who whom why will with would you your yours"
    # Common nouns, adjectives and verbs of the length an abbreviation has.
    " able about above across act action add after again against age agree air all allow"
    " almost alone along already also although always among amount another answer any"
    " appear apply area arm around arrive art ask aside away back bad bag ball bank base"
    " basic beat become bed before begin behind believe below best better between beyond"
    " big bill bit black block blood blue board boat body book born both box boy break"
    " bring broad brother budget build burn business busy buy call calm camp cap car card"
    " care carry case cash cast cat catch cause cell center central century certain chain"
    " chair chance change charge cheap check chief child choice choose city civil claim"
    " class clean clear climb clock close coach coal coast code cold collect college color"
    " come common company compare complete concern condition consider contain continue"
    " control cook cool copy corner cost count country couple course court cover create"
    " crew crime cross crowd cry cup current cut damage dance danger dark data date day"
    " dead deal dear death debate debt decide deep defense degree delay deliver demand"
    " deny depend describe design desk detail develop die diet differ dinner direct dirt"
    " discuss disease divide doctor dog door doubt down draw dream dress drink drive drop"
    " dry due dust duty each early earn earth ease east easy eat edge effect effort egg"
    " eight either elect else empty end enemy energy enjoy enough enter entire equal error"
    " escape even event ever every exact example exist exit expect expert explain eye face"
    " fact fail fair fall false family famous far farm fast fat father fault favor fear"
    " feed feel few field fight figure file fill film final find fine finger finish fire"
    " firm first fish fit five fix flat floor flow fly focus fold follow food foot force"
    " forest forget form former forward found four frame free fresh friend front fruit"
    " full fun fund future gain game garden gas gate gather general gift girl give glass"
    " goal gold good grade grant grass great green ground group grow guard guess guest"
    " guide gun hair half hall hand hang happen happy hard hat hate head health hear heart"
    " heat heavy help hide high hill hire hit hold hole holiday home honest hope horse"
    " host hot hotel hour house huge human hunt hurt idea ill image impact important"
    " improve include income increase indeed index industry inform inside instead"
    " interest into invite iron issue item join joint joke joy judge jump just keep key"
    " kid kill kind king kitchen knee know lack lady lake land language large last late"
    " laugh law lay lead learn least leave left leg legal length less let letter level"
    " lie life lift light like limit line link lip list listen little live load loan local"
    " lock long look lose loss lot loud love low luck lunch machine main major make man"
    " manage many map mark market marry mass master match matter may mean meat media meet"
    " member memory mention message metal method middle might mile milk mind mine minor"
    " minute miss mix model modern moment money month moon moral morning mother motion"
    " mount mouth move movie much music name nation native nature near neck need never"
    " new news next nice night nine none normal north nose note nothing notice novel"
    " number nurse object occur ocean odd offer office often oil old open opera option"
    " orange order origin others outside owe pace pack page pain paint pair panel paper"
    " parent park part party pass past path patient pattern pay peace pen people per"
    " perfect perform period person phase phone photo pick picture piece pilot pink pipe"
    " place plan plant plate play please plenty plot plus point police policy pool poor"
    " pop port pose position possible post pound pour power practice praise pray prefer"
    " prepare present press pretty prevent price pride print prior prize problem process"
    " produce profit program project proof proper protect proud prove provide public pull"
    " pure purple purpose push put quality queen question quick quiet quite race radio"
    " rail rain raise range rank rapid rare rate rather reach read ready real reason"
    " recall receive record red reduce refer reflect refuse regard region regular reject"
    " relate relax release remain remember remove rent repair repeat replace reply report"
    " request require rescue respond rest result retain return reveal review rich ride"
    " right ring rise risk river road rock role roll roof room root rope rough round route"
    " row rule run rural safe sail salt same sample sand save say scale scene school"
    " science score sea search season seat second secret section secure see seed seek seem"
    " sell send sense series serious serve set settle seven shake shall shape share sharp"
    " she sheet shelf shell shift shine ship shirt shock shoe shoot shop short shot should"
    " show shut sick side sign silent silver similar simple since sing single sink sir"
    " sister sit site size skill skin sky sleep slide slight slip slow small smart smile"
    " smoke snow social soft soil sold soldier solid solve some son song soon sorry sort"
    " soul sound soup source south space speak special speed spell spend spirit split"
    " sport spot spread spring square stage stand star start state stay steal steam steel"
    " step stick still stock stone stop store storm story straight strange street stress"
    " strike strong study stuff style subject succeed such sudden suffer sugar suggest"
    " suit summer sun supply support suppose sure surface surprise survey sweet swim"
    " switch symbol system table take talk tall tape target task taste tax teach team tear"
    " tell ten tend term test text thank theme theory thick thin thing think third though"
    " threat three throw thus tie tight time tiny tip tire title today toe together tone"
    " tonight tool tooth top total touch tough tour toward town trace track trade train"
    " travel treat tree trial trip trouble true trust truth try tube turn twice two type"
    " ugly unable uncle unit unless unlike upon upper urban urge use useful usual value"
    " van vast video view village visit voice vote wage wait wake walk wall want war warm"
    " warn wash waste watch water wave way weak wear weather week weight welcome well west"
    " wet wheel whether whole whose wide wife wild win wind window wine wing winner winter"
    " wire wise wish woman wonder wood word work world worry worse worth wrap write wrong"
    " yard yeah year yellow yes yet young youth zone"
)

COMMON_ENGLISH_WORDS: Final[frozenset[str]] = frozenset(word.upper() for word in _WORDS.split())


def is_common_word(key: str) -> bool:
    """Whether a normalized glossary key is also an ordinary English word."""
    return key.upper() in COMMON_ENGLISH_WORDS


__all__ = ["COMMON_ENGLISH_WORDS", "is_common_word"]

"""Shared deterministic artifact tokenization and weighted cosine."""
import re
import math
import unicodedata

COMMON_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from
had has have having he her here hers herself him himself his how i if in into
is it its itself may me might more most must my myself no nor not of on once
only or other our ours ourselves out over own same shall she should so some
such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
ad ai agli al alla alle allo anche che chi ci coi col come con da dagli dai
dal dalla dalle dallo dei del della delle dello di dove e ed era essere gli
ha hai hanno ho i il in io la le li lo loro lui ma mi nei nel nella nelle
nello negli noi non o per piu più quale quando questo questa questi queste
quello quella quelli quelle se sei senza si sia siano siamo siete solo sono
su sul sulla sulle sullo sugli tra tu tutti tutte tutto una uno un voi vi
""".split())


CODE_KEYWORDS = frozenset("""
abstract alignas alignof and and_eq asm assert auto base bitand bitor bool
boolean break byte case catch char checked class compl const constexpr
consteval constinit continue co_await co_return co_yield decltype default
delegate delete do double dynamic else enum event explicit export extern
false final finally fixed float for foreach friend goto if implements import
in inline instanceof int interface internal is lock long module mutable
namespace native new noexcept not not_eq null nullptr object operator or
or_eq out override package params private protected public readonly record
ref register reinterpret_cast requires return sbyte sealed short signed sizeof
stackalloc static static_assert static_cast strictfp string struct super switch
synchronized template this thread_local throw throws transient true try typedef
typeid typename typeof uint ulong unchecked union unsigned unsafe ushort using
var virtual void volatile wchar_t while with xor xor_eq
""".split())


CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(text: str, side: str) -> frozenset[str]:
    """Tokenize the complete text; no character/token truncation is used."""
    if not isinstance(text, str):
        raise TypeError("Gate inputs must be artifact text strings")
    text = CAMEL_BOUNDARY.sub(" ", unicodedata.normalize("NFKC", text))
    stopwords = COMMON_STOPWORDS | CODE_KEYWORDS if side == "code" else COMMON_STOPWORDS
    return frozenset(
        term for match in WORDS.finditer(text)
        if len(term := match.group().casefold()) > 1
        and not term.isnumeric() and term not in stopwords
    )


def _vector(terms: frozenset[str], idf: dict[str, float],
            unseen_idf: float) -> tuple[dict[str, float], float]:
    weights = {term: idf.get(term, unseen_idf) for term in terms}
    return weights, math.sqrt(math.fsum(value * value for value in weights.values()))


def _cosine(left: tuple[dict[str, float], float],
            right: tuple[dict[str, float], float]) -> float:
    a, na = left
    b, nb = right
    if not na or not nb:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    value = math.fsum(weight * b.get(term, 0.0) for term, weight in a.items()) / (na * nb)
    return max(0.0, min(1.0, value))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] + (location - low) * (ordered[high] - ordered[low])

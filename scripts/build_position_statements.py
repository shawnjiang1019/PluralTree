"""Rewrite survey (question, option) pairs into DECLARATIVE POSITION STATEMENTS.

Offline, one-time, cached. Produces the frozen artifact that alignment.reward
loads to build Position.embed_text:

    "On balance, do you think the growing number of guns in the country is good
     or bad for society?  Bad for society"                      <- what we had
    "The growing number of guns in the country is bad for society."   <- target

Why (measured, docs/reward_gate_failure.md): at min_depth_words=0 -- the depth
gate fully disabled -- 76% of responses still matched ZERO positions, against a
judge that scored the same responses at 0.43 mean coverage. Two mechanisms, both
caused by embedding the FULL leaf string:

  (a) REGISTER MISMATCH. A question-shaped string compared against answer-shaped
      response prose gives low cosine; mpnet is symmetric and this comparison is
      not. `mentioned = sim.max(0) >= 0.50` therefore almost never fires.
  (b) LOW DISCRIMINABILITY. Every option of an item repeats the same question
      stem, so its positions are near-identical to each other; the argmax in
      position_depths scatters units across positions and none accumulates the
      words min_depth_words wants.

A declarative statement is answer-shaped (fixes (a)) and puts the differentiating
content in the WHOLE string instead of a trailing fragment (fixes (b)).

REPRESENTATION ONLY. The positions and their prevalences stay exactly what the
survey says -- this changes how a position is WORDED, never which positions exist
or how much weight they carry. The template backend is a surface transform; the
llm backend is instructed to rephrase and nothing else.

COVERAGE. The template rules fire on 50% of positions measured over all 2,004 raw
ATP questions (6,782 (question, option) pairs) in the OpinionQA release. A miss
emits "<question> <option>" -- today's string exactly -- so an unmatched row is
never worse than the status quo, and every build prints its own match rate rather
than assuming this one. Rules were added in descending order of corpus frequency;
what remains is a long tail of one-off stems ("Which of these comes closest...",
subject-enumeration items like "Men are better"), which is what --backend llm is
for. A rule whose output still looks like a question is REJECTED to the fallback
(_is_clean) -- a fragment scores worse than the honest original.

CAVEAT, stated up front rather than discovered later: the reward's claim to be
usable outside the eval rests on being causally INDEPENDENT of the OvertonBench
judge. Rewriting pollster options into natural viewpoint language makes them
resemble the judge's human-written viewpoint clusters, so this fix buys matchable
targets at the cost of some of that independence. It is a real cost; it belongs in
the writeup.

Usage:
    # build (frozen artifact; refuses to clobber without --force)
    OPINIONQA_DIR=... python scripts/build_position_statements.py \
        --dataset opinionqa --out artifacts/position_statements.jsonl

    # optional LLM pass over the rows no template rule matched
    ... --backend llm --base_url http://localhost:8000/v1 --model Qwen/...

    # visible quality check on hand-written ATP-style pairs (no data, no model)
    python scripts/build_position_statements.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment.reward import statement_key   # noqa: E402  (one key definition)

SCHEMA_VERSION = 1

_WS = re.compile(r"\s+")

# First word kept capitalized when an option is lowered into mid-sentence
# position. Small and hand-maintained: the alternative (a POS tagger) is a
# dependency, and getting this wrong only costs cosmetics.
_PROPER = {"u.s.", "us", "america", "american", "americans", "black", "white",
           "hispanic", "latino", "asian", "god", "congress", "republicans",
           "democrats", "republican", "democratic", "china", "russia", "europe",
           "christians", "muslims", "jews", "president", "trump", "obama", "biden"}

# Options that are already noun phrases get an article when a rule slots them
# after a copula ("is <a major reason>"). Restricted to the rules whose option
# families are known to be noun-shaped -- blanket article insertion mangles
# adjectival options ("a very important").
_NO_ARTICLE = {"a", "an", "the", "no", "not", "none", "some", "any", "all",
               "my", "his", "her", "their", "top", "should"}


# ---------------------------------------------------------------------------
# Surface helpers
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").replace("’", "'")).strip()


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _uncap(s: str) -> str:
    """Lower the first letter, unless it is 'I' or a proper noun."""
    head = s.split(" ", 1)[0].strip(",.")
    if head == "I" or head.lower() in _PROPER or head[:2].isupper():
        return s
    return s[:1].lower() + s[1:]


def _low(opt: str) -> str:
    """Option lowered for mid-sentence use ('Very important' -> 'very important')."""
    o = _norm(opt)
    head = o.split(" ", 1)[0].lower().strip(",")
    if head in _PROPER or (o[:1].isupper() and o[1:2].isupper()):   # proper / ACRONYM
        return o
    return o[:1].lower() + o[1:]


def _art(opt: str) -> str:
    """Option with an indefinite article when it needs one ('Major reason' -> 'a major reason')."""
    o = _low(opt)
    head = o.split(" ", 1)[0].lower()
    if head in _NO_ARTICLE:
        return o
    return ("an " if o[:1].lower() in "aeiou" else "a ") + o


_SECOND_PERSON = [(re.compile(r"\byours\b", re.I), "mine"),
                  (re.compile(r"\byourself\b", re.I), "myself"),
                  (re.compile(r"\byour\b", re.I), "my"),
                  # object of a preposition -> "me", everywhere else "I": "the
                  # debt you have" must not become "the debt me have".
                  (re.compile(r"\b(to|for|about|with|of|at|on|from|than) you\b", re.I),
                   r"\1 me"),
                  (re.compile(r"\byou\b", re.I), "I")]


def _first_person(s: str) -> str:
    """'your household' -> 'my household'. Only for rules whose statement is
    already framed as 'I ...'; elsewhere the second person is the survey's own
    wording and rewriting it would be a content change."""
    for pat, rep in _SECOND_PERSON:
        s = pat.sub(rep, s)
    return s


# Agreement damage from splicing a plural frame onto a singular placeholder
# ("...I do not think THIS ARE likely to happen").
_AGREE = [(re.compile(r"\bthis are\b", re.I), "this is"),
          (re.compile(r"\bthis were\b", re.I), "this was"),
          (re.compile(r"\bthis do\b", re.I), "this does"),
          (re.compile(r"\bI were\b"), "I was"),          # from "when you were..."
          (re.compile(r"\bis is\b", re.I), "is")]


def _sentence(s: str) -> str:
    s = _norm(s).rstrip(" ,;:")
    if not s:
        return s
    for pat, rep in _AGREE:
        s = pat.sub(rep, s)
    return _cap(s) + ("" if s.endswith((".", "!", "?")) else ".")


# Leading survey throat-clearing. It is CONTENT-BEARING often enough ("In the
# last 12 months, ...", "Thinking about your own job, ...") that we do not drop
# it: it is detached so the rules see a clean clause, then re-attached in front
# of the statement.
_LEAD = re.compile(
    r"^(?P<lead>(?:in general|on balance|overall|generally speaking|all in all"
    r"|as you may know|as far as you know|in your opinion|in your view"
    r"|thinking [^,?]{0,100}|now thinking [^,?]{0,100}"
    r"|when it comes to [^,?]{0,80}|regardless of [^,?]{0,80}"
    r"|looking ahead[^,?]{0,40}|in the (?:last|next|past) [^,?]{0,40}"
    r"|in order to [^,?]{0,80}|if you were deciding [^,?]{0,100}"
    r"|in the future|these days|by the year [^,?]{0,20}|compared [^,?]{0,50}))"
    r",\s+(?P<core>.+)$", re.IGNORECASE)


def split_lead(question: str) -> tuple[str, str]:
    """(leading adverbial, main clause). Lead is '' when there is none."""
    q = _norm(question).rstrip(" ?.")
    m = _LEAD.match(q)
    return (m.group("lead"), m.group("core")) if m else ("", q)


_DOYOUTHINK = re.compile(
    r"^(?:how (?:much|many|often|well|big)\s+)?"
    r"(?:do|does|did) you (?:think|say|feel|believe)(?: that)?\s+", re.IGNORECASE)
_WOULDYOUSAY = re.compile(r"^would you say(?: that)?\s+", re.IGNORECASE)
# Battery instructions: pure administrivia, no propositional content.
_INSTRUCTION = re.compile(
    r"^(?:for each(?: one)?(?: of [^,]{1,25})?,?\s*)?"
    r"(?:please )?(?:indicate|tell us|choose|select|say)\s*"
    r"(?:whether|which|if|how)?\s*(?:you think\s+)?", re.IGNORECASE)


def _strip_frame(core: str) -> str:
    """Drop the interrogative frame verbs that carry no propositional content."""
    prev = None
    while prev != core:
        prev = core
        core = _DOYOUTHINK.sub("", core)
        core = _WOULDYOUSAY.sub("", core)
        core = _INSTRUCTION.sub("", core)
    return core.strip()


# "if at all" / "if any" / "if ever" / "if anything" -- ATP hedges the degree
# scale in half a dozen wordings and puts the hedge in several positions
# ("how often, IF EVER, do you", "how concerned are you, IF AT ALL, about").
# It is a hedge on the SCALE, not content, so it is deleted before matching
# rather than enumerated in every pattern (measured: 178 'how often, if ever,'
# stems alone, all of them missed while the patterns only knew 'if at all').
_HEDGE = re.compile(
    r"\s*,?\s*\bif (?:at all|any|ever|anything|anything at all)\b\s*,?\s*",
    re.IGNORECASE)
_IFA = r"\s*"


def _dehedge(s: str) -> str:
    return _norm(_HEDGE.sub(" ", s))

# Tokens a clause may legitimately end on when we cut it at the enumerated
# options ("... race relations are | getting better, getting worse, ...").
_COPULA_TAIL = {"is", "are", "was", "were", "be", "been", "being", "become",
                "becomes", "becoming", "gotten", "get", "gets", "getting", "made"}

# ...and tokens it may NOT: a cut that leaves a dangling preposition or
# determiner produced a fragment, not a clause.
_DANGLING = {"of", "to", "the", "a", "an", "and", "or", "in", "on", "for", "that",
             "with", "about", "by", "from", "as", "at", "than", "if", "but",
             "when", "how", "what", "who", "which", "you", "do", "does", "did"}

# Auxiliaries that front a polar question; moved back into place so the cut
# clause reads as a statement ("Has X been | mostly positive" -> "X has been ...").
_AUX_FRONT = {"is", "are", "was", "were", "has", "have", "had", "should",
              "would", "will", "can", "could", "does"}


def _deinvert(prefix: str) -> str:
    """Undo subject-auxiliary inversion on a cut clause.

    "Are race relations in the US"      -> "race relations in the US are"
    "Has what you heard about X been"   -> "what you heard about X has been"
    Only for the auxiliaries that can move without morphology -- do/did need
    the main verb re-tensed, so those are left alone (they fall back).
    """
    toks = prefix.split()
    if len(toks) < 3 or toks[0].lower() not in _AUX_FRONT:
        return prefix
    aux, rest = toks[0].lower(), toks[1:]
    if rest[-1].lower() in _COPULA_TAIL and rest[-1].lower() != aux:
        rest.insert(len(rest) - 1, aux)          # "... been" -> "... has been"
    else:
        rest.append(aux)                         # "race relations ..." -> "... are"
    return " ".join(rest)


# ---------------------------------------------------------------------------
# Template rules
# ---------------------------------------------------------------------------
# Each rule is (name, compiled pattern over the main clause, builder(m, opt)).
# Ordered: the specific frames first, the option-echo cut last (it is the most
# general and the most likely to misfire on a stem a specific rule owns).
def _rules():
    R = []

    def rule(name, pattern, fn):
        R.append((name, re.compile(pattern, re.IGNORECASE), fn))

    # "How much, if at all, is X a reason why Y?"  -> "X is a major reason why Y."
    # ATP's own template leaves a stray copula in X ("...to make money is a
    # reason why..."), so trim one off the subject before re-attaching ours.
    rule("reason_why",
         r"^how much" + _IFA + r"(?:do you think\s+)?"
         r"(?:is|are|was|were)?\s*(?P<x>.+?)\s+a reason\s+(?P<y>(?:why|for|to)\s+.+)$",
         lambda m, o: f"{_cap(_detrail_copula(m['x']))} is {_art(o)} {m['y']}"
         if "reason" in o.lower()
         else f"{_cap(_detrail_copula(m['x']))} is {_art(o)} reason {m['y']}")

    # "How important, if at all, is it (to you) for X to Y?" -> "It is essential for X to Y."
    rule("importance_frame",
         r"^how important" + _IFA + r"(?:do you think\s+)?(?:is|are) it(?: to you)?"
         r"(?:, personally,?)?\s+(?P<rest>(?:for|to|that)\s+.+)$",
         lambda m, o: f"It is {_low(o)} {m['rest']}")

    # "How important is X to Y?" -> "X is very important to Y."
    rule("importance_to",
         r"^how important" + _IFA + r"(?:is|are)\s+(?P<x>.+?)\s+to\s+(?P<y>.+)$",
         lambda m, o: f"{_cap(m['x'])} is {_low(o)} to {m['y']}")

    # "How important is X?" -> "X is very important."
    rule("importance_np",
         r"^how important" + _IFA + r"(?:is|are)\s+(?P<x>.+)$",
         lambda m, o: f"{_cap(m['x'])} is {_low(o)}")

    # "How much of a problem is X?" -> "X is a very big problem."
    rule("problem",
         r"^how (?:much|big) of an? [a-z ]{2,20}" + _IFA + r"(?:do you think\s+)?"
         r"(?:is|are)\s+(?P<x>.+)$",
         lambda m, o: f"{_cap(m['x'])} is {_art(o)}")

    # "How much of a challenge do you think X is in doing Y?"
    #   -> "X is a moderately big challenge in doing Y."
    rule("problem_inverted",
         r"^how (?:much|big) of an? [a-z ]{2,20}" + _IFA + r"(?:do you think\s+)?"
         r"(?P<x>.+?)\s+(?:is|are)\s+(?P<rest>.+)$",
         lambda m, o: f"{_cap(m['x'])} is {_art(o)} {m['rest']}")

    # "What priority would you give to X?" -> "X should be a top priority."
    rule("priority_give",
         r"^what priority would you give to\s+(?P<x>.+)$",
         lambda m, o: f"{_cap(m['x'])} {_low(o)}"
         if _low(o).startswith("should") else f"{_cap(m['x'])} should be {_art(o)}")

    # "How much priority should X be given?" -> "X should be given top priority."
    rule("priority_given",
         r"^how much priority" + _IFA + r"(?:do you think\s+)?(?P<x>.+?)\s+"
         r"should be given$",
         lambda m, o: f"{_cap(m['x'])} {_low(o)}"
         if _low(o).startswith("should") else f"{_cap(m['x'])} should be given {_low(o)}")

    # "X is a reason why Y" -> "X is a major reason why Y." Covers both the
    # battery frame ("...whether IT is a reason why...", where the item supplies
    # the subject) and the stem-subject form left by _strip_frame.
    rule("is_a_reason",
         r"^(?P<x>.{2,120}?)\s+(?:is|are)\s+a reason\s+(?P<y>(?:why|for)\s+.+)$",
         lambda m, o: f"{_this(_detrail_copula(m['x']))} is {_art(o)} {m['y']}"
         if "reason" in o.lower()
         else f"{_this(_detrail_copula(m['x']))} is {_art(o)} reason {m['y']}")

    # "How does being a man affect people's ability to get ahead?"
    #   -> "Being a man helps a lot when it comes to people's ability to get ahead."
    # When the option already names the affected thing ("Mostly helps a woman's
    # chances"), the duplicate head of the object is dropped instead.
    rule("impacts",
         r"^how (?:do|does) (?:you think\s+)?(?P<x>.+?)\s+"
         r"(?:impacts?|affects?|influences?)\s+(?P<y>.+)$",
         lambda m, o: f"{_cap(m['x'])} {_join_object(o, m['y'])}")

    # "How many of your friends own guns?" -> "All or most of my friends own guns."
    rule("how_many_of",
         r"^how many of\s+(?P<x>.+)$",
         lambda m, o: f"{_cap(_dethem(o))} of {_first_person(m['x'])}")

    # "How much confidence do you have in X ...?" -> "I have a great deal of confidence in X."
    rule("confidence_in",
         r"^how much confidence" + _IFA + r"do you have in\s+(?P<x>.+)$",
         lambda m, o: f"I have {_art(o)} in {_first_person(m['x'])}"
         if "confidence" in o.lower()
         else f"I have {_art(o)} confidence in {_first_person(m['x'])}")

    # "How confident are you that X?" -> "I am very confident that X."
    rule("confident_that",
         r"^how confident" + _IFA + r"are you\s+(?P<rest>(?:that|in|about)\s+.+)$",
         lambda m, o: f"I am {_low(o)} {_first_person(m['rest'])}")

    # "How satisfied/concerned are you with/about X?" -> "I am very satisfied with X."
    rule("attitude_toward",
         r"^how (?:satisfied|concerned|worried|happy)" + _IFA
         + r"are you\s+(?P<rest>(?:with|about|by|over)\s+.+)$",
         lambda m, o: f"I am {_low(o)} {_first_person(m['rest'])}")

    # "Do you favor or oppose X?" -> "I strongly oppose X."
    rule("favor_oppose",
         r"^(?:do|would) you (?:favor or oppose|oppose or favor)\s+(?P<x>.+)$",
         lambda m, o: f"I {_low(o)} {_first_person(m['x'])}")

    # "Do you approve or disapprove of X?" -> "I approve of X."
    rule("approve",
         r"^(?:do|would) you (?:approve or disapprove|disapprove or approve) of\s+(?P<x>.+)$",
         lambda m, o: f"I {_low(o)} of {m['x']}")

    # "How often do you X?" -> "I X a few times a week." (second person -> first,
    # so the statement reads as a held position rather than as an interview turn)
    rule("how_often_you",
         r"^how often" + _IFA + r"(?:do|did|have) you\s+(?P<rest>.+)$",
         lambda m, o: f"I {_first_person(m['rest'])} {_low(o)}")

    # "How often would you say X?" -> "X some of the time."
    rule("how_often_say",
         r"^how often" + _IFA + r"would you say\s+(?P<rest>.+)$",
         lambda m, o: f"{_cap(m['rest'])} {_low(o)}")

    # "How much, if at all, do you worry about X?" -> "I worry a little about X."
    # Fires only when the option repeats the stem's verb ("Worry a little"), which
    # is what makes the option a drop-in replacement for it.
    rule("degree_verb_you",
         r"^how (?:much|often|closely|strongly)" + _IFA
         + r"(?:do|did) you\s+(?P<v>\w+)\s*(?P<rest>.*)$",
         lambda m, o: f"I {_low(o)} {_first_person(m['rest'])}"
         if _low(o).split(" ")[0].rstrip("s") == m["v"].lower().rstrip("s") else "")

    # "How well, if at all, does X describe Y?" -> "X describes Y very well."
    rule("how_well",
         r"^how well" + _IFA + r"(?:do|does) (?P<x>.+?) "
         r"(?P<v>describe|handle|represent|reflect|fit|match)\s+(?P<y>.+)$",
         lambda m, o: f"{_cap(m['x'])} {m['v']}s {m['y']} {_low(o)}"
         if not _low(o).startswith(("describes", "does not describe"))
         else f"{_cap(m['x'])} {_low(o)}")

    # "How much pressure, if any, do you think men face to X?"
    #   -> "Men face a lot of pressure to X."
    rule("how_much_np",
         r"^how much (?P<np>[a-z ]{3,20}?)" + _IFA + r"(?:do|does) (?:you think\s+)?"
         r"(?P<subj>.+?)\s+(?P<v>face|have|feel|experience|get)\s+(?P<rest>.*)$",
         lambda m, o: (f"{_cap(_first_person(m['subj']))} {m['v']} "
                       f"{_quantify(o, m['np'])} {_first_person(m['rest'])}"
                       if m["subj"].strip().lower() in {"you", "your"}
                       else f"{_cap(m['subj'])} {m['v']} "
                            f"{_quantify(o, m['np'])} {m['rest']}"))

    # Catch-all for the "how much / how often" family, whose options are bare
    # degrees ("A lot", "Not at all", "A great deal") with no verb to re-attach:
    # keep the clause, put the degree at the end where an English speaker would.
    #   "How much do you think what happens to Asians affects your own life?"
    #     + "A lot" -> "What happens to Asians affects my own life a lot."
    # LAST in the table: every frame above knows a better place for its option,
    # and this only fires once _strip_frame has left a declarative clause.
    rule("degree_tail",
         r"^(?P<rest>.+)$",
         lambda m, o: f"{_cap(_first_person(m['rest']))} {_low(o)}"
         if _DEGREE_ONLY.match(o) and not _INTERROGATIVE.match(m["rest"])
         and not _HOWDEG.search(m["rest"]) else "")

    return R


# Start-of-statement interrogatives. "What happens to X affects Y" is a perfectly
# good declarative, so 'what' is deliberately absent.
_INTERROGATIVE = re.compile(
    r"^(?:how|which|who|why|when|where|do|does|did|is|are|was|were|has|have|"
    r"had|would|should|could|can|will|please)\b", re.IGNORECASE)
_HOWDEG = re.compile(r"\bhow (?:much|often|many|well|big)\b", re.IGNORECASE)

# A statement that still contains the interview turn ("...do you think...") is a
# misfire: a rule spliced the option into a question instead of into a clause.
# Rejecting it costs nothing -- the row falls back to today's behaviour, which is
# the thing we are trying to beat, not something we can lose to.
_DIRTY = re.compile(
    r"\b(?:do|does|did|would|will|should|have|has) you\b|\byou say that\b|"
    r"\bhow (?:much|often|many) (?:do|does|did|would|is|are)\b|"
    r"\bplease (?:indicate|tell|choose)\b", re.IGNORECASE)


def _is_clean(stmt: str) -> bool:
    """A statement, not a question with an option glued into it."""
    return ("?" not in stmt and not _DIRTY.search(stmt)
            and not _INTERROGATIVE.match(stmt))


def _this(x: str) -> str:
    """Battery placeholders read better as 'This' than as themselves."""
    return "This" if _norm(x).lower() in {"it", "this", "that"} else _cap(x)


# Options that state a degree and nothing else -- they can be appended to a
# clause but cannot replace a verb in it.
_DEGREE_ONLY = re.compile(
    r"^(?:a lot|a little|a great deal|a fair amount|some|somewhat|not much|"
    r"not too much|not at all|nothing at all|none|none at all|a bit|"
    r"very much|a moderate amount|hardly at all|only a little|a great amount)$",
    re.IGNORECASE)


_TRAIL_COPULA = re.compile(r"\s+(?:is|are|was|were)$", re.IGNORECASE)
# Measure phrases take "of" before the noun ("a lot OF pressure"); bare
# quantifiers do not ("some pressure").
_OF_HEAD = re.compile(r"\b(lot|deal|amount|bit|number|little|much)$", re.IGNORECASE)


def _detrail_copula(s: str) -> str:
    return _TRAIL_COPULA.sub("", _norm(s))


def _dethem(option: str) -> str:
    """'All of them' -> 'all' (the 'of <NP>' comes from the question)."""
    return re.sub(r"\s+of them$", "", _low(option), flags=re.IGNORECASE)


def _overlap_split(option: str, rest: str) -> str:
    """``rest`` with its leading words dropped where the option already said them.

    "Mostly helps a woman's chances" + "a woman's chances of getting elected"
      -> "of getting elected"   (so the two splice into one phrase, not two)
    """
    o_toks = _low(option).lower().split()
    r_toks = rest.split()
    for n in range(min(len(o_toks), len(r_toks)), 0, -1):
        if o_toks[-n:] == [t.lower() for t in r_toks[:n]]:
            return " ".join(r_toks[n:])
    return rest


def _join_object(option: str, obj: str) -> str:
    """Option + the thing it acts on, with or without a linking phrase."""
    rest = _overlap_split(option, obj)
    if rest != obj:                                  # option already names it
        return f"{_low(option)} {rest}".strip()
    return f"{_low(option)} when it comes to {obj}"


def _quantify(option: str, noun: str) -> str:
    """'A lot' + 'pressure' -> 'a lot of pressure'; 'Some pressure' -> unchanged."""
    o, n = _low(option), _norm(noun).strip()
    if not n or n.lower() in o.lower():
        return o
    return f"{o} of {n}" if _OF_HEAD.search(o) else f"{o} {n}"


RULES = _rules()


# Option heads too generic to anchor a cut on.
_WEAK_HEAD = {"a", "an", "the", "not", "no", "yes", "some", "all", "more", "less",
              "about", "very", "somewhat", "strongly", "major", "minor", "top",
              "essential", "important", "other", "both", "neither", "same"}


def _echo_cut(core: str, option: str, options: list[str]) -> str | None:
    """Cut the stem where it enumerates its own options, then splice the option in.

    "the number of legal immigrants the U.S. admits should be increased,
     decreased, or kept about the same"  +  "Decreased"
      -> "The number of legal immigrants the U.S. admits should be decreased."
    "not enough regulation of major corporations contributes to economic
     inequality"  +  "Contributes a great deal"
      -> "Not enough regulation ... contributes a great deal to economic inequality."

    Two passes: the whole option, then its head word -- ATP splits an option
    across the "or" ("...is good or bad for society" / "Good for society"), so the
    full option often never appears contiguously. What follows the enumeration is
    RESTORED (it is content: "to economic inequality"), unless the option already
    ends in it (it is the shared "...for society" tail).
    """
    low = core.lower()

    def _find(t: str) -> tuple[int, int]:
        i = low.find(t)
        ok = i > 0 and (low[i - 1].isspace() or low[i - 1] in "(\"'")
        return (i, i + len(t)) if ok else (-1, -1)

    for pass_i in (0, 1):
        spans = []
        for o in options:
            t = _norm(o).lower().rstrip(".")
            if pass_i:                                   # head-word pass
                t = t.split(" ", 1)[0].strip(",")
                if t in _WEAK_HEAD:
                    continue
            if len(t) < (3 if pass_i else 4):            # 'bad', 'yes' are heads
                continue
            s, e = _find(t)
            if s > 0:
                spans.append((s, e))
        # >=2 echoes on the head pass: one common word is a coincidence, the
        # option list showing up in the stem is not.
        if not spans or (pass_i and len(spans) < 2):
            continue
        prefix = _deinvert(core[:min(s for s, _ in spans)].strip(" ,;:"))
        if not prefix:
            continue
        tail = prefix.rsplit(" ", 1)[-1].lower()
        if tail in _DANGLING or (pass_i == 0 and tail not in _COPULA_TAIL):
            continue
        rest = core[max(e for _, e in spans):].strip(" ,;:")
        rest = re.sub(r"^(?:or|and)\s+", "", rest, flags=re.IGNORECASE).strip(" ,;:")
        if rest and (_low(option).lower().endswith(rest.lower())
                     or rest.lower() in _low(option).lower()):
            rest = ""                                    # the option already says it
        if re.match(r"^(?:\w+\s+)?not\b", rest, re.IGNORECASE):
            rest = ""                                    # "...will or WILL NOT happen"
        return f"{_cap(prefix)} {_low(option)}" + (f" {rest}" if rest else "")
    return None


_CLAUSE_VERB = re.compile(
    r"\b(is|are|was|were|be|should|would|will|can|must|do|does|did|has|have|"
    r"makes|make|helps|help|hurts|hurt|comes|goes|gets|needs|takes|says)\b",
    re.IGNORECASE)


def _option_is_statement(option: str) -> str | None:
    """Some ATP items put the whole position IN the option ('Which comes closest
    to your view?' -> 'Government regulation of business is necessary to protect
    the public interest'). Then the option already IS the declarative statement
    and any frame we wrap around it only dilutes the embedding."""
    o = _norm(option)
    if len(o.split()) >= 6 and _CLAUSE_VERB.search(o):
        return o
    return None


# Strict: 'Yes' / 'No, have never owned a gun' are polar answers; 'No impact
# either way' and 'Not a reason' are not, and treating them as polar produced
# statements that negated the wrong thing.
_YESNO = re.compile(r"^(yes|no)(?:\s*[,:]\s*(?P<tail>.+))?$", re.IGNORECASE)


def _yes_no(core: str, option: str) -> str | None:
    """Polar 'Do you think X?' -> 'Yes, I think X.' / 'No, I do not think X.'"""
    m = _YESNO.match(_norm(option).rstrip("."))
    if not m or not _DOYOUTHINK.match(core):
        return None
    body = _strip_frame(core)
    if m.group(1).lower() == "yes":
        return f"Yes, I think {body}"
    return f"No, I do not think {body}"


# ATP batteries: one framing question, then the item, separated by the question
# mark ("Would you favor or oppose the following? If robots were limited to...").
# The item carries nearly all the discriminating content, so it must survive.
_BATTERY_SLOT = re.compile(
    r"\b(?:the following(?: words or phrases| statements?| things?)?|"
    r"each of (?:these|the following)|these things|the statements below)\b",
    re.IGNORECASE)


_BATTERY_MARK = re.compile(
    r"\b(?:the following|each one|for each|each of these|whether|please indicate|"
    r"please tell us|is for them to|these things)\b", re.IGNORECASE)


def _battery_split(question: str) -> tuple[str, str]:
    """(framing stem, item) for a battery question; ('', '') when it is not one.

    The separator is '?' most of the time, but ATP also uses ':' ("...how
    important is it for them to: Keep all of their guns unloaded") and a plain
    '.' ("...a reason why you own a gun. For protection"). The weaker separators
    only count when the stem looks like a battery frame -- otherwise every
    two-sentence question would be split at its first period.
    """
    q = _norm(question)
    for sep, needs_mark in (("?", False), (":", True), (". ", True)):
        best = ("", "")
        i = q.find(sep)
        while 0 < i < len(q) - len(sep):
            stem, item = q[:i].strip(), q[i + len(sep):].strip(" ?.")
            if (item and len(stem.split()) >= 3
                    and (not needs_mark or (_BATTERY_MARK.search(stem)
                                            and item[:1].isupper()))):
                best = (stem, item)     # keep the LAST valid split: "...the U.S. is
                #                         going in the wrong direction. Colleges..."
                #                         must break at the item, not at "U.S."
            i = q.find(sep, i + 1)
        if best[0]:
            return best
    return "", ""


def to_statement(question: str, option: str,
                 options: list[str] | None = None) -> tuple[str, str]:
    """(declarative statement, rule name). rule == 'fallback' when nothing matched.

    Fallback is the CURRENT behaviour -- "<question> <option>" -- so a miss costs
    nothing relative to today and is counted, not hidden.
    """
    q, opt = _norm(question), _norm(option)
    options = [_norm(o) for o in (options or [opt])]

    self_contained = _option_is_statement(opt)
    if self_contained:
        return _sentence(self_contained), "option_is_statement"

    # Battery item FIRST: the item lives after the stem's own question mark, so a
    # rule run on the raw string splices the option into the framing question and
    # leaves the item dangling. Splice the item into the frame's slot instead
    # ("...worry about THE FOLLOWING happening to you?" + "Being the victim of a
    # mass shooting"), or state the item and then the position when there is no slot.
    stem, item = _battery_split(q)
    if stem:
        hit = _battery_statement(stem, item, opt, options)
        if hit:
            return hit

    hit = _apply(q, opt, options)
    if hit:
        return hit

    return f"{q} {opt}".strip(), "fallback"      # == today's embed_text


def _battery_statement(stem: str, item: str, opt: str,
                       options: list[str]) -> tuple[str, str] | None:
    """Frame + item -> one statement.

    Two ways to combine them, and which reads better depends on the item:
      splice   the item into the frame's slot -- right for a PHRASE item
               ("...worry about |being the victim of a mass shooting| happening")
      two-sentence  state the item, then the position about it -- right for a
               CLAUSE item, where splicing produces two finite verbs in one
               clause ("I think inequality would be worse are likely to happen").
    The item is never dropped either way; it carries the discriminating content.
    """
    slotted = _BATTERY_SLOT.sub(_uncap(item), stem, count=1)
    has_slot = slotted != stem

    def _splice():
        # echo cutting is off here: the item's own verbs alias the option heads
        # ("Doctors WILL rely..." vs "Will definitely happen") and the cut lands
        # inside the item.
        h = _apply(slotted, opt, options, allow_echo=False) if has_slot else None
        return (h[0], f"battery_{h[1]}") if h else None

    def _continue():
        # ":" batteries: the item finishes the stem's own sentence.
        h = _apply(f"{stem} {_uncap(item)}", opt, options)
        return (h[0], f"batteryc_{h[1]}") if h else None

    def _two_sentence():
        frame = _BATTERY_SLOT.sub("this", stem, count=1) if has_slot else f"{stem} this"
        h = _apply(frame, opt, options)
        return (f"{_sentence(item)} {h[0]}", f"battery2_{h[1]}") if h else None

    # A clause item splices badly (two finite verbs in one clause); a phrase item
    # splices well. Either way the ITEM MUST SURVIVE -- a rule that matches the
    # frame and drops the item produces a statement about nothing, which is worse
    # than the fallback.
    order = ((_two_sentence, _continue, _splice) if _option_is_statement(item)
             else (_splice, _continue, _two_sentence))
    keep = " ".join(item.split()[:3]).lower()
    for f in order:
        out = f()
        if out and keep in out[0].lower():
            return out
    return None


def _apply(q: str, opt: str, options: list[str],
           allow_echo: bool = True) -> tuple[str, str] | None:
    """One pass of the rule table over a question. None when nothing fired."""
    lead, core = split_lead(_dehedge(q))

    def _finish(stmt: str) -> str:
        if lead:
            stmt = f"{_cap(lead)}, {_uncap(stmt)}"
        return _sentence(stmt)

    for target in (core, _dehedge(q).rstrip(" ?.")):  # clean clause, then the raw one
        yn = _yes_no(target, opt)
        if yn:
            return (_finish(yn) if target is core else _sentence(yn)), "yes_no"
        # Frame-stripped first (a declarative clause is what most rules want),
        # then the raw target -- the "how much ... do you think" rules match the
        # interrogative and would be starved by the strip.
        for body in dict.fromkeys((_strip_frame(target), target)):
            for name, pat, fn in RULES:
                m = pat.match(body)
                if m:
                    try:
                        out = fn(m, opt)
                    except Exception:           # a rule must never break the build
                        continue
                    if out and len(out.split()) >= 3 and _is_clean(out):
                        return (_finish(out) if target is core else _sentence(out)), name
            cut = _echo_cut(body, opt, options) if allow_echo else None
            if cut and _is_clean(cut):
                return (_finish(cut) if target is core else _sentence(cut)), "option_echo"
    return None


# ---------------------------------------------------------------------------
# LLM backend (optional; OpenAI-compatible, same client as retrieval.answer)
# ---------------------------------------------------------------------------
# Surface rephrasing ONLY. The survey supplies WHAT the positions are and how
# prevalent each is; the model is not allowed to touch either. Anything the model
# adds is content the reward would then credit a response for expressing without
# any survey evidence that anyone holds it.
LLM_SYSTEM = (
    "You rewrite survey answer options as plain declarative sentences.\n"
    "You are given a survey question, its full list of answer options, and ONE "
    "target option. Write the single sentence a person who chose the TARGET "
    "option would say to state their position.\n"
    "Rules:\n"
    "- Rephrase the surface form ONLY. Do not add any claim, reason, example, "
    "qualifier, or justification that is not already in the question or the "
    "target option.\n"
    "- Do not drop, soften, or strengthen anything that is there. Preserve the "
    "degree word exactly ('somewhat' stays 'somewhat', 'a major reason' stays "
    "major).\n"
    "- Do not mention how common the position is, who holds it, or the other "
    "options. Prevalence comes from the survey, never from you.\n"
    "- Do not hedge, editorialize, or mention the survey.\n"
    "- One sentence. No quotes, no preamble, no explanation."
)


def llm_statement(question: str, option: str, options: list[str],
                  base_url: str, model: str, timeout: float = 60.0) -> str:
    from retrieval.answer import chat

    user = (f"Question: {question}\n"
            f"Options: {'; '.join(options)}\n"
            f"Target option: {option}\n"
            f"Sentence:")
    out = chat(base_url, model,
               [{"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": user}],
               temperature=0.0, max_tokens=96, timeout=timeout)
    return _sentence(_norm(out).strip('"').split("\n")[0])


# ---------------------------------------------------------------------------
# Graph -> (question, options) leaves
# ---------------------------------------------------------------------------
def leaf_items(graph) -> list[tuple[str, list[str], list[str]]]:
    """Unique (question, options, full_texts) over the graph's opinion leaves.

    opinion_texts stores '<question> <option>' per option; the shared prefix is
    the question. Leaves of the same question repeat across subgroups, so the
    artifact is keyed by TEXT, not by node.
    """
    seen: dict[str, tuple[str, list[str], list[str]]] = {}
    for _nid, texts in getattr(graph, "opinion_texts", {}).items():
        if not texts:
            continue
        pref = os.path.commonprefix(texts)
        cut = pref.rfind(" ")
        pref = pref[:cut + 1] if cut > 0 else pref
        q = pref.strip()
        opts = [t[len(pref):].strip(" \t\"'-") or t.strip() for t in texts]
        key = statement_key(q)
        if key not in seen:
            seen[key] = (q, opts, [t.strip() for t in texts])
    return list(seen.values())


def load_graph(dataset: str, seed: int):
    if dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        return load_opinionqa(split_seed=seed, leakage_safe=True)
    from data.loaders.globalopinionqa import load_globalopinionqa
    return load_globalopinionqa(split_seed=seed, leakage_safe=True)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_records(items, backend: str, args) -> tuple[list[dict], dict]:
    """Records + counters. Template runs for every row; llm only rewrites rows
    the templates missed (``--llm_all`` overrides), because a template hit is
    already faithful by construction and an LLM call is not."""
    recs, n_rule = [], {}
    for q, opts, texts in items:
        for opt, full in zip(opts, texts):
            stmt, rule = to_statement(q, opt, opts)
            recs.append({"key": statement_key(full), "question": q, "option": opt,
                         "statement": stmt, "rule": rule})
            n_rule[rule] = n_rule.get(rule, 0) + 1

    matched = sum(v for k, v in n_rule.items() if k != "fallback")
    stats = {"n": len(recs), "n_rule": n_rule,
             "template_match_rate": matched / len(recs) if recs else 0.0}

    if backend == "llm":
        todo = [r for r in recs
                if args.llm_all or r["rule"] == "fallback"][:args.llm_limit or None]
        opts_by_q = {statement_key(q): o for q, o, _ in items}
        n_ok = n_err = 0
        for i, r in enumerate(todo, 1):
            try:
                r["statement"] = llm_statement(
                    r["question"], r["option"],
                    opts_by_q.get(statement_key(r["question"]), [r["option"]]),
                    args.base_url, args.model)
                r["rule"] = "llm"
                n_ok += 1
            except Exception as e:                # keep the template/fallback text
                n_err += 1
                if n_err <= 3:
                    print(f"  llm error ({e}) on: {r['question'][:60]}", file=sys.stderr)
            if i % 100 == 0:
                print(f"  llm {i}/{len(todo)} ({n_err} errors)", file=sys.stderr)
        stats.update(llm_rewritten=n_ok, llm_errors=n_err)
    return recs, stats


def write_artifact(path: str, recs: list[dict], meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Self-test: hand-written ATP-style pairs, printed before/after
# ---------------------------------------------------------------------------
# One item per common ATP frame, plus a deliberate miss (the last one) so the
# fallback path is visible. Asserted lightly and PRINTED in full: the value of
# this transform is legibility, and an assertion cannot see whether the output
# reads like something a person would say.
_SELFTEST_ITEMS: list[tuple[str, list[str]]] = [
    ("On balance, do you think the growing number of guns in the country is good "
     "or bad for society?",
     ["Good for society", "Bad for society"]),
    ("Do you think race relations in the United States are generally good or "
     "generally bad?",
     ["Generally good", "Generally bad"]),
    ("Do you think the number of legal immigrants the U.S. admits should be "
     "increased, decreased, or kept about the same?",
     ["Increased", "Decreased", "Kept about the same"]),
    ("In general, how important, if at all, is it to you for someone in a top "
     "executive business position to be willing to take risks?",
     ["Essential", "Important, but not essential", "Not important"]),
    ("How much, if at all, is your partner not being ready financially a reason "
     "why you are not engaged or married to your current partner?",
     ["Major reason", "Minor reason", "Not a reason"]),
    ("How much of a problem, if any, is made-up news and information in the "
     "country today?",
     ["A very big problem", "A moderately big problem", "A small problem",
      "Not a problem at all"]),
    ("How much confidence, if any, do you have in scientists to act in the best "
     "interests of the public?",
     ["A great deal of confidence", "A fair amount of confidence",
      "Not too much confidence", "No confidence at all"]),
    ("Do you favor or oppose stricter gun laws in the United States?",
     ["Strongly favor", "Favor", "Oppose", "Strongly oppose"]),
    ("In the last 12 months, how often did you eat dinner with any of the other "
     "members of your household?",
     ["Basically every day", "A few times a week", "A few times a month",
      "Less than once a month"]),
    ("Do you think the federal government should be doing more to reduce the "
     "effects of climate change?",
     ["Yes", "No"]),
    ("Which of these comes closest to your view about how much the government "
     "should regulate business?",
     ["Government regulation of business is necessary to protect the public "
      "interest", "Government regulation of business usually does more harm "
      "than good"]),
    # battery: one framing question, one item after the separator
    ("For each one of the following, please indicate whether you think it is a "
     "reason why there aren't more women in top executive business positions. "
     "Many businesses are not ready to hire women for top executive positions",
     ["Major reason", "Minor reason", "Not a reason"]),
    # deliberate miss: nothing in the rule table frames a "which comes closest"
    # stem whose options are bare labels, so this falls back -- visibly.
    ("Which of these do you think is the bigger problem in the country today?",
     ["Racism", "Political correctness"]),
]


def _selftest() -> None:
    print("=== build_position_statements: template backend ===\n")
    n_hit = n_tot = 0
    for q, opts in _SELFTEST_ITEMS:
        print(f"Q: {q}")
        for o in opts:
            stmt, rule = to_statement(q, o, opts)
            n_tot += 1
            n_hit += rule != "fallback"
            print(f"   before  {q} {o}")
            print(f"   after   [{rule}] {stmt}")
        print()
    rate = n_hit / n_tot
    print(f"template match rate: {n_hit}/{n_tot} = {rate:.2f} "
          f"({n_tot - n_hit} fall back to '<question> <option>', i.e. today's "
          f"behaviour)")
    print("  ^ these items were chosen to exercise the rules, so this rate is an "
          "upper bound.\n    Measured over all 2,004 raw ATP questions (6,782 "
          "positions): 0.50.\n    The build prints the real rate for the graph "
          "it is run on; --backend llm covers the tail.")

    # Invariants, not quality: quality is what the printout above is for.
    q, opts = _SELFTEST_ITEMS[0]
    s_good, r_good = to_statement(q, opts[0], opts)
    s_bad, _ = to_statement(q, opts[1], opts)
    assert r_good != "fallback", (s_good, r_good)
    assert s_good != s_bad, s_good                       # options stay distinguishable
    assert "?" not in s_good, s_good                     # declarative, not a question
    assert s_good[0].isupper() and s_good.endswith("."), s_good
    # a fallback is exactly today's string, so a miss never regresses anything
    miss, rule = to_statement("An unparseable stem", "Some option")
    assert rule == "fallback" and miss == "An unparseable stem Some option", miss
    # no rule may silently drop the option's degree word
    s, _ = to_statement(_SELFTEST_ITEMS[5][0], "A small problem",
                        _SELFTEST_ITEMS[5][1])
    assert "small problem" in s.lower(), s
    # a battery item must survive: it is where the content is
    bq, bopts = _SELFTEST_ITEMS[11]
    s, rule = to_statement(bq, bopts[0], bopts)
    assert "many businesses" in s.lower(), (rule, s)

    n_round = _selftest_roundtrip()
    print("\nself-test OK (invariants: declarative, option-distinct, degree "
          "preserved, battery item kept, fallback == current behaviour;"
          f" {n_round} statements round-tripped through alignment.reward)")


def _selftest_roundtrip() -> int:
    """Builder -> artifact -> reward, on a stub graph.

    The contract between the two files is a single string key, and a mismatch
    there fails SILENTLY (every lookup misses, every position falls back to the
    old text, and the reward is broken exactly as before but now invisibly).
    So it is checked end to end rather than assumed.
    """
    import tempfile

    from alignment.reward import load_position_statements, positions_from_subtree

    q = ("On balance, do you think the growing number of guns in the country is "
         "good or bad for society?")
    opts = ["Good for society", "Bad for society"]

    class _G:
        children_indices = [[1], [], []]
        opinion_texts = {1: [f"{q} {o}" for o in opts]}
        opinion_dist = {1: [0.45, 0.55]}

    items = leaf_items(_G())
    assert len(items) == 1 and items[0][1] == opts, items

    class _Args:                                   # template backend, no network
        llm_all = llm_limit = 0
        base_url = model = ""

    recs, stats = build_records(items, "template", _Args())
    path = os.path.join(tempfile.mkdtemp(), "position_statements.jsonl")
    write_artifact(path, recs, {"schema_version": SCHEMA_VERSION,
                                "backend": "template", "created": "selftest",
                                **{k: v for k, v in stats.items() if k != "n_rule"}})

    stmts = load_position_statements(path)
    pos = positions_from_subtree(_G(), 0, statements=stmts)
    assert len(pos) == 2, pos
    assert all(p.embed_text.endswith("for society.") for p in pos), pos
    assert {p.option for p in pos} == set(opts)          # labels unchanged
    assert abs(sum(p.prevalence for p in pos) - 1.0) < 1e-9, pos   # prevalence unchanged
    return len(stmts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the frozen (question, option) -> declarative statement artifact")
    ap.add_argument("--backend", choices=["template", "llm"], default="template")
    ap.add_argument("--dataset", choices=["opinionqa", "globalopinionqa"],
                    default="opinionqa")
    ap.add_argument("--seed", type=int, default=42, help="graph split seed")
    ap.add_argument("--out", default="artifacts/position_statements.jsonl")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing artifact. It is FROZEN by design: "
                         "regenerating it silently would change every reward score "
                         "mid-phase and make runs incomparable.")
    ap.add_argument("--limit", type=int, default=0, help="first N questions (debug)")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--llm_all", action="store_true",
                    help="rewrite EVERY row with the LLM, not just template misses")
    ap.add_argument("--llm_limit", type=int, default=0, help="cap LLM calls (debug)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if os.path.exists(args.out) and not args.force:
        ap.error(f"{args.out} exists and the artifact is frozen; pass --force to rebuild")
    if args.backend == "llm" and not args.model:
        ap.error("--backend llm needs --model")

    graph = load_graph(args.dataset, args.seed)
    items = leaf_items(graph)
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} unique questions from {args.dataset}")

    recs, stats = build_records(items, args.backend, args)
    meta = {"schema_version": SCHEMA_VERSION, "backend": args.backend,
            "dataset": args.dataset, "seed": args.seed,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model if args.backend == "llm" else "",
            "n_questions": len(items), **stats,
            "note": "FROZEN: consumed by alignment.reward; rebuilding changes every "
                    "reward score. Statements are surface rewrites of survey "
                    "(question, option) pairs -- positions and prevalences are "
                    "unchanged."}
    write_artifact(args.out, recs, meta)

    print(f"wrote {args.out}: {stats['n']} positions, "
          f"template match rate {stats['template_match_rate']:.3f}")
    for k, v in sorted(stats["n_rule"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v:6d}  {v / stats['n']:.3f}")
    for r in recs[:5]:
        print(f"  [{r['rule']}] {r['statement']}")


if __name__ == "__main__":
    main()

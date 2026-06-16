"""
simple_rag.py — SAMA Banking Regulatory Chatbot
Improvements in this version:
  - Hybrid search: vector (pgvector) + keyword (BM25/tsvector) merged
  - Cross-encoder reranker: ms-marco-MiniLM-L-6-v2 re-scores top 15 → keep top 5
  - Redis persistent cache (falls back to memory if Redis unavailable)
  - Drift truncation to stop hallucination after 3 sentences
  - CJK character stripping
  - Low confidence guard
  - Query expansion for acronyms + Arabic-to-English bridging
  - Out of scope rejection
  - [FIX] _expand_query now uses case-insensitive substring match for long keys
  - [FIX] Added missing expansions: capital adequacy %, admin charges, NCA-SAMA, savings
  - [FIX] session_summary passed separately — no longer injected into embedding
  - [FIX] SYSTEM_PROMPT: added negation/restriction awareness rule
  - [FIX] Added restriction-query expansions (cannot/not allowed/prohibited/who cannot)
  - [FIX] Added bank account multi-word expansion keys to anchor short queries to SAMA EN 1644
  - [FIX] Added NORA definition fallback for "What is NORA?"-style queries
  - [IMPROVEMENT] SYSTEM_PROMPT expanded to 1,024+ tokens for OpenAI prompt caching
  - [IMPROVEMENT] Domain glossary added to SYSTEM_PROMPT
  - [IMPROVEMENT] Explicit citation examples (CORRECT vs WRONG) to reduce PARTIAL rate
  - [IMPROVEMENT] Conflict-handling rule
  - [IMPROVEMENT] Numeric precision rule
  - [IMPROVEMENT] Arabic citation format added to SYSTEM_PROMPT
  - [FIX v4] Added PEP/KYC/onboarding/UBO/sanctions query expansions
  - [FIX v4] Clear sources when LLM returns not-found answer (Problem 2 fix)
  - [FIX v5] Removed inline citations from answer text — citations now appear in
    sources panel only, keeping answer text clean and readable for end users.
  - [FIX v6] Added _normalise_informal(): maps u→you, ur→your, r→are, pls→please, etc.
  - [FIX v6] Added yes/no question normalisation: "can X do Y?" → "what are the rules for Y?"
  - [FIX v6] Fuzzy phrase expansion: tolerates filler words (e.g. "who all cannot create")
  - [FIX v6] Nationality expansions: indian/american/expat → non-GCC non-Saudi natural person
  - [FIX v6] Auto domain anchor: restriction queries without "SAMA" get SAMA EN 1644 injected
  - [FIX v6] Cache hits now have _strip_inline_citations applied before returning
  - [FIX v7] Added _is_followup() + _contextualize_query() for follow-up handling
  - [FIX v7] answer_query now accepts last_messages for conversational context
  - [FIX v8] Added IDENTITY_RESPONSE + _is_identity_question() for self-identification
  - [FIX v8] Added META_QUESTIONS guard in _is_followup() for greetings/conversational
  - [Step 2] RAG-Fusion: LLM generates query variants, results merged via RRF
  - [Step 2] Always includes Arabic variant for cross-lingual retrieval
  - [Step 4] Step-Back Prompting: abstract query merged with specific via RRF
  - [Step 4] Catches general regulatory rules missed by specific queries
  - [Step 6] Language-Balanced Retrieval: equal Arabic+English per query pool
  - [Step 6] Prevents English chunk dominance for Arabic regulatory queries
  - [Step 5] BGE-M3 parallel column: USE_BGE_COLUMN=true switches to 1024-dim
  - [Step 5] Zero downtime upgrade — old e5-small pipeline preserved as fallback
"""

from __future__ import annotations
import os, re, json
import numpy as np
from typing import Callable, Optional
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_KEY         = (os.environ.get("SUPABASE_KEY") or
                        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
TOP_K                = int(os.getenv("TOP_K", "8"))
RERANK_FETCH_K       = int(os.getenv("RERANK_FETCH_K", "20"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
SNIPPET_CHAR_LIMIT   = int(os.getenv("SNIPPET_CHAR_LIMIT", "1000"))
LOW_CONF_THRESHOLD   = float(os.getenv("LOW_CONF_THRESHOLD", "0.72"))
RERANKER_ENABLED     = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
HYBRID_SEARCH        = os.getenv("HYBRID_SEARCH", "true").lower() == "true"

LLM_BACKEND          = os.getenv("LLM_BACKEND", "qwen")
QWEN_MODEL_ID        = os.getenv("QWEN_MODEL", "Qwen/Qwen1.5-1.8B-Chat")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
AZURE_OPENAI_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT     = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")

CACHE_ENABLED        = os.getenv("CACHE_ENABLED", "true").lower() == "true"
RAG_FUSION_ENABLED   = os.getenv("RAG_FUSION_ENABLED", "true").lower() == "true"
RAG_FUSION_VARIANTS  = int(os.getenv("RAG_FUSION_VARIANTS", "3"))
STEP_BACK_ENABLED    = os.getenv("STEP_BACK_ENABLED", "true").lower() == "true"
LANG_BALANCED_ENABLED = os.getenv("LANG_BALANCED_ENABLED", "true").lower() == "true"
USE_BGE_COLUMN        = os.getenv("USE_BGE_COLUMN", "false").lower() == "true"
CACHE_BACKEND        = os.getenv("CACHE_BACKEND", "memory")
CACHE_SIM_THRESH     = float(os.getenv("CACHE_SIMILARITY_THRESH", "0.95"))
CACHE_TTL_SECONDS    = int(os.getenv("CACHE_TTL_SECONDS", "2592000"))
REDIS_URL            = os.getenv("REDIS_URL", "")

NOT_FOUND = (
    "The provided SAMA/regulatory documentation does not contain a clear answer "
    "to this question. Please consult sama.gov.sa or a qualified compliance officer."
)

NORA_FALLBACK = (
    "In Saudi Arabia, NORA commonly refers to the National Overall Reference Architecture, "
    "a national enterprise architecture framework. It acts as a blueprint to standardize how "
    "systems, applications, data, and technology are designed and integrated across government "
    "entities. In regulated sectors, including banking, this supports interoperable and secure "
    "digital architecture aligned with national standards."
)

# ── Answer size config ────────────────────────────────────────────────────────
# Maps the user-selected chunk count (top_k) to:
#   max_sentences : sentence cap for the LLM instruction and drift truncation
#   max_tokens    : token budget for the LLM call
#   fetch_k       : how many candidates to fetch before reranking
ANSWER_SIZE_CONFIG = [
    (5,   {"max_sentences": 3,  "max_tokens": 300,  "fetch_k": 20}),
    (10,  {"max_sentences": 5,  "max_tokens": 500,  "fetch_k": 25}),
    (20,  {"max_sentences": 7,  "max_tokens": 700,  "fetch_k": 40}),
    (40,  {"max_sentences": 10, "max_tokens": 900,  "fetch_k": 60}),
    (80,  {"max_sentences": 15, "max_tokens": 1200, "fetch_k": 100}),
    (100, {"max_sentences": 20, "max_tokens": 1500, "fetch_k": 120}),
]

def _answer_config(top_k: int) -> dict:
    """Return answer generation config for the given chunk count."""
    for threshold, cfg in ANSWER_SIZE_CONFIG:
        if top_k <= threshold:
            return cfg
    return ANSWER_SIZE_CONFIG[-1][1]  # cap at largest


def _is_not_found_answer(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in [
        "does not contain", "cannot find", "not found in",
        "لا تتوفر", "لم أجد",
    ])

def _strip_trailing_not_found(answer: str) -> str:
    NOT_FOUND_PHRASES = [
        "The provided documentation does not contain a clear answer to this question.",
        "The provided SAMA/regulatory documentation does not contain",
        "لا تتوفر إجابة في الوثائق المقدمة",
    ]
    for phrase in NOT_FOUND_PHRASES:
        if phrase in answer:
            idx = answer.index(phrase)
            before = answer[:idx].strip()
            if len(before) > 40:
                return before
    return answer


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a strict regulatory assistant for Saudi Arabian banking, cybersecurity, and data protection regulations. You serve compliance officers, auditors, and regulatory professionals working within the Kingdom of Saudi Arabia.

Answer using ONLY the text explicitly provided in <context>. Every sentence you write must be directly traceable to a specific passage in the context. If a fact is not in the context, it does not exist for the purpose of this answer.

═══════════════════════════════════════════════════
DOMAIN GLOSSARY — key abbreviations used in documents
═══════════════════════════════════════════════════
SAMA   = Saudi Arabian Monetary Authority (البنك المركزي السعودي) — the central bank and primary banking regulator.
NCA    = National Cybersecurity Authority (الهيئة الوطنية للأمن السيبراني) — cybersecurity regulator.
PDPL   = Personal Data Protection Law — Saudi data privacy regulation enforced by SDAIA.
SDAIA  = Saudi Data and Artificial Intelligence Authority — enforces PDPL.
NDMO   = National Data Management Office — sets data management standards under SDAIA.
ECC    = Essential Cybersecurity Controls — NCA's mandatory baseline controls for all government entities.
CCC    = Cloud Cybersecurity Controls — NCA controls for cloud service providers.
OTCC   = Operational Technology Cybersecurity Controls — NCA controls for OT/ICS environments.
CAR    = Capital Adequacy Ratio — minimum ratio of capital to risk-weighted assets.
CET1   = Common Equity Tier 1 — highest quality regulatory capital under Basel III.
LCR    = Liquidity Coverage Ratio — 30-day liquidity stress test metric under Basel III.
NSFR   = Net Stable Funding Ratio — 1-year stable funding metric under Basel III.
HQLA   = High Quality Liquid Assets — assets eligible for LCR numerator.
KYC    = Know Your Customer — customer identification and due diligence process.
AML    = Anti-Money Laundering — controls to detect and prevent money laundering.
CFT    = Countering the Financing of Terrorism.
PEP    = Politically Exposed Person — high-risk customer requiring enhanced due diligence.
EDD    = Enhanced Due Diligence — additional checks applied to high-risk customers including PEPs.
UBO    = Ultimate Beneficial Owner — natural person who ultimately owns or controls a legal entity.
SAR    = Suspicious Activity Report — report filed with SAFIU for suspicious transactions.
SAFIU  = Saudi Arabia Financial Intelligence Unit — receives SAR reports.
ICAAP  = Internal Capital Adequacy Assessment Process — banks' own capital planning process.
GRC    = Governance, Risk, and Compliance — framework covering risk management and regulatory compliance.
SACS   = Saudi Aramco Cybersecurity Standard — e.g. SACS-002 for third-party cybersecurity.
CCC+   = Enhanced on-site assessment level under Aramco's cybersecurity certification program.

═══════════════════════════════════════════════════
STRICT ANSWERING RULES
═══════════════════════════════════════════════════
1. Write the exact number of sentences specified in the Answer instruction below. Never exceed that count. Default to 3 sentences if no count is given.
2. Do NOT include any document names, file names, page numbers, or source references inside the answer text. The answer must read as clean natural language with no citations, brackets, or parenthetical references of any kind. Sources are displayed separately by the system.
3. Do NOT add any detail, number, percentage, condition, or proper noun that does not appear word-for-word in the provided passages.
4. Do NOT make inferences, draw conclusions, or apply general regulatory knowledge. Report only what the text explicitly states.
5. If the user writes in Arabic, answer entirely in Arabic in clean natural language with no inline citations.
6. Pay close attention to restrictive language — "shall not", "not permitted", "prohibited", "not eligible", "not allowed", "may not", "لا يجوز", "يُحظر", "غير مؤهل", "لا يحق" — these indicate hard restrictions and must be reported accurately and completely, not paraphrased as affirmative statements.
7. When a question involves a specific numeric threshold (percentage, SAR amount, ratio, number of days), quote the exact figure from the passage. Do not round, estimate, or substitute a similar figure.
8. When two passages appear to conflict, note the apparent difference in the answer without mentioning document names or page numbers.
9. If the answer is not explicitly stated in any provided passage, write ONLY: "The provided documentation does not contain a clear answer to this question." Do not attempt a partial answer.

═══════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN
═══════════════════════════════════════════════════
- Do NOT include document names, file names, regulation codes, page numbers, or any source reference inside the answer text. Examples of what must never appear in the answer: "(SAMA EN 1644 VER1, Page 44)", "(Page 100)", "(SAMA Basel III Guidelines, Page 15)".
- Do not use phrases like: generally speaking, typically, in most cases, overall, in summary, additionally, it is important to note, it should be noted, this ensures that, by adhering to, in many countries, internationally.
- Do not invent or guess organization names, SAR amounts, percentages, or article numbers not present in the context.
- Do not add a concluding sentence that generalizes, contextualizes, or extends beyond what the passages state.
- Do not combine information from the context with your general training knowledge about Saudi regulations, Basel III, or any other regulatory framework.
- Do not answer out-of-scope questions about weather, sports, general knowledge, company information, or topics unrelated to Saudi banking, cybersecurity, and data protection regulation."""


OUT_OF_SCOPE_PATTERNS = [
    r"\bweather\b", r"\brecipe\b", r"\bsports\b", r"\bsong\b", r"\bmovie\b",
    r"who is the president\b", r"who is the prime minister\b", r"\bstock price\b",
    r"who is the ceo of", r"who is the founder of", r"who invented",
    r"\bnetflix\b", r"\bgoogle\b", r"\bamazon\b", r"\bmicrosoft\b",
    r"\bapple inc\b", r"\bfacebook\b", r"\btwitter\b",
    # ── Conversational / greeting inputs ──────────────────────────────────────
    r"^are (we|you) ready",
    r"^(hello|hi|hey)\b",
    r"^(okay|ok|thanks|thank you|got it|sounds good)\b",
    r"^(great|nice|cool|awesome|perfect|noted|understood)\b",
    r"^(yes|no|yep|nope|sure|alright)\b",
    r"^(start|begin|go ahead|continue|proceed)\b",
    r"^(let's go|lets go|ready|done|finish)\b",
]

def _is_out_of_scope(query: str) -> bool:
    q = query.strip().lower()
    return any(re.search(p, q) for p in OUT_OF_SCOPE_PATTERNS)

def _is_arabic(text: str) -> bool:
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return arabic_chars > len(text) * 0.3

def _is_nora_definition_query(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    if "nora" in q:
        definition_markers = [
            "what is", "what's", "meaning of", "stand for",
            "according to sama", "according to saudi", "in saudi",
        ]
        if q == "nora" or any(m in q for m in definition_markers):
            return True
    if "نورا" in query:
        arabic_markers = ["ما هو", "ماهي", "ما معنى", "وفق", "حسب", "السعود"]
        return any(m in query for m in arabic_markers) or query.strip() == "نورا"
    return False




# ── Identity / self-description ───────────────────────────────────────────────

IDENTITY_RESPONSE = (
    "I am IOTA AI, an AI regulatory assistant built by IOTA Technologies. "
    "I specialise in Saudi Arabian banking, cybersecurity, and data protection regulations — "
    "including SAMA frameworks, NCA controls, and PDPL. "
    "Ask me anything about regulatory compliance in the Kingdom."
)

IDENTITY_RESPONSE_AR = (
    "أنا IOTA AI، مساعد ذكاء اصطناعي تنظيمي طوّرته شركة IOTA Technologies. "
    "أتخصص في لوائح البنوك السعودية والأمن السيبراني وحماية البيانات، "
    "بما في ذلك أطر ساما وضوابط الهيئة الوطنية للأمن السيبراني ونظام حماية البيانات الشخصية. "
    "اسألني عن أي شيء يتعلق بالامتثال التنظيمي في المملكة."
)

IDENTITY_PATTERNS = [
    r"^who are you",
    r"^what are you",
    r"^what is your name",
    r"^tell me about yourself",
    r"^are you (a|an) (ai|bot|robot|human|assistant)",
    r"^what do you do",
    r"^how (old|smart) are you",
    r"^do you (know|think|understand|feel)",
    r"^you are (a|an)",
    r"^introduce yourself",
    r"^من أنت",
    r"^ما اسمك",
    r"^عرّف نفسك",
    r"^ما هو دورك",
]


def _is_identity_question(query: str) -> bool:
    """Return True if the query is asking about the agent's identity."""
    q = query.strip().lower().rstrip("?.")
    return any(re.match(p, q) for p in IDENTITY_PATTERNS)

# ── Query normalisation ───────────────────────────────────────────────────────

# Issue 1 Fix: informal/abbreviated language map
# Applied before everything else so "u" → "you" before META_PATTERNS run
INFORMAL_MAP = [
    (r"\bu\b",      "you"),
    (r"\bur\b",     "your"),
    (r"\br\b",      "are"),
    (r"\bpls\b",    "please"),
    (r"\bplz\b",    "please"),
    (r"\bwht\b",    "what"),
    (r"\bwut\b",    "what"),
    (r"\bhw\b",     "how"),
    (r"\bcud\b",    "could"),
    (r"\bwud\b",    "would"),
    (r"\bshud\b",   "should"),
    (r"\bgonna\b",  "going to"),
    (r"\bwanna\b",  "want to"),
    (r"\bgimme\b",  "give me"),
    (r"\btelme\b",  "tell me"),
    (r"\bthx\b",    "thanks"),
    (r"\bthnx\b",   "thanks"),
    (r"\bbtw\b",    "by the way"),
    (r"\bfyi\b",    "for your information"),
    (r"\bsmth\b",   "something"),
    (r"\bsmthg\b",  "something"),
    (r"\binfo\b",   "information"),
    (r"\bdetails\b","details"),
]

def _normalise_informal(query: str) -> str:
    """
    Issue 1 Fix: Replace informal abbreviations with formal equivalents.
    Runs before everything else so downstream patterns work correctly.
    e.g. "what do u know about cobit?" → "what do you know about cobit?"
    """
    q = query
    for pattern, replacement in INFORMAL_MAP:
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    return q



# ── [FIX v7] Conversational context — follow-up query detection ───────────────

FOLLOWUP_SIGNALS = [
    "what if", "and if", "what about", "how about",
    "in that case", "what does that", "what else",
    "what if they", "does that mean", "in this case",
    "and what", "so what", "if so", "what if the",
    "also what", "what about the", "then what",
    "what happens if", "and if the", "but what",
    "and does", "and do", "and can", "and would",
    "and is", "and are", "what if it", "and how",
]

PRONOUN_STARTS = [
    "they ", "it ", "this ", "that ", "these ", "those ",
    "he ", "she ",
    "هم ", "هي ", "هو ", "ذلك ", "هذا ", "هذه ",
]


def _is_followup(query: str) -> bool:
    """
    Detect whether a query is a follow-up that depends on conversation context.
    Returns True when the query cannot be understood without prior messages.

    Signals:
      • Starts with a follow-up phrase ("what if", "and if", "how about" …)
      • Starts with "and", "but", "also", "then", "so"
      • Starts with a context-dependent pronoun ("it", "they", "this" …)
      • Very short AND lacks a regulatory anchor keyword
    """
    q = query.lower().strip().rstrip("?.")

    # Never treat identity/greeting/meta questions as follow-ups
    META_QUESTIONS = [
        "who are you", "what are you", "what is your name",
        "what can you do", "are you an ai", "are you a bot",
        "tell me about yourself", "how are you", "what do you do",
        "introduce yourself", "are we ready", "are you ready",
        "hello", "hi", "hey", "okay", "ok", "thanks", "thank you",
        "got it", "sounds good", "great", "nice", "cool", "awesome",
        "let's go", "lets go", "start", "begin", "go ahead",
        "من أنت", "ما اسمك", "عرّف نفسك",
    ]
    if any(q == m or q.startswith(m) for m in META_QUESTIONS):
        return False

    if any(q.startswith(s) for s in FOLLOWUP_SIGNALS):
        return True

    if q.startswith(("and ", "but ", "also ", "then ", "so ")):
        return True

    if any(q.startswith(p) for p in PRONOUN_STARTS):
        return True

    # Very short query with no regulatory anchor
    ANCHOR_TERMS = {
        "sama", "nca", "aml", "kyc", "pdpl", "ecc", "ccc",
        "pep", "ubo", "sar", "bank", "account", "basel",
        "capital", "liquidity", "iso", "fatf", "sme",
    }
    words = q.split()
    if len(words) <= 4 and not any(t in q for t in ANCHOR_TERMS):
        return True

    return False


def _contextualize_query(
    query: str,
    session_summary: str,
    last_messages: list[dict],
) -> str:
    """
    [FIX v7] Rewrite a follow-up question as a complete standalone question
    using recent conversation history so the RAG retrieval has enough context.

    Example:
      Prev Q: "What documents does the bank require to open an account?"
      Prev A: "The bank shall obtain the necessary documents…"
      Follow-up: "What if the SME is within Saudi and registered in Saudi Arabia?"
      Rewritten: "What are the bank account opening requirements for an SME
                  registered and operating within Saudi Arabia?"

    Returns the original query unchanged when:
      • Query is already standalone (_is_followup returns False)
      • No context is available
      • The LLM call fails for any reason
    """
    if not _is_followup(query):
        return query

    # Build context string
    context_parts: list[str] = []

    if session_summary and session_summary.strip():
        context_parts.append(f"Conversation topic: {session_summary.strip()}")

    if last_messages:
        for m in last_messages[-3:]:       # at most last 3 exchanges
            u = (m.get("user_message") or "").strip()
            a = (m.get("assistant_message") or "").strip()
            if u:
                context_parts.append(f"Previous question: {u}")
            if a:
                context_parts.append(f"Previous answer: {a[:400]}")

    if not context_parts:
        return query                       # no context — use original

    context = "\n".join(context_parts)

    prompt = (
        "You are helping a Saudi banking and cybersecurity regulatory chatbot "
        "understand follow-up questions.\n\n"
        f"Conversation context:\n{context}\n\n"
        f"Follow-up question: \"{query}\"\n\n"
        "Rewrite the follow-up as a COMPLETE STANDALONE question about Saudi "
        "banking or cybersecurity regulations that includes all necessary context "
        "from the conversation above.\n"
        "Return ONLY the rewritten question — no explanation, no quotation marks, "
        "nothing else.\n"
        "If the question is already standalone and clear, return it unchanged."
    )

    try:
        # Try Azure first, then OpenAI, then skip gracefully
        if LLM_BACKEND == "azure" and AZURE_OPENAI_KEY and AZURE_ENDPOINT:
            import openai as _oai
            client = _oai.AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                azure_endpoint=AZURE_ENDPOINT,
                api_version="2024-02-01",
            )
            model = AZURE_DEPLOYMENT
        elif OPENAI_API_KEY:
            import openai as _oai
            client = _oai.OpenAI(api_key=OPENAI_API_KEY)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            return query                   # no LLM available — skip

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=90,
            temperature=0.1,
        )
        rewritten = resp.choices[0].message.content.strip().strip("\"'")
        if rewritten and rewritten.lower() != query.lower():
            print(f"[ctx] Rewritten: '{query}' → '{rewritten}'")
            return rewritten

    except Exception as e:
        print(f"[ctx] Contextualize failed (non-fatal): {e}")

    return query                           # fallback to original



# Issue 5 Fix: yes/no question normalisation
# "can an indian create a bank account?" → "what are the rules for creating a bank account?"
YES_NO_PATTERNS = [
    # "can/could X verb Y?" → "what are the rules for verb Y?"
    (r"^can (?:a|an|the)?\s*\w+(?:\s+\w+)? (create|open|have|get|obtain|use|access|apply for|hold)\s+(.+?)\??$",
     r"what are the rules for \1 \2"),
    (r"^could (?:a|an|the)?\s*\w+(?:\s+\w+)? (create|open|have|get|obtain|use|access|apply for|hold)\s+(.+?)\??$",
     r"what are the rules for \1 \2"),
    # "is X allowed to verb Y?" → "what are the rules for verb Y?"
    (r"^is (?:a|an|the)?\s*\w+(?:\s+\w+)? (?:allowed|permitted|eligible) to\s+(.+?)\??$",
     r"what are the rules for \1"),
    # "are X allowed to verb Y?" → "what are the rules for verb Y?"
    (r"^are (?:\w+(?:\s+\w+)?)? (?:allowed|permitted|eligible) to\s+(.+?)\??$",
     r"what are the rules for \1"),
    # "is it possible for X to Y?" → "what are the rules for Y?"
    (r"^is it (?:possible|allowed|permitted) for (?:a|an|the)?\s*\w+(?:\s+\w+)? to\s+(.+?)\??$",
     r"what are the rules for \1"),
]

def _normalise_yes_no(query: str) -> str:
    """
    Issue 5 Fix: Convert yes/no questions to factual regulatory questions.
    e.g. "can an indian create a bank account?" → "what are the rules for create a bank account?"
    """
    q = query.strip()
    q_lower = q.lower().rstrip("?.")
    for pattern, replacement in YES_NO_PATTERNS:
        match = re.match(pattern, q_lower, re.IGNORECASE)
        if match:
            normalised = re.sub(pattern, replacement, q_lower, flags=re.IGNORECASE).strip()
            print(f"[yes_no] '{q}' → '{normalised}'")
            return normalised
    return q


# Issue 7 Fix: restriction query domain anchor
# If a query is about restrictions/prohibitions but has no SAMA anchor,
# inject SAMA EN 1644 terms to prevent it from floating without direction.
RESTRICTION_SIGNALS = [
    "cannot", "can not", "cannot open", "not allowed", "not permitted",
    "prohibited", "who cannot", "who can't", "not eligible", "ineligible",
    "restrictions", "restrict", "forbidden", "banned", "cannot create",
    "cannot have", "not create", "not open",
]

DOMAIN_ANCHOR = (
    "bank account opening restrictions prohibited SAMA EN 1644 "
    "saudi arabian monetary authority shall not eligible"
)

def _inject_domain_anchor(query: str, expanded: str) -> str:
    """
    Issue 7 Fix: If query contains restriction language but no SAMA anchor,
    append SAMA EN 1644 domain anchor to prevent the query from floating.
    """
    q_lower = query.lower()
    has_restriction = any(s in q_lower for s in RESTRICTION_SIGNALS)
    has_domain_anchor = "sama" in q_lower or "1644" in q_lower or "saudi" in q_lower
    if has_restriction and not has_domain_anchor:
        print(f"[anchor] Injecting domain anchor for restriction query: '{query}'")
        return expanded + " " + DOMAIN_ANCHOR
    return expanded


# Strips meta-phrasing like "what knowledge do you have of X" → "what is X"
META_PATTERNS = [
    (r"what (knowledge|info|information|details) do you have (on|about|of)\s+", "what is "),
    (r"what do you know (about|of|on)\s+",                                       "what is "),
    (r"can you (explain|describe|summarise|summarize|tell me about)\s+",          "what is "),
    (r"do you have (info|information|knowledge|details|data) (on|about|of)\s+",   "what is "),
    (r"tell me (about|what you know about|more about)\s+",                        "what is "),
    (r"give me (info|information|details|an overview) (on|about|of)\s+",          "what is "),
    (r"(show|send|list|provide) (me )?(the )?(details|info|information) (on|about|of)\s+", "what is "),
    (r"(show|send|list|provide) (me )?(the )?\s+",                                "what is "),
    (r"i (want|need|would like) (to know|information) (about|on|of)\s+",          "what is "),
    (r"(explain|describe) (to me )?(the |a |an )?\s+",                            "what is "),
]

def _normalise_query(query: str) -> str:
    """
    Strip meta-phrasing and extract the real subject of the question.
    e.g. "what knowledge do you have of cobit?" → "what is cobit?"
         "send SAMA guidelines" → "what is SAMA guidelines"
         "can you explain PEP?" → "what is PEP?"
    """
    q = query.strip()
    q_lower = q.lower().rstrip("?.")
    for pattern, replacement in META_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            subject = q[match.end():].strip().rstrip("?.")
            if subject:
                normalised = replacement + subject
                print(f"[normalise] '{q}' → '{normalised}'")
                return normalised
    return q


def _extract_subject(query: str) -> str:
    """
    Extract the core subject term(s) from a query for use in keyword (BM25) search.
    Removes question words and meta-starters so BM25 gets the cleanest possible signal.
    e.g. "what is cobit?" → "cobit"
         "what are the AML obligations?" → "AML obligations"
    """
    q = query.strip().lower().rstrip("?.")
    # Remove leading question starters
    starters = r"^(what is|what are|what does|how does|how do|explain|describe|define|tell me about|give me|show me|send me|list|provide)\s+"
    subject = re.sub(starters, "", q).strip()
    # Remove filler words
    fillers = r"\b(the|a|an|some|all|any|their|its|our|your|my)\b"
    subject = re.sub(fillers, " ", subject).strip()
    subject = re.sub(r"\s{2,}", " ", subject).strip()
    return subject if subject else query.strip()


QUERY_EXPANSIONS = {
    # ── English acronyms ──────────────────────────────────────────────────────
    "kyc":    "know your customer customer due diligence verification identification",
    "aml":    "anti-money laundering suspicious transactions monitoring",
    "cft":    "counter financing terrorism",
    "ctf":    "counter terrorism financing",
    "icaap":  "internal capital adequacy assessment process",
    "lcr":    "liquidity coverage ratio high quality liquid assets cash outflows inflows",
    "nsfr":   "net stable funding ratio available stable funding required stable funding",
    "sama":   "saudi arabian monetary authority central bank",
    "cma":    "capital market authority",
    "pdpl":   "personal data protection law",
    "car":    "capital adequacy ratio",
    "rwa":    "risk weighted assets",
    "cdd":    "customer due diligence know your customer",
    "edd":    "enhanced due diligence high risk customers PEP politically exposed",
    "hqla":   "high quality liquid assets liquidity coverage ratio",
    "ltv":    "loan to value ratio risk weight residential real estate",
    "retail": "retail customers individual natural persons resident bank account",
    "ubo":    "ultimate beneficial owner UBO corporate onboarding verification SAMA EN 1704",
    "sar":    "suspicious activity report SAR financial intelligence unit SAFIU SAMA AML",
    "pep":    "politically exposed person enhanced due diligence EDD SAMA EN 1704 high risk customer AML",

    # ── PEP / KYC / Onboarding (English) ─────────────────────────────────────
    "politically exposed":              "PEP enhanced due diligence senior management approval source of wealth SAMA EN 1704",
    "enhanced due diligence":           "EDD PEP high risk customer enhanced measures SAMA EN 1704 AML CTF",
    "fails enhanced due diligence":     "PEP failed EDD terminate relationship suspicious activity SAMA EN 1704 AML",
    "pep customer":                     "politically exposed person enhanced due diligence SAMA EN 1704 AML high risk",
    "pep fails":                        "PEP failed enhanced due diligence terminate relationship SAMA EN 1704",
    "kyc requirements":                 "know your customer KYC individuals corporates identity verification SAMA EN 1644 SAMA EN 1704",
    "kyc individuals":                  "individual natural persons KYC requirements SAMA EN 1644 identity documents",
    "kyc corporates":                   "corporate juristic persons KYC UBO beneficial owner SAMA EN 1704 1644",
    "beneficial owner":                 "beneficial owner UBO verification identification SAMA EN 1704 AML corporate",
    "verify ubo":                       "ultimate beneficial owner UBO verification corporate onboarding SAMA EN 1704",
    "ultimate beneficial owner":        "UBO ultimate beneficial owner corporate verification SAMA EN 1704 AML ownership",
    "sanctions screening":              "sanctions list screening UN OFAC PCCML SAMA EN 1704 AML CTF prohibited",
    "sanctions list":                   "sanctions lists UN Security Council OFAC FATF screening SAMA EN 1704 1428",
    "suspicious activity":              "suspicious activity report SAR SAFIU reporting AML CTF SAMA EN 1704",
    "account opening workflow":         "bank account opening step by step workflow SAMA EN 1644 procedures requirements",
    "step by step":                     "bank account opening workflow procedures steps SAMA EN 1644",
    "manual review":                    "manual review compliance approval bank account high risk SAMA EN 1644",
    "straight through processing":      "STP automatic processing bank account opening low risk SAMA EN 1644",
    "remote onboarding":                "remote account opening digital online SAMA EN 1644 verification identity",
    "digital onboarding":               "digital remote account opening SAMA EN 1644 SAMA EN 2888 mobile app",
    "digital account opening limits":   "digital account opening limits SAMA EN 1644 mobile online banking",
    "biometric verification":           "biometric identity verification NafathID Absher national single sign on SAMA",
    "absher":                           "Absher national single sign on portal identity verification SAMA digital onboarding",
    "yaqeen":                           "Yaqeen identity verification national portal SAMA onboarding KYC",
    "digital signature":                "electronic signature digital Electronic Transactions Law Saudi Arabia",
    "expired id":                       "expired identification documents bank account exception renewal SAMA EN 1644",
    "id under renewal":                 "national ID under renewal exception bank account SAMA EN 1644 180 days",
    "minor customer":                   "minor underage customer bank account guardian SAMA EN 1644 15 18 hijri years",
    "joint account":                    "joint bank account opening digital online SAMA EN 1644",
    "corporate onboarding":             "corporate juristic person onboarding documents commercial register SAMA EN 1644",
    "commercial registration":          "commercial register CR validation verification SAMA EN 1644 Ministry of Commerce",
    "articles of association":          "articles of association memorandum validation corporate onboarding SAMA EN 1644",
    "signatory approval":               "authorized signatory approval corporate bank account SAMA EN 1644",
    "incomplete application":           "incomplete bank account application missing documents SAMA EN 1644 requirements",
    "name mismatch":                    "customer name mismatch documents identity verification SAMA EN 1644",
    "document submission":              "document submission formats digital physical bank account SAMA EN 1644",
    "non-resident onboarding":          "non-resident bank account opening conditions SAMA EN 1644 Ministry of Interior",
    "large corporate":                  "large corporate onboarding requirements SAMA EN 1644 juristic persons documents",
    "sme onboarding":                   "SME small medium enterprise onboarding requirements SAMA EN 1644 commercial register",
    "biometric fallback":               "biometric verification failure fallback OTP SMS alternative SAMA EN 2888",
    "mobile app onboarding":            "mobile app onboarding digital account SAMA EN 2888 1644 OTP verification",
    "ongoing due diligence":            "ongoing due diligence CDD periodic review customer information SAMA EN 1704 1644",
    "rejection criteria":               "account opening rejection criteria requirements SAMA EN 1644 compliance",
    "what lists must be checked":       "sanctions lists screening UN OFAC FATF PCCML SAMA AML CTF prohibited",
    "lists checked":                    "sanctions screening lists UN Security Council OFAC FATF SAMA EN 1704",

    # ── English technical expansions ──────────────────────────────────────────
    "cap on cash inflows":    "75% cap total cash inflows outflows LCR Basel III liquidity",
    "cash inflows cap":       "75 percent cap inflows outflows LCR net cash",
    "inflows cap":            "aggregate cap 75 percent total cash outflows LCR",
    "leverage ratio":         "leverage ratio tier 1 capital total exposure measure 3 percent Basel III framework",
    "calculate the leverage": "leverage ratio tier 1 capital exposure measure calculation consolidated standalone",
    "leverage framework":     "leverage ratio framework scope regulatory consolidation domestic banks",
    "loss event threshold":   "loss event threshold 20000 EUR operational risk internal data collection AMA",
    "loss event":             "loss event threshold EUR operational risk data collection",
    "operational risk capital": "loss event data collection threshold operational risk capital AMA standardised",
    "loan-to-value":          "loan to value ratio LTV risk weight 100 percent residential real estate",
    "ltv ratio":              "LTV loan to value risk weight residential mortgage property",
    "risk weight":            "risk weight loan to value LTV residential real estate mortgage",
    "net stable funding":     "net stable funding ratio NSFR available stable funding required stable resilience promote",
    "stable funding ratio":   "NSFR net stable funding ratio 100 percent available required stable funding resilience",
    "nsfr resilience":        "NSFR promote resilience longer term funding structure available stable funding",
    "prepaid fees":           "fees charges prepaid payment service issuance reloading card",
    "fees prepaid":           "fees charges prepaid payment service issuance reloading card",
    "prepaid payment service fees": "issuance fees charges prepaid card payment service types",
    "types of fees prepaid":  "types fees prepaid payment service issuance reload transaction",
    "savings products":       "savings accounts deposits savings products general rules banks financial institution",
    "general rules savings":  "savings accounts bank deposits general rules regulations",
    "clawback":               "clawback malus deferred remuneration adjustment vesting Saudi Arabia relevant laws criteria",
    "clawback arrangements":  "clawback arrangements deferred remuneration malus Saudi Arabia relevant laws policy criteria adjusting",
    "binding common rules":   "binding common rules BCR controller competent authority report compliance personal data protection",
    "controller":             "controller competent authority binding common rules BCR personal data transfer",
    "binding common":         "binding common rules BCR controller personal data protection transfer",
    "personal data scientific": "personal data scientific research purposes consent exception",
    "byod":                   "bring your own device mobile security OT ICS cybersecurity risk assessment",
    "mobile devices byod":    "BYOD mobile device security policy OT ICS management approval",
    "fraud investigation":    "member organisations notify SAMA general department cyber risk control immediately significant fraud",
    "bank fraud":             "notify SAMA cyber risk control fraudulent typology significant fraud internal external",
    "fraud initiated":        "notify SAMA general department cyber risk control new fraudulent typology significant fraud",
    "shariah audit":          "internal shariah audit function purpose islamic banking independent assessment",

    # ── Capital adequacy specifics ────────────────────────────────────────────
    "minimum capital adequacy":     "minimum capital adequacy ratio 8 percent 10.5 percent Basel III Tier 1 CET1 banks SAMA",
    "capital adequacy requirement": "capital adequacy ratio minimum 8% 10.5% Tier 1 CET1 conservation buffer Basel III SAMA banks",
    "capital adequacy ratio banks": "CAR minimum 8 percent 10.5 percent CET1 Tier1 total capital ratio Basel III SAMA",
    "minimum capital requirement":  "minimum capital requirement new bank license SAR paid-up capital establishment",
    "capital requirements bank":    "minimum capital adequacy ratio Basel III Tier 1 CET1 8 percent 10.5 percent SAMA banks",
    "capital adequacy minimum":     "capital adequacy minimum ratio 8% 10.5% CET1 Tier 1 total capital Basel III SAMA",
    "capital adequacy percentage":  "capital adequacy ratio percentage minimum 8 10.5 percent CET1 Tier 1 Basel III",
    "minimum capital ratio":        "minimum capital adequacy ratio CET1 Tier 1 total capital 8 percent 10.5 percent SAMA Basel III",

    # ── Admin service charges ─────────────────────────────────────────────────
    "admin service charge":         "administrative service charges maximum cap fees banking services SAMA regulation limit",
    "administrative service":       "administrative service charges maximum fees cap banking consumer protection SAMA",
    "service charges maximum":      "maximum administrative service charges fees cap limit banking SAMA regulation",
    "charges maximum":              "maximum fees charges administrative services banking consumer protection SAMA",
    "admin charges":                "administrative charges maximum cap banking services fees SAMA regulation limit",
    "service charge limit":         "administrative service charge maximum limit cap banking SAMA consumer protection",

    # ── NCA and SAMA relationship ─────────────────────────────────────────────
    "nca sama relationship":        "NCA SAMA cybersecurity framework applicability financial sector relationship authority",
    "nca and sama":                 "NCA national cybersecurity authority SAMA member organizations financial sector applicability overlap",
    "relationship between nca":     "NCA SAMA cybersecurity framework financial institutions applicability authority",
    "sama and nca":                 "SAMA NCA cybersecurity framework member organizations applicability financial sector",
    "nca vs sama":                  "NCA SAMA cybersecurity frameworks applicability financial sector relationship difference",
    "nca sama difference":          "NCA SAMA cybersecurity framework applicability financial institutions banks difference",

    # ── Savings products ──────────────────────────────────────────────────────
    "rules for savings":            "savings accounts deposits general rules regulations SAMA banks financial products",
    "savings account rules":        "savings account regulations general rules SAMA banks deposits products",
    "savings account regulation":   "savings accounts deposits general rules regulations SAMA banks",
    "savings deposit rules":        "savings deposits accounts general rules regulations SAMA banks financial",

    # ── Annual disclosure ─────────────────────────────────────────────────────
    "annual disclosure requirement": "annual disclosure requirements banks pillar 3 total assets 4.46 billion SAR SAMA",
    "disclosure requirement banks":  "annual disclosure requirements pillar 3 banks assets SAR SAMA reporting",
    "pillar 3 disclosure":           "pillar 3 disclosure requirements annual banks total assets SAR SAMA",

    # ── Loan-to-deposit ───────────────────────────────────────────────────────
    "loan to deposit ratio":         "loan to deposit ratio LDR banks disclosure reporting SAMA requirements",
    "loan deposit ratio":            "loan to deposit ratio LDR banks SAMA reporting requirements",

    # ── PDPL penalties English ────────────────────────────────────────────────
    "pdpl violation penalty":        "personal data protection law PDPL violations penalties fines Saudi Arabia SDAIA",
    "pdpl fine":                     "PDPL personal data protection law fines penalties violations Saudi Arabia",
    "data protection penalty":       "personal data protection law PDPL penalties violations fines Saudi Arabia SDAIA",

    # ── Bank account multi-word keys ──────────────────────────────────────────
    "bank account opening":          "bank account opening rules requirements procedures eligibility SAMA EN 1644",
    "open bank account":             "bank account opening requirements procedures eligibility SAMA EN 1644",
    "bank account rules":            "bank account opening rules regulations SAMA EN 1644 requirements",
    "open an account":               "bank account opening rules eligibility requirements procedures SAMA EN 1644",
    "create bank account":           "bank account opening eligibility requirements restrictions SAMA EN 1644",
    "create a bank account":         "bank account opening eligibility requirements restrictions prohibited SAMA EN 1644",

    # ── Restriction / negation queries ────────────────────────────────────────
    "cannot open bank account":      "bank account restrictions prohibited persons not eligible cannot open SAMA EN 1644 shall not",
    "not allowed bank account":      "bank account restrictions prohibited not permitted cannot open SAMA regulations shall not",
    "who cannot bank":               "bank account restrictions prohibited persons eligibility requirements SAMA EN 1644 shall not",
    "bank account restrictions":     "bank account restrictions prohibited entities persons SAMA EN 1644 shall not open",
    "not eligible bank":             "bank account eligibility restrictions prohibited persons SAMA regulations shall not",
    "restrictions bank account":     "restrictions prohibited cannot open bank account SAMA EN 1644 eligibility shall not",
    "prohibited bank account":       "prohibited persons entities cannot open bank account SAMA EN 1644 restrictions",
    "who cannot create":             "restrictions prohibited not eligible cannot open bank account SAMA EN 1644",
    "not permitted bank":            "bank account not permitted prohibited restrictions SAMA EN 1644 shall not open",
    "ineligible bank account":       "ineligible not eligible prohibited bank account restrictions SAMA EN 1644",
    "bank account eligibility":      "bank account eligibility requirements restrictions prohibited persons SAMA EN 1644",
    "who is not allowed":            "restrictions prohibited not permitted not eligible SAMA regulations shall not",
    "not allowed to open":           "restrictions prohibited cannot open bank account SAMA EN 1644 shall not",

    # ── Filler-word variants (Issue 4 Fix) ───────────────────────────────────
    # "who all cannot" has "all" between "who" and "cannot" breaking exact match
    "who all cannot create":     "restrictions prohibited not eligible cannot open bank account SAMA EN 1644",
    "who all cannot open":       "restrictions prohibited not eligible cannot open bank account SAMA EN 1644",
    "who all can not":           "restrictions prohibited not eligible cannot open bank account SAMA EN 1644",
    "who all are not allowed":   "restrictions prohibited not permitted bank account SAMA EN 1644 shall not",
    "who all is not allowed":    "restrictions prohibited not permitted bank account SAMA EN 1644 shall not",
    "who else cannot":           "restrictions prohibited not eligible cannot open bank account SAMA EN 1644",
    "who else can not":          "restrictions prohibited not eligible cannot open bank account SAMA EN 1644",

    # ── Nationality → category expansions (Issue 6 Fix) ──────────────────────
    # Documents use "non-GCC non-Saudi natural person" — users use specific nationalities
    "indian":     "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "american":   "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "british":    "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "european":   "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "pakistani":  "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "egyptian":   "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "filipino":   "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "bangladeshi":"non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "chinese":    "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "expat":      "non-Saudi non-GCC natural person expatriate resident bank account SAMA EN 1644",
    "expatriate": "non-Saudi non-GCC natural person expatriate resident bank account SAMA EN 1644",
    "foreigner":  "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644 restrictions",
    "foreign national": "non-Saudi non-GCC natural person non-resident bank account SAMA EN 1644",
    "non-saudi":  "non-Saudi natural person bank account eligibility restrictions SAMA EN 1644",
    "non saudi":  "non-Saudi natural person bank account eligibility restrictions SAMA EN 1644",
    "non-gcc":    "non-GCC natural person bank account eligibility restrictions SAMA EN 1644",
    "non gcc":    "non-GCC natural person bank account eligibility restrictions SAMA EN 1644",
    "gcc national": "GCC natural person bank account eligibility SAMA EN 1644 resident",
    "من لا يمكنه فتح حساب":          "bank account restrictions prohibited persons cannot open SAMA EN 1644 shall not",
    "المحظورون من فتح حساب":          "bank account restrictions prohibited not permitted SAMA regulations shall not",
    "من لا يحق له فتح حساب":          "bank account restrictions prohibited persons not eligible SAMA EN 1644",
    "الممنوعون من فتح حساب بنكي":    "bank account restrictions prohibited persons cannot open SAMA EN 1644",
    "لا يجوز فتح حساب":              "bank account shall not open prohibited restrictions SAMA EN 1644",
    "فتح الحساب البنكي":              "bank account opening rules eligibility requirements restrictions SAMA EN 1644",
    "شروط فتح الحساب":               "bank account opening conditions requirements restrictions eligibility SAMA EN 1644",
    "من يُمنع من فتح حساب":          "bank account restrictions prohibited persons not permitted SAMA EN 1644 shall not",

    # ── ISO Standards (English) ───────────────────────────────────────────────
    "iso 27001":   "ISO 27001 information security management system ISMS certification audit controls",
    "iso 27701":   "ISO 27701 privacy information management system PIMS personal data protection extension",
    "iso 27400":   "ISO IEC 27400 IoT internet of things security privacy guidelines",
    "iso 27403":   "ISO IEC 27403 IoT internet of things security controls",
    "iso 20000":   "ISO 20000 IT service management ITSM service delivery processes",
    "iso 22301":   "ISO 22301 business continuity management system BCMS requirements resilience",
    "iso 23200":   "ISO 23200 blockchain distributed ledger technology DLT financial services",
    "iso 42001":   "ISO 42001 artificial intelligence management system AI governance framework",
    "isms":        "information security management system ISMS ISO 27001 establish implement maintain",
    "information security management": "ISMS ISO 27001 information security management system certification",

    # ── NCA / ECC / Cybersecurity (English) ──────────────────────────────────
    "nca framework":          "NCA national cybersecurity authority essential cybersecurity controls ECC governance",
    "ecc controls":           "essential cybersecurity controls ECC NCA national cybersecurity authority minimum requirements",
    "ccc framework nca":      "cloud cybersecurity controls CCC NCA national cybersecurity authority cloud service providers",
    "critical infrastructure nca": "critical national infrastructure CNI cybersecurity NCA definition sectors",
    "nca governance":         "NCA cybersecurity governance policies frameworks standards controls guidelines",
    "nca incident":           "NCA cybersecurity incident management response requirements entities",
    "nca risk controls":      "NCA essential cybersecurity controls ECC government entities risk management compliance",
    "nca monitoring":         "NCA cybersecurity monitoring requirements event logs telework systems",

    # ── GRC Services (English) ────────────────────────────────────────────────
    "grc gap assessment":     "GRC gap assessment maturity review governance risk compliance cybersecurity",
    "grc maturity":           "GRC maturity assessment gap analysis governance risk compliance framework",
    "soc compliance":         "SOC security operations center compliance alignment GRC services",
    "policy framework development": "policy framework development GRC governance risk compliance",
    "risk assessment control": "risk assessment control implementation GRC governance compliance",
    "ongoing managed compliance": "ongoing managed compliance support GRC continuous monitoring services",
    "internal audit certification": "internal audit certification readiness GRC ISO 27001 compliance preparation",

    # ── Aramco CCC (English) ──────────────────────────────────────────────────
    "ccc program":            "cybersecurity compliance certification CCC third party vendors Saudi Aramco SACS-002",
    "ccc standard":           "CCC standard remote verification authorized audit firm self-compliance assessment SACS-002",
    "ccc plus":               "CCC+ on-site assessment higher risk vendors network connectivity critical data",
    "sacs-002":               "SACS-002 third party cybersecurity standard Saudi Aramco minimum requirements",
    "self-compliance assessment": "self-compliance assessment SACS-002 CCC third party cybersecurity Aramco",
    "ccc assessment":         "cybersecurity compliance certification CCC assessment levels standard plus Aramco",

    # ── PDPL (English) ────────────────────────────────────────────────────────
    "pdpl penalties":         "personal data protection law PDPL violations penalties fines Saudi Arabia",
    "pdpl scope":             "personal data protection law PDPL scope application organizations Saudi Arabia",
    "sdaia role":             "SDAIA Saudi data AI authority personal data protection law enforcement oversight",
    "ndmo role":              "NDMO national data management office personal data protection standards compliance",
    "data subject rights":    "data subject rights PDPL personal data protection law Saudi Arabia access erasure",

    # ── SAMA Cybersecurity Framework (English) ────────────────────────────────
    "sama cybersecurity framework": "SAMA cybersecurity framework member organizations banks insurance financing",
    "sama cyber framework sectors": "SAMA cybersecurity framework applicable banks insurance financing credit bureaus",
    "sama third party security":    "SAMA cybersecurity framework third party security compliance financial institutions",
    "sama incident response":        "SAMA cybersecurity framework incident response monitoring policy",
    "sama cyber risk management":    "SAMA cybersecurity framework risk management controls member organizations",
    "sama security operations":      "SAMA cybersecurity framework security operations domains banking sector",

    # ── Arabic → English bridges: SAMA / Banking ─────────────────────────────
    "البنك المركزي السعودي":   "SAMA saudi arabian monetary authority central bank البنك المركزي",
    "مؤسسة النقد":             "SAMA saudi arabian monetary authority مؤسسة النقد العربي السعودي",
    "رأس المال":               "capital adequacy requirements banks Basel III minimum capital ratio",
    "متطلبات رأس المال":       "capital adequacy ratio minimum requirements Basel III banks SAMA",
    "كفاية رأس المال":         "capital adequacy ratio CAR Basel III minimum requirements banks",
    "العقوبات":                "penalties violations banking control law SAMA regulations sanctions",
    "المبالاة":                "negligence penalties violations banking regulations SAMA sanctions",
    "المبالاة أو التقصير":     "negligence penalties fines violations banking control law SAMA",
    "التقصير":                 "negligence violations penalties banking regulations SAMA",
    "IFRS 9":                  "IFRS 9 التعرضات المتعثرة non-performing exposures default credit impaired stage",
    "التعرضات المتعثرة":       "IFRS 9 non-performing exposures default credit impaired stage classification",
    "نسبة القرض":              "loan to deposit ratio LDR reporting requirements banks disclosure",
    "القرض إلى الودائع":       "loan to deposit ratio LDR reporting requirements annual disclosure banks",
    "الكشف السنوي":            "annual disclosure requirements banks pillar 3 assets billion SAR",
    "4.46 مليار":              "annual disclosure banks 4.46 billion SAR pillar 3 disclosure requirements",
    "نسبة تغطية السيولة":      "liquidity coverage ratio LCR HQLA high quality liquid assets Basel III",
    "نسبة التمويل المستقر":    "net stable funding ratio NSFR available stable funding required Basel III",

    # ── Arabic → English bridges: Savings ────────────────────────────────────
    "منتجات الادخار":          "savings products accounts deposits general rules SAMA banks regulations",
    "حسابات التوفير":          "savings accounts deposits general rules regulations SAMA banks products",
    "قواعد الادخار":           "savings accounts deposits general rules regulations SAMA banks",
    "منتجات التوفير":          "savings products accounts deposits general rules SAMA banks",

    # ── Arabic → English bridges: Capital adequacy ────────────────────────────
    "الحد الأدنى لنسبة كفاية رأس المال": "minimum capital adequacy ratio 8 percent 10.5 percent CET1 Tier1 Basel III SAMA",
    "نسبة كفاية رأس المال":    "capital adequacy ratio CAR minimum percentage CET1 Tier 1 Basel III SAMA banks",
    "متطلبات الحد الأدنى لرأس المال": "minimum capital adequacy requirements 8% 10.5% CET1 Tier 1 Basel III SAMA banks",

    # ── Arabic → English bridges: Admin charges ───────────────────────────────
    "رسوم الخدمات الإدارية":   "administrative service charges maximum cap fees banking SAMA regulation limit",
    "الحد الأقصى للرسوم":      "maximum fees charges administrative services banking consumer protection SAMA cap",
    "رسوم الخدمات المصرفية":   "banking service fees charges maximum cap administrative SAMA regulation",

    # ── Arabic → English bridges: Annual disclosure ───────────────────────────
    "متطلبات الإفصاح السنوي":  "annual disclosure requirements pillar 3 banks assets billion SAR SAMA",
    "الإفصاح السنوي للبنوك":   "annual disclosure requirements banks pillar 3 total assets 4.46 billion SAR",
    "إفصاح الركيزة الثالثة":   "pillar 3 disclosure requirements annual banks total assets SAR SAMA",

    # ── Arabic → English bridges: Loan-to-deposit ────────────────────────────
    "نسبة القرض إلى الودائع":  "loan to deposit ratio LDR banks disclosure reporting SAMA requirements",
    "نسبة الإقراض إلى الودائع": "loan to deposit ratio LDR banks SAMA reporting requirements",

    # ── Arabic → English bridges: PDPL penalties ─────────────────────────────
    "عقوبات نظام حماية البيانات": "personal data protection law PDPL penalties violations fines Saudi Arabia SDAIA",
    "مخالفات حماية البيانات":  "PDPL violations penalties fines personal data protection law Saudi Arabia",
    "غرامات نظام حماية البيانات": "PDPL fines penalties violations personal data protection law Saudi Arabia",

    # ── Arabic → English bridges: NCA-SAMA relationship ──────────────────────
    "العلاقة بين الهيئة الوطنية للأمن السيبراني وساما": "NCA SAMA cybersecurity framework relationship financial sector applicability",
    "الهيئة الوطنية للأمن السيبراني وساما": "NCA SAMA cybersecurity framework applicability financial institutions banks",
    "العلاقة بين الهيئة الوطنية وساما": "NCA SAMA cybersecurity framework relationship financial sector",

    # ── Arabic → English bridges: NCA / ECC / Cybersecurity ──────────────────
    "الهيئة الوطنية للأمن السيبراني": "NCA national cybersecurity authority essential cybersecurity controls ECC",
    "الضوابط الأساسية للأمن السيبراني": "essential cybersecurity controls ECC NCA minimum requirements national entities",
    "ضوابط ECC":               "essential cybersecurity controls ECC NCA national cybersecurity authority",
    "ضوابط الامتثال للأمن السيبراني": "cloud cybersecurity controls CCC NCA compliance requirements",
    "البنية التحتية الحيوية":  "critical national infrastructure CNI cybersecurity NCA definition sectors",
    "إدارة الحوادث":           "cybersecurity incident management response NCA ECC requirements",
    "الرقابة والمراقبة":       "NCA cybersecurity monitoring surveillance requirements event logs controls",
    "متطلبات الحوكمة":         "cybersecurity governance NCA ECC policies frameworks standards controls",
    "مزودي الخدمات السحابية":  "cloud service providers CSP NCA cloud cybersecurity controls CCC compliance",
    "الأمن السيبراني لساما":   "SAMA cybersecurity framework member organizations banks financial institutions",
    "إطار الأمن السيبراني لساما": "SAMA cybersecurity framework applicability sectors banks insurance financing",
    "متطلبات الاستجابة للحوادث": "SAMA cybersecurity framework incident response policy monitoring capabilities",
    "متطلبات امتثال الأطراف الثالثة": "SAMA cybersecurity framework third party security compliance financial institutions",
    "عمليات الأمن السيبراني":  "SAMA cybersecurity framework security operations domains banking sector",
    "متطلبات إدارة المخاطر":   "SAMA cybersecurity framework risk management controls member organizations",

    # ── Arabic → English bridges: ISO Standards ──────────────────────────────
    "معيار ISO 27001":         "ISO 27001 information security management system ISMS certification controls",
    "ISO 27001":               "ISO 27001 information security management system ISMS certification",
    "معيار ISO 27701":         "ISO 27701 privacy information management PIMS personal data protection",
    "ISO 27701":               "ISO 27701 privacy information management system personal data",
    "معيار ISO 27400":         "ISO IEC 27400 IoT internet of things security privacy",
    "ISO 27400":               "ISO IEC 27400 IoT internet of things security guidelines",
    "معيار ISO 20000":         "ISO 20000 IT service management ITSM processes service delivery",
    "ISO 20000":               "ISO 20000 service management system processes",
    "معيار ISO 22301":         "ISO 22301 business continuity management system BCMS resilience requirements",
    "ISO 22301":               "ISO 22301 business continuity management resilience",
    "معيار ISO 23200":         "ISO 23200 blockchain distributed ledger technology DLT",
    "ISO 23200":               "ISO 23200 blockchain distributed ledger technology financial services",
    "معيار ISO 42001":         "ISO 42001 artificial intelligence management system AI governance",
    "ISO 42001":               "ISO 42001 artificial intelligence AI management system governance",
    "نظام إدارة أمن المعلومات": "ISMS information security management system ISO 27001 establish implement",
    "ISMS":                    "information security management system ISO 27001 ISMS certification",

    # ── Arabic → English bridges: PDPL / Data Protection ─────────────────────
    "نظام حماية البيانات الشخصية": "personal data protection law PDPL Saudi Arabia SDAIA controller processor",
    "حماية البيانات الشخصية":  "personal data protection law PDPL Saudi Arabia controller processor rights",
    "نطاق تطبيق نظام":         "personal data protection law PDPL scope application organizations Saudi Arabia",
    "العقوبات المقررة":         "personal data protection law PDPL penalties violations fines Saudi Arabia",
    "مخالفة نظام حماية البيانات": "PDPL violations penalties fines personal data protection law Saudi Arabia",
    "هيئة البيانات والذكاء الاصطناعي": "SDAIA Saudi data AI authority personal data protection competent authority",
    "SDAIA":                   "Saudi data AI authority SDAIA personal data protection law enforcement",
    "المكتب الوطني لإدارة البيانات": "NDMO national data management office personal data protection standards compliance",
    "NDMO":                    "national data management office NDMO personal data protection standards",
    "حقوق أصحاب البيانات":     "data subject rights personal data protection PDPL access erasure withdrawal",
    "المتحكمين في البيانات":   "data controllers obligations personal data protection PDPL compliance requirements",
    "الالتزامات المفروضة على المتحكمين": "controller obligations personal data protection PDPL data minimization audit",
    "أنواع البيانات الشخصية":  "personal data types PDPL sensitive data definition categories protection",

    # ── Arabic → English bridges: Aramco CCC / SACS-002 ──────────────────────
    "شهادة الامتثال للأمن السيبراني": "cybersecurity compliance certification CCC Saudi Aramco third party SACS-002",
    "موردي أرامكو":            "Saudi Aramco vendors third party cybersecurity SACS-002 CCC compliance",
    "برنامج أرامكو للأمن السيبراني": "Saudi Aramco cybersecurity compliance certification CCC SACS-002 third party",
    "مستوى CCC القياسي":       "CCC standard remote verification self-compliance assessment authorized audit firm",
    "مستوى CCC+":              "CCC+ on-site assessment higher risk vendors network connectivity Aramco",
    "التقييم الذاتي للامتثال": "self-compliance assessment CCC SACS-002 third party cybersecurity Aramco",
    "الأطراف الخارجية":        "third party vendors CCC+ higher risk on-site assessment Aramco cybersecurity",
    "SACS-002":                "SACS-002 third party cybersecurity standard Saudi Aramco minimum requirements",

    # ── Arabic → English bridges: GRC Services ───────────────────────────────
    "تقييم الفجوات":           "GRC gap assessment maturity review governance risk compliance cybersecurity",
    "مراجعة النضج":            "GRC maturity review assessment governance risk compliance framework",
    "حوكمة المخاطر والامتثال": "GRC governance risk compliance services gap assessment maturity",
    "تطوير السياسات والأطر":   "GRC policy framework development governance risk compliance",
    "تقييم المخاطر وتطبيق الضوابط": "GRC risk assessment control implementation governance compliance",
    "المواءمة مع SOC":         "SOC security operations center compliance alignment GRC services",
    "دعم الامتثال المُدار":    "ongoing managed compliance support GRC continuous monitoring services",
    "التدقيق الداخلي والاستعداد": "internal audit certification readiness GRC ISO compliance preparation",
    "شهادات GRC":              "GRC certification readiness internal audit ISO 27001 compliance",

    # ── Arabic → English bridges: PEP / KYC / Onboarding ────────────────────
    "الشخص المعرض سياسياً":    "PEP politically exposed person enhanced due diligence SAMA EN 1704 AML high risk",
    "العناية المشددة":          "enhanced due diligence EDD PEP high risk customer SAMA EN 1704 AML CTF",
    "العناية الواجبة المعززة":  "enhanced due diligence EDD PEP politically exposed person SAMA EN 1704",
    "فشل العناية الواجبة":      "failed enhanced due diligence EDD PEP terminate relationship SAMA EN 1704",
    "متطلبات العناية الواجبة":  "customer due diligence CDD KYC requirements SAMA EN 1704 1644 AML",
    "المالك المستفيد الفعلي":   "ultimate beneficial owner UBO verification corporate SAMA EN 1704 AML",
    "الشخص المعرض للمخاطر":    "PEP politically exposed person enhanced due diligence SAMA EN 1704",
    "قوائم العقوبات":           "sanctions lists screening UN OFAC FATF PCCML SAMA EN 1704 AML prohibited",
    "فحص العقوبات":             "sanctions screening lists UN OFAC FATF PCCML SAMA EN 1704 AML",
    "تقرير الاشتباه":           "suspicious activity report SAR SAFIU SAMA EN 1704 AML CTF reporting",
    "سير العمل لفتح الحساب":    "bank account opening workflow step by step SAMA EN 1644 procedures",
    "المراجعة اليدوية":         "manual review compliance approval bank account high risk SAMA EN 1644",
    "التحقق من الهوية البيومترية": "biometric identity verification Absher NafathID SAMA digital onboarding",
    "التوقيع الرقمي":           "electronic digital signature Electronic Transactions Law Saudi Arabia",
    "بطاقة الهوية منتهية الصلاحية": "expired ID documents bank account exception renewal SAMA EN 1644 180 days",
    "هوية تحت التجديد":         "national ID under renewal exception bank account SAMA EN 1644 180 days",
    "عميل قاصر":               "minor customer bank account guardian SAMA EN 1644 15 18 hijri years",
    "حساب مشترك":              "joint bank account opening SAMA EN 1644 digital",
    "تأهيل الشركات":            "corporate onboarding requirements documents commercial register SAMA EN 1644",
    "السجل التجاري":            "commercial register CR validation verification SAMA EN 1644 Ministry of Commerce",
    "عقد التأسيس":              "articles of association memorandum validation corporate onboarding SAMA EN 1644",
    "طلب غير مكتمل":            "incomplete application missing documents bank account SAMA EN 1644 requirements",
    "تعارض الأسماء":            "name mismatch customer documents identity verification SAMA EN 1644",
    "التأهيل عن بُعد":          "remote onboarding digital account opening SAMA EN 1644 SAMA EN 2888",
    "حدود فتح الحساب الرقمي":   "digital account opening limits SAMA EN 1644 mobile online banking",
    "فشل التحقق البيومتري":     "biometric verification failure fallback OTP SMS alternative SAMA EN 2888",
}


def _expand_query(query: str) -> str:
    """
    Match expansion keys against the query and append expansion terms.
    Issue 4 Fix: Also tries fuzzy multi-word matching that tolerates 1-2
    filler words between key terms (e.g. "who all cannot" matches "who cannot").
    """
    q = query.strip()
    expansions = []
    q_lower = q.lower()
    for key, expansion in QUERY_EXPANSIONS.items():
        key_lower = key.lower()
        matched = False

        # Standard exact match
        if len(key) <= 20:
            if re.search(rf"\b{re.escape(key_lower)}\b", q_lower):
                matched = True
        else:
            if key_lower in q_lower:
                matched = True

        # Issue 4 Fix: Fuzzy match for short multi-word keys (2-4 words)
        # Allows up to 2 words between the key's words
        if not matched and 2 <= len(key_lower.split()) <= 4 and len(key) <= 30:
            words = key_lower.split()
            # Build pattern: word1 ... word2 ... word3 with up to 2 words between each
            fuzzy_pattern = r"\b" + r"\b(?:\s+\w+){0,2}\s+\b".join(re.escape(w) for w in words) + r"\b"
            if re.search(fuzzy_pattern, q_lower):
                matched = True
                print(f"[fuzzy_expand] '{key}' matched in '{q_lower}'")

        if matched:
            expansions.append(expansion)

    return (q + " " + " ".join(expansions)).strip() if expansions else q


_supabase     = None
_embedder     = None
_reranker     = None
_qwen_pipe    = None
_mem_cache:   list[dict] = []
_redis_client = None

def _get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase

def _get_embedder():
    global _embedder
    if _embedder is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embedder] Loading {EMBEDDING_MODEL} on {device.upper()}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"[embedder] Ready on {device.upper()}.")
    return _embedder

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print("[reranker] Loading BAAI/bge-reranker-v2-m3 ...")
            _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            print("[reranker] Loaded.")
        except Exception as e:
            print(f"[reranker] WARNING: Could not load reranker: {e}. Falling back to vector-only.")
            _reranker = "disabled"
    return _reranker

def _get_qwen():
    global _qwen_pipe
    if _qwen_pipe is None:
        from transformers import pipeline
        import torch
        print(f"[llm] Loading {QWEN_MODEL_ID}...")
        _qwen_pipe = pipeline(
            "text-generation", model=QWEN_MODEL_ID,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", trust_remote_code=True,
        )
        print("[llm] Loaded.")
    return _qwen_pipe

def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis as redis_lib
        client = redis_lib.from_url(
            REDIS_URL, decode_responses=False,
            socket_timeout=2, socket_connect_timeout=2,
        )
        client.ping()
        count = client.llen("sama:cache:embeddings")
        print(f"[cache] Redis connected. Cached entries: {count}")
        _redis_client = client
    return _redis_client

def _embed(text: str) -> list[float]:
    model = _get_embedder()
    prefixed = f"query: {text}" if "e5" in EMBEDDING_MODEL.lower() else text
    return model.encode(prefixed, normalize_embeddings=True).tolist()

_EMBED_KEY = "sama:cache:embeddings"

def _cache_lookup(vec: list[float]) -> dict | None:
    if not CACHE_ENABLED: return None
    if CACHE_BACKEND == "redis" and REDIS_URL:
        try:
            r = _get_redis()
            raw_list = r.lrange(_EMBED_KEY, 0, -1)
            if not raw_list: return None
            q = np.array(vec)
            best_idx, best_sim = -1, 0.0
            for i, raw in enumerate(raw_list):
                cached_vec = np.array(json.loads(raw))
                if cached_vec.shape != q.shape:
                    # Dimension mismatch — embedding model changed; skip cache
                    print(f"[cache] Dimension mismatch ({cached_vec.shape} vs {q.shape}) — cache miss")
                    break
                sim = float(np.dot(q, cached_vec))
                if sim > best_sim:
                    best_sim, best_idx = sim, i
            if best_sim >= CACHE_SIM_THRESH and best_idx >= 0:
                raw_result = r.get(f"sama:cache:results:{best_idx}")
                if raw_result:
                    print(f"[cache] HIT (redis sim={best_sim:.4f})")
                    return json.loads(raw_result)
        except Exception as e:
            print(f"[cache] Redis lookup failed: {e}")
    else:
        q = np.array(vec)
        for entry in _mem_cache:
            if float(np.dot(q, np.array(entry["embedding"]))) >= CACHE_SIM_THRESH:
                print("[cache] HIT (memory)")
                return entry["result"]
    return None

def _cache_store(vec: list[float], result: dict) -> None:
    if not CACHE_ENABLED: return
    if CACHE_BACKEND == "redis" and REDIS_URL:
        try:
            r = _get_redis()
            idx = r.llen(_EMBED_KEY)
            r.rpush(_EMBED_KEY, json.dumps(vec))
            r.setex(f"sama:cache:results:{idx}", CACHE_TTL_SECONDS, json.dumps(result))
            r.expire(_EMBED_KEY, CACHE_TTL_SECONDS)
            print(f"[cache] STORED redis idx={idx}")
        except Exception as e:
            print(f"[cache] Redis store failed: {e}")
            _mem_cache.append({"embedding": vec, "result": result})
    else:
        _mem_cache.append({"embedding": vec, "result": result})



# ── [Step 2 — RAG-Fusion] Multi-query retrieval with Reciprocal Rank Fusion ───

def _generate_query_variants(query: str) -> list[str]:
    """
    Generate query variants for RAG-Fusion multi-query retrieval.
    Returns [original_query] + LLM-generated variants.
    Always includes 1 Arabic variant (English query) or 1 English variant (Arabic query).
    Falls back to [original_query] on any failure — pipeline continues normally.
    """
    if not RAG_FUSION_ENABLED:
        return [query]

    # Short queries don't benefit from multi-query
    if len(query.split()) < 5:
        return [query]

    is_arabic_q = _is_arabic(query)
    lang_instruction = (
        "Include 1 variant in English."
        if is_arabic_q else
        "Include 1 variant in Arabic (translating the regulatory concepts)."
    )

    prompt = (
        "You are a Saudi banking and cybersecurity regulation expert.\n"
        f"Generate {RAG_FUSION_VARIANTS} different phrasings of this regulatory question.\n"
        f"{lang_instruction}\n"
        "Return ONLY the phrasings, one per line, no numbering, no explanation.\n"
        f"Question: {query}"
    )

    try:
        if LLM_BACKEND == "azure" and AZURE_OPENAI_KEY and AZURE_ENDPOINT:
            import openai as _oai
            client = _oai.AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                azure_endpoint=AZURE_ENDPOINT,
                api_version="2024-02-01",
            )
            model = AZURE_DEPLOYMENT
        elif OPENAI_API_KEY:
            import openai as _oai
            client = _oai.OpenAI(api_key=OPENAI_API_KEY)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            return [query]

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        raw      = resp.choices[0].message.content.strip()
        variants = [v.strip() for v in raw.split("\n") if v.strip() and len(v.strip()) > 5]

        all_queries = [query]
        for v in variants[:RAG_FUSION_VARIANTS]:
            if v.lower() != query.lower():
                all_queries.append(v)

        print(f"[rag_fusion] {len(all_queries)-1} variants generated for: '{query[:60]}'")
        return all_queries

    except Exception as e:
        print(f"[rag_fusion] Variant generation failed (non-fatal): {e}")
        return [query]


def _reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked chunk lists using Reciprocal Rank Fusion (RRF).
    Score = sum(1 / (k + rank)) across all query variant result lists.
    Chunks appearing highly ranked in multiple lists score highest.
    k=60 is the standard constant from the original RRF paper.
    """
    scores:    dict[str, float] = {}
    chunk_map: dict[str, dict]  = {}

    for results in results_list:
        for rank, chunk in enumerate(results, 1):
            cid = chunk.get("id", "")
            if not cid:
                # Fallback ID for chunks without UUID
                cid = f"{chunk.get('document_name','')}_{chunk.get('page_start','')}_{rank}"

            scores[cid]    = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = chunk_map.get(cid, chunk)

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    merged = []
    for cid in sorted_ids:
        chunk = chunk_map[cid].copy()
        # Use RRF score as similarity proxy for downstream compatibility
        chunk["similarity"] = round(scores[cid], 6)
        merged.append(chunk)

    return merged


# -- [Step 4 - Step-Back Prompting] Abstract query generation ----------------

_STEP_BACK_EXAMPLES = (
    "Specific: What documents does an SME need to open a bank account?\n"
    "Abstract: What are SAMA account opening requirements for business entities?\n\n"
    "Specific: What is the minimum capital adequacy ratio for Saudi banks?\n"
    "Abstract: What are Basel III capital requirements under SAMA regulations?\n\n"
    "Specific: Can a politically exposed person open a bank account in Saudi Arabia?\n"
    "Abstract: What are SAMA KYC and due diligence requirements for high-risk customers?\n\n"
    "Specific: \u0645\u0627 \u0647\u064a \u0645\u062a\u0637\u0644\u0628\u0627\u062a \u0641\u062a\u062d \u062d\u0633\u0627\u0628 \u0644\u0644\u0645\u0646\u0634\u0622\u062a \u0627\u0644\u0635\u063a\u064a\u0631\u0629 \u0648\u0627\u0644\u0645\u062a\u0648\u0633\u0637\u0629\u061f\n"
    "Abstract: \u0645\u0627 \u0647\u064a \u0645\u062a\u0637\u0644\u0628\u0627\u062a \u0633\u0627\u0645\u0627 \u0644\u0641\u062a\u062d \u0627\u0644\u062d\u0633\u0627\u0628\u0627\u062a \u0627\u0644\u0628\u0646\u0643\u064a\u0629 \u0644\u0644\u0634\u0631\u0643\u0627\u062a\u061f\n"
)


def _generate_step_back_query(query: str) -> Optional[str]:
    """
    [Step 4 - Step-Back Prompting] Generate an abstract version of the query
    that captures the underlying regulatory principle.

    Example:
      Specific: "What documents does an SME need to open a bank account?"
      Abstract: "What are SAMA account opening requirements for business entities?"

    The abstract query is searched alongside the original and variants.
    Results merged via RRF so reranker sees both specific and general context.
    Falls back to None on any failure. Set STEP_BACK_ENABLED=false to disable.
    """
    if not STEP_BACK_ENABLED:
        return None
    if len(query.split()) < 5:
        return None

    prompt = (
        "You are a Saudi banking and cybersecurity regulation expert.\n\n"
        "Given a specific regulatory question, write a MORE GENERAL version that captures "
        "the underlying regulatory principle or framework being asked about.\n\n"
        "Rules:\n"
        "- Abstract to the regulatory principle, not the specific detail\n"
        "- Stay in the same language as the input question\n"
        "- Return ONLY the abstract question, maximum 20 words, no explanation\n\n"
        "Examples:\n"
        + _STEP_BACK_EXAMPLES +
        f"Specific: {query}\nAbstract:"
    )

    try:
        if LLM_BACKEND == 'azure' and AZURE_OPENAI_KEY and AZURE_ENDPOINT:
            import openai as _oai
            client = _oai.AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                azure_endpoint=AZURE_ENDPOINT,
                api_version="2024-02-01",
            )
            model = AZURE_DEPLOYMENT
        elif OPENAI_API_KEY:
            import openai as _oai
            client = _oai.OpenAI(api_key=OPENAI_API_KEY)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            return None

        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=50,
            temperature=0.1,
        )
        abstract = resp.choices[0].message.content.strip().strip('"\' ')
        if abstract and len(abstract.split()) >= 3 and abstract.lower() != query.lower():
            print(f"[step_back] '{query[:50]}' -> '{abstract}'")
            return abstract
        return None

    except Exception as e:
        print(f'[step_back] Failed (non-fatal): {e}')
        return None

def fetch_chunks(query_vec: list[float], limit: int | None = None, language_filter: str | None = None) -> list[dict]:
    # [Step 5] Switch RPC based on which embedding column is active
    rpc_name = "match_chunks_bge" if USE_BGE_COLUMN else "match_chunks"
    rpc = _get_supabase().rpc(rpc_name, {
        "query_embedding": query_vec,
        "match_threshold": SIMILARITY_THRESHOLD,
        "match_count":     (limit or TOP_K) * (2 if language_filter else 1),
    })
    if language_filter:
        rpc = rpc.eq("language", language_filter)
    results = rpc.execute().data or []
    return results[:(limit or TOP_K)]

def fetch_chunks_keyword(query: str, limit: int = 10, language_filter: str | None = None) -> list[dict]:
    try:
        rpc = _get_supabase().rpc("keyword_search_chunks", {
            "search_query": query,
            "match_count":  limit * (2 if language_filter else 1),
        })
        if language_filter:
            rpc = rpc.eq("language", language_filter)
        results = rpc.execute().data or []
        results = results[:limit]
        for r in results:
            if "similarity" not in r:
                r["similarity"] = 0.75
        return results
    except Exception as e:
        print(f"[hybrid] Keyword search unavailable: {e}")
        return []

def fetch_chunks_hybrid(query: str, query_vec: list[float], limit: int = 15, subject: str = "", language_filter: str | None = None) -> list[dict]:
    vector_results  = fetch_chunks(query_vec, limit=limit, language_filter=language_filter)
    keyword_results: list[dict] = []
    if HYBRID_SEARCH:
        # Run keyword search on the full expanded query
        keyword_results = fetch_chunks_keyword(query, limit=limit, language_filter=language_filter)
        # Also run keyword search on the extracted subject if it differs meaningfully
        if subject and subject.lower() != query.lower() and len(subject) >= 3:
            subject_results = fetch_chunks_keyword(subject, limit=limit, language_filter=language_filter)
            # Merge subject results in — dedup happens below
            keyword_results = keyword_results + subject_results
    seen_ids: set = set()
    merged: list[dict] = []
    for chunk in vector_results + keyword_results:
        cid = chunk.get("id")
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)
        merged.append(chunk)
    return merged


def fetch_chunks_language_balanced(
    query: str,
    query_vec: list[float],
    limit: int = 15,
    subject: str = "",
) -> list[dict]:
    """
    [Step 6 - Language-Balanced Retrieval] Fetch equal Arabic and English chunks
    separately then merge. Prevents English chunk dominance over Arabic content.

    Without: Arabic query might return 8 English + 2 Arabic from top-10
    With:    Arabic query always returns up to 5 Arabic + 5 English chunks
    """
    half = max(limit // 2, 5)
    ar_chunks = fetch_chunks_hybrid(query, query_vec, limit=half,
                                    subject=subject, language_filter="ar")
    en_chunks = fetch_chunks_hybrid(query, query_vec, limit=half,
                                    subject=subject, language_filter="en")
    print(f"[lang_balanced] AR={len(ar_chunks)} EN={len(en_chunks)}")
    seen_ids: set = set()
    merged: list[dict] = []
    for chunk in ar_chunks + en_chunks:
        cid = chunk.get("id")
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)
        merged.append(chunk)
    return merged

def rerank_chunks(query: str, chunks: list[dict], top_n: int = 5) -> tuple[list[dict], float | None]:
    if not RERANKER_ENABLED or not chunks:
        return chunks[:top_n], None
    reranker = _get_reranker()
    if reranker == "disabled":
        return chunks[:top_n], None
    try:
        pairs = [(query, c.get("content", "")) for c in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top = [c for _, c in ranked[:top_n]]
        top_score = float(ranked[0][0])
        print(f"[reranker] {len(chunks)} -> {top_n}. Top score: {top_score:.3f}")
        return top, top_score
    except Exception as e:
        print(f"[reranker] Failed: {e}. Using original order.")
        return chunks[:top_n], None

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        doc   = c.get("document_name", "Unknown")
        p_s   = c.get("page_start", "?")
        p_e   = c.get("page_end", "?")
        title = c.get("section_title") or ""
        ref   = f"{doc}, Pages {p_s}-{p_e}" + (f", {title}" if title else "")
        parts.append(f"[Passage {i}] ({ref})\n{c['content']}")
    return "\n\n".join(parts)

def _user_prompt(context_text: str, query: str, session_summary: str = "", max_sentences: int = 3) -> str:
    summary_block = (
        f"<conversation_context>\n{session_summary}\n</conversation_context>\n\n"
        if session_summary else ""
    )
    if _is_arabic(query):
        instruction = (
            f"Answer in Arabic in up to {max_sentences} clean natural sentences. "
            "Do NOT include any document names, file names, or page numbers in the answer. "
            "If not found: لا تتوفر إجابة في الوثائق المقدمة"
        )
    else:
        instruction = (
            f"Answer in ENGLISH in up to {max_sentences} clean natural sentences. "
            "Even if the context passages are in Arabic, your answer must be in English. "
            "Do NOT include any document names, file names, page numbers, or parenthetical source references in the answer text."
        )
    return f"{summary_block}<context>\n{context_text}\n</context>\n\nQuestion: {query}\n\n{instruction}\n\nAnswer:"

_DRIFT_SIGNALS = [
    "in many countries", "it is important to note", "it should be noted",
    "generally speaking", "in general", "typically", "in most cases",
    "it is worth noting", "by adhering to", "this ensures that",
    "overall,", "in summary,", "in conclusion,", "furthermore, banks",
    "moreover, banks", "additionally, banks must", "international monetary fund",
    "world bank", "central bank of saudi arabia (cba)",
]

def _truncate_at_drift(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    for sent in sentences:
        if any(s in sent.lower() for s in _DRIFT_SIGNALS):
            break
        kept.append(sent)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept).strip() if kept else text

def _strip_inline_citations(text: str) -> str:
    """
    FIX v5: Post-processing safety net — strip any inline citations the LLM
    may still produce despite the prompt instruction, e.g.:
      (SAMA EN 1644 VER1, Page 44)
      (Page 100)
      (SAMA Basel III Guidelines, Pages 15-16)
    """
    # Remove patterns like (Any text, Page X) or (Any text, Pages X-Y)
    text = re.sub(r"\s*\([^)]*[Pp]ages?\s*\d+[^)]*\)", "", text)
    # Remove standalone (Page X) or (Pages X-Y)
    text = re.sub(r"\s*\([Pp]ages?\s*[\d\-]+\)", "", text)
    # Clean up any double spaces left behind
    text = re.sub(r"  +", " ", text).strip()
    return text

def _clean_output(text: str, query: str, max_sentences: int = 3) -> str:
    for marker in ["Question:", "User:", "Human:", "<context>", "Note:", "System:"]:
        if marker in text:
            text = text[:text.index(marker)].strip()
    text = re.sub(
        r"[\u4e00-\u9fff\u3000-\u303f\u3100-\u312f\uac00-\ud7af\uff00-\uffef\u2e80-\u2eff]+",
        "", text
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if not _is_arabic(query):
        text = _truncate_at_drift(text, max_sentences=max_sentences)
    # FIX v5: Always strip any remaining inline citations as a safety net
    text = _strip_inline_citations(text)
    return text

def _generate_qwen(ctx: str, query: str, on_chunk: Optional[Callable] = None,
                   session_summary: str = "", max_tokens: int = 300, max_sentences: int = 3) -> str:
    pipe = _get_qwen()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _user_prompt(ctx, query, session_summary, max_sentences)},
    ]
    full_input = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = pipe(
        full_input,
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
        temperature=None,
        top_p=None,
    )
    answer = _clean_output(out[0]["generated_text"], query, max_sentences)
    if on_chunk: on_chunk(answer)
    return answer

def _generate_openai(ctx: str, query: str, on_chunk: Optional[Callable] = None,
                     session_summary: str = "", max_tokens: int = 300, max_sentences: int = 3) -> str:
    import openai
    stream = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": _user_prompt(ctx, query, session_summary, max_sentences)}],
        temperature=0.1, max_tokens=max_tokens, stream=True,
    )
    answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        answer += delta
        if on_chunk and delta: on_chunk(delta)
    return answer

def _generate_azure(ctx: str, query: str, on_chunk: Optional[Callable] = None,
                    session_summary: str = "", max_tokens: int = 300, max_sentences: int = 3) -> str:
    import openai
    stream = openai.AzureOpenAI(
        api_key=AZURE_OPENAI_KEY, azure_endpoint=AZURE_ENDPOINT, api_version="2024-02-01",
    ).chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": _user_prompt(ctx, query, session_summary, max_sentences)}],
        temperature=0.1, max_tokens=max_tokens, stream=True,
    )
    answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        answer += delta
        if on_chunk and delta: on_chunk(delta)
    return answer

def _generate(ctx: str, query: str, on_chunk: Optional[Callable] = None,
              session_summary: str = "", max_tokens: int = 300, max_sentences: int = 3) -> str:
    if LLM_BACKEND == "openai":
        return _generate_openai(ctx, query, on_chunk, session_summary, max_tokens, max_sentences)
    if LLM_BACKEND == "azure":
        return _generate_azure(ctx, query, on_chunk, session_summary, max_tokens, max_sentences)
    return _generate_qwen(ctx, query, on_chunk, session_summary, max_tokens, max_sentences)

def answer_query(
    user_query: str,
    top_k: int | None = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    debug: bool = False,
    session_summary: str = "",
    last_messages: list[dict] | None = None,
    **kwargs,
) -> dict:
    if not user_query or not user_query.strip():
        return {"answer": "Please provide a question.", "sources": [], "cached": False, "method": "none"}

    query = user_query.strip()

    if _is_identity_question(query):
        resp = IDENTITY_RESPONSE_AR if _is_arabic(query) else IDENTITY_RESPONSE
        if on_chunk: on_chunk(resp)
        return {"answer": resp, "sources": [], "cached": False, "method": "identity"}

    if _is_nora_definition_query(query):
        if on_chunk:
            on_chunk(NORA_FALLBACK)
        return {
            "answer": NORA_FALLBACK,
            "sources": [],
            "cached": False,
            "method": "generative",
            "candidate_count": 0,
            "reranker_top_score": None,
        }

    if _is_out_of_scope(query):
        answer = "This question is outside the scope of SAMA/banking regulatory documentation."
        if on_chunk: on_chunk(answer)
        return {"answer": answer, "sources": [], "cached": False, "method": "out_of_scope"}

    # Issue 1 Fix: normalise informal/abbreviated language first
    query = _normalise_informal(query)
    # [FIX v7] Rewrite follow-up questions as standalone using conversation context
    query = _contextualize_query(query, session_summary, last_messages or [])
    # Issue 5 Fix: convert yes/no questions to factual regulatory questions
    query = _normalise_yes_no(query)
    # Normalise meta-phrasing ("what do you know about X" → "what is X")
    query = _normalise_query(query)
    # Extract the core subject for enhanced BM25 keyword search
    subject = _extract_subject(query)

    expanded  = _expand_query(query)
    # Issue 7 Fix: inject domain anchor for restriction queries with no SAMA signal
    expanded  = _inject_domain_anchor(query, expanded)
    query_vec = _embed(expanded)

    cached = _cache_lookup(query_vec)
    if cached:
        # Issue 2 Fix: strip any inline citations from cached answers before returning
        cached_answer = _strip_inline_citations(cached.get("answer", ""))
        cached["answer"] = cached_answer
        if on_chunk: on_chunk(cached_answer)
        return {**cached, "cached": True, "method": "cached"}

    final_top_k = top_k or TOP_K
    config      = _answer_config(final_top_k)
    fetch_k     = config["fetch_k"]
    max_tokens  = config["max_tokens"]
    max_sentences = config["max_sentences"]

    if debug:
        print(f"[pipeline] top_k={final_top_k} → sentences={max_sentences}, tokens={max_tokens}, fetch_k={fetch_k}")

    # Pick fetch strategy based on LANG_BALANCED_ENABLED
    def _fetch(q, vec, limit, subject=''):
        if LANG_BALANCED_ENABLED:
            return fetch_chunks_language_balanced(q, vec, limit=limit, subject=subject)
        return fetch_chunks_hybrid(q, vec, limit=limit, subject=subject)

    # [Step 4 - Step-Back Prompting] Generate abstract query before retrieval
    step_back = None
    if STEP_BACK_ENABLED and len(query.split()) >= 5:
        step_back = _generate_step_back_query(query)

    # [Step 2 - RAG-Fusion] Multi-query retrieval + RRF merge
    # Step-back result added as extra pool in RRF
    # NOTE: track max_raw_sim from raw results BEFORE RRF overwrites similarity
    #       with tiny RRF scores — needed for LOW_CONF_THRESHOLD check below
    long_enough  = len(query.split()) >= 5
    max_raw_sim  = 0.0   # will be set from raw retrieval results

    if RAG_FUSION_ENABLED and long_enough:
        variants    = _generate_query_variants(query)
        all_results = []
        for variant in variants:
            v_exp = _inject_domain_anchor(variant, _expand_query(variant))
            v_vec = _embed(v_exp)
            v_res = _fetch(v_exp, v_vec, limit=fetch_k, subject=subject)
            all_results.append(v_res)
            if v_res:
                max_raw_sim = max(max_raw_sim,
                    max(float(c.get("similarity", 0)) for c in v_res))
        # Add step-back abstract results to the pool
        if step_back:
            sb_exp = _inject_domain_anchor(step_back, _expand_query(step_back))
            sb_vec = _embed(sb_exp)
            sb_res = _fetch(sb_exp, sb_vec, limit=fetch_k, subject=subject)
            all_results.append(sb_res)
            if sb_res:
                max_raw_sim = max(max_raw_sim,
                    max(float(c.get("similarity", 0)) for c in sb_res))
        candidates = _reciprocal_rank_fusion(all_results)[:fetch_k]
        if debug:
            n = len(variants) + (1 if step_back else 0)
            print(f'[pipeline] {n} query pools -> {len(candidates)} RRF-merged candidates | raw_sim={max_raw_sim:.4f}')
    elif step_back:
        # RAG-Fusion off but step-back on — merge original + abstract
        orig_res = _fetch(expanded, query_vec, limit=fetch_k, subject=subject)
        sb_exp   = _inject_domain_anchor(step_back, _expand_query(step_back))
        sb_vec   = _embed(sb_exp)
        sb_res   = _fetch(sb_exp, sb_vec, limit=fetch_k, subject=subject)
        candidates = _reciprocal_rank_fusion([orig_res, sb_res])[:fetch_k]
        all_raw = orig_res + sb_res
        max_raw_sim = max((float(c.get("similarity", 0)) for c in all_raw), default=0.0)
    else:
        candidates  = _fetch(expanded, query_vec, limit=fetch_k, subject=subject)
        max_raw_sim = max((float(c.get("similarity", 0)) for c in candidates), default=0.0)

    if debug:
        print(f"\n[pipeline] {len(candidates)} hybrid candidates for: '{query}'")
        print(f"[pipeline] subject: '{subject}'")
        print(f"[pipeline] expanded: '{expanded[:120]}...'")
        for i, c in enumerate(candidates[:5]):
            print(f"  [{i+1}] sim={c.get('similarity',0):.4f} | {c.get('document_name','?')} p{c.get('page_start','?')}")

    if not candidates:
        if on_chunk: on_chunk(NOT_FOUND)
        return {"answer": NOT_FOUND, "sources": [], "cached": False, "method": "not_found"}

    chunks, reranker_top_score = rerank_chunks(query, candidates, top_n=final_top_k)

    # Use max_raw_sim (pre-RRF cosine similarity) for confidence check.
    # RRF scores (~0.016) are not comparable to LOW_CONF_THRESHOLD (~0.72).
    # Fall back to candidates similarity when RAG-Fusion was not used.
    top_sim = max_raw_sim if max_raw_sim > 0 else max(
        float(c.get("similarity", 0)) for c in candidates
    )

    if top_sim < LOW_CONF_THRESHOLD:
        if debug: print(f"[pipeline] low confidence ({top_sim:.4f} < {LOW_CONF_THRESHOLD})")
        if on_chunk: on_chunk(NOT_FOUND)
        return {"answer": NOT_FOUND, "sources": [], "cached": False, "method": "not_found"}

    answer = _generate(build_context(chunks), query, on_chunk, session_summary=session_summary,
                       max_tokens=max_tokens, max_sentences=max_sentences)
    answer = _strip_trailing_not_found(answer)

    # If LLM says not found, return empty sources
    if _is_not_found_answer(answer):
        if debug: print(f"[pipeline] LLM returned not-found — clearing sources for clean UX")
        return {
            "answer": answer,
            "sources": [],
            "cached": False,
            "method": "generative",
            "candidate_count": len(candidates),
            "reranker_top_score": reranker_top_score,
        }

    seen: set[tuple] = set()
    sources = []
    for c in chunks:
        key = (c.get("document_name", ""), c.get("page_start"), c.get("page_end"))
        if key in seen: continue
        seen.add(key)
        sources.append({
            "document_name": c.get("document_name", "Unknown"),
            "page_start":    c.get("page_start"),
            "page_end":      c.get("page_end"),
            "section_title": c.get("section_title"),
            "similarity":    round(float(c.get("similarity", 0)), 4),
            "snippet":       (c.get("content") or "")[:SNIPPET_CHAR_LIMIT],
        })

    result = {"answer": answer, "sources": sources, "cached": False, "method": "generative",
             "candidate_count": len(candidates), "reranker_top_score": reranker_top_score}
    if not _is_not_found_answer(answer):
        _cache_store(query_vec, result)
    return result


def format_response_for_display(user_query: str, result: dict) -> str:
    answer  = (result.get("answer") or "").strip()
    sources = result.get("sources") or []
    lines   = [f"User's question    : {user_query}", f"IOTA AI's Response : {answer}", "Sources :"]
    if not sources:
        lines.append("  (none)")
    else:
        for i, s in enumerate(sources, 1):
            lines.append(f"  {i}. {s.get('document_name','')} (pages {s.get('page_start','?')}-{s.get('page_end','?')}) sim={s.get('similarity',0)}")
    return "\n".join(lines)
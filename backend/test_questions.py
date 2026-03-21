"""
test_questions.py — Test the SAMA chatbot API
Scoring:
  - Method check: did the system behave correctly (route correctly)?
  - LLM-as-judge: is the answer grounded in the retrieved chunks?
  - Source verification: do key phrases from the answer appear in retrieved snippets?

Run while api.py is running:
    python test_questions.py
"""

import time
import requests
import json
import os
import re
import openai
from datetime import datetime

API_URL = "http://localhost:8000/api/query"
TIMEOUT = 420

# ── Log file setup ────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "test_logs")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"test_run_{RUN_TIMESTAMP}.log")
JSON_FILE = os.path.join(LOG_DIR, f"test_run_{RUN_TIMESTAMP}.json")

# ── OpenAI client for LLM-as-judge ───────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
_judge_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

QUESTIONS = [
    # Core SAMA knowledge
    ("What is SAMA?",                                                               "regulatory"),
    ("What is NORA?",                                                               "regulatory"),

    # Regulation content
    ("What are the minimum capital adequacy requirements for banks under SAMA?",     "regulatory"),
    ("Who cannot open a bank account in Saudi Arabia?",                             "regulatory"),
    ("What are the AML requirements under SAMA?",                                   "regulatory"),
    ("What are the know your customer requirements for retail customers?",           "regulatory"),
    ("What are the rules for opening bank accounts in Saudi Arabia?",               "regulatory"),
    ("What is the minimum capital requirement for a new bank license?",             "regulatory"),

    # Tricky / hallucination tests
    ("Did King Abdullah sign or write any SAMA documents?",                         "trick"),
    ("What are the KYC requirements for retail customers?",                         "regulatory"),

    # Out of scope
    ("What is the weather in Riyadh?",                                              "out_of_scope"),
    ("Who is the CEO of Apple?",                                                    "out_of_scope"),

    # Arabic
    ("ما هو البنك المركزي السعودي؟",                                               "arabic"),
    ("ما هي متطلبات رأس المال للبنوك؟",                                            "arabic"),

    # Auto-generated from chunk content
    ("What are the recommended password protection measures for third-party access?", "regulatory"),
    ("What are the requirements for conducting penetration testing according to the cybersecurity standards?", "regulatory"),
    ("How should Member Organisations handle authentication for higher risk transactions?", "regulatory"),
    ("What cybersecurity requirements must be implemented for mobile devices under a BYOD policy?", "regulatory"),
    ("How often must audits of the security of OT/ICS networks be carried out?",    "regulatory"),
    ("What are the criticality levels defined for facilities in an OT/ICS cybersecurity environment?", "regulatory"),
    ("What are the criteria for a bank to be eligible for the IRB Approach for capital adequacy?", "regulatory"),
    ("What is the minimum amount of HQLA that a bank must maintain according to Basel III LCR standards?", "regulatory"),
    ("What is the cap on cash inflows as a percentage of total cash outflows?",     "regulatory"),
    ("How should banks calculate the leverage ratio according to regulatory guidelines?", "regulatory"),
    ("What is the loss event threshold used in the operational risk capital calculation?", "regulatory"),
    ("What is the risk weight for a loan-to-value ratio greater than 100%?",        "regulatory"),
    ("What entities must comply with the Anti-Money Laundering regulations regarding prepaid products?", "regulatory"),
    ("How should a financial institution respond when submitting a report to SAFIU on suspicious transactions?", "regulatory"),
    ("What measures must banks take to protect customers assets against fraud?",     "regulatory"),
    ("What actions should a bank take if a fraud investigation is initiated?",      "regulatory"),
    ("What documents are required for opening an account for national societies and committees?", "regulatory"),
    ("What are the requirements for opening an escrow account for a real estate developer in Saudi Arabia?", "regulatory"),
    ("What are the requirements for opening bank accounts for government entities to receive donations?", "regulatory"),
    ("What must a bank do when contacted about an incompetent person accounts?",    "regulatory"),
    ("What is the maximum financing limit for each Participant in the Debt-Based Crowdfunding Platform?", "regulatory"),
    ("What types of fees may be imposed for the issuance of a prepaid payment service?", "regulatory"),
    ("What is the maximum amount of administrative service charges that can be recovered from a borrower?", "regulatory"),
    ("What are the general rules for savings products in banks?",                   "regulatory"),
    ("What are the mandatory content requirements for deferred remuneration reporting for banks?", "regulatory"),
    ("What considerations must a bank take into account when implementing clawback arrangements for deferred remuneration?", "regulatory"),
    ("What is the role of banks in ensuring compliance with Shariah principles in their activities?", "regulatory"),
    ("What is the purpose of the internal Shariah audit function?",                 "regulatory"),
    ("What qualifications should risk officers have to manage the risk of non-compliance with Shariah principles?", "regulatory"),
    ("How does the Net Stable Funding Ratio aim to promote resilience in banks funding sources?", "regulatory"),
    ("How are inflows from securities maturing within 30 days treated under Basel III LCR standards?", "regulatory"),
    ("What are the restrictions on a banks ability to transfer liquidity during stressed conditions?", "regulatory"),
    ("Under what conditions may personal data be collected for scientific purposes without consent?", "regulatory"),
    ("What are the general rules for data sharing among entities?",                 "regulatory"),
    ("What must the Controller provide to the competent authority regarding Binding Common Rules compliance?", "regulatory"),
    ("ما هي العقوبات المفروضة على المبالاة أو التقصير وفقًا للأنظمة واللوائح؟", "arabic"),
    ("كيف يعرف IFRS 9 التعرضات المتعثرة؟",                                       "arabic"),
    ("ما هي المتطلبات الخاصة بتقرير نسبة القرض إلى الودائع للبنوك؟",              "arabic"),
    ("ما هي متطلبات الكشف السنوي للبنوك التي تزيد أصولها عن 4.46 مليار ريال سعودي؟", "arabic"),

    # ── Additional English (to reach 50) ─────────────────────────────────────
    ("What is the scope of application of the Saudi Personal Data Protection Law?", "regulatory"),
    ("What penalties does the Saudi PDPL impose for violations?", "regulatory"),
    ("How does ISO 23200 address blockchain and distributed ledger technologies?", "regulatory"),
    ("What is the relationship between NCA and SAMA regarding cybersecurity frameworks?", "regulatory"),

    # ── Additional Arabic (to reach 50) ───────────────────────────────────────
    ("ما نطاق تطبيق نظام حماية البيانات الشخصية على المنظمات السعودية؟", "arabic"),
    ("ما العقوبات المقررة على مخالفة نظام حماية البيانات الشخصية في المملكة؟", "arabic"),
    ("ما هو معيار ISO 23200 وكيف يتعلق بتقنية البلوك تشين؟", "arabic"),
    ("ما العلاقة بين الهيئة الوطنية للأمن السيبراني وساما في مجال الأمن السيبراني؟", "arabic"),
    ("ما متطلبات التدقيق الداخلي والاستعداد للحصول على شهادات GRC؟", "arabic"),
    ("كيف تحدد الهيئة الوطنية للأمن السيبراني متطلبات الرقابة والمراقبة؟", "arabic"),
    ("ما أنواع البيانات الشخصية التي يحميها نظام حماية البيانات الشخصية؟", "arabic"),
    ("ما هو معيار ISO/IEC 27400 وما نطاق تطبيقه في أمن إنترنت الأشياء؟", "arabic"),
    ("ما الفرق بين تقييم النضج وتقييم الفجوات في خدمات GRC؟", "arabic"),
    ("ما التزامات مزودي الخدمات السحابية وفق ضوابط NCA للأمن السيبراني؟", "arabic"),
    ("كيف يدعم إطار الأمن السيبراني لساما عمليات الأمن السيبراني في القطاع المالي؟", "arabic"),
    ("ما الشروط التي يجب على المتحكم في البيانات استيفاؤها عند نقل البيانات خارج المملكة؟", "arabic"),
    ("ما أهداف برنامج الصندوق التنظيمي التجريبي لمؤسسة النقد العربي السعودي؟", "arabic"),

    # ── Aramco CCC / Third Party Cybersecurity (English) ─────────────────────
    ("What is the purpose of the Aramco CCC program for third-party vendors?", "regulatory"),
    ("What standard must all Aramco vendors comply with under the CCC program?", "regulatory"),
    ("What is the difference between CCC Standard and CCC+ assessment levels?", "regulatory"),
    ("Who conducts the remote verification in the Aramco CCC Standard assessment?", "regulatory"),
    ("What does CCC+ assessment involve compared to CCC Standard?", "regulatory"),
    ("What is SACS-002 and which organizations must comply with it?", "regulatory"),
    ("What is a self-compliance assessment in the context of Aramco third-party cybersecurity?", "regulatory"),
    ("Which types of vendors are subject to higher-risk on-site CCC+ evaluation?", "regulatory"),

    # ── GRC Services (English) ────────────────────────────────────────────────
    ("What services are included in a GRC Gap Assessment and Maturity Review?", "regulatory"),
    ("What does SOC and Compliance Alignment involve in GRC services?", "regulatory"),
    ("What is the purpose of Policy and Framework Development in GRC?", "regulatory"),
    ("What is included in Risk Assessment and Control Implementation under GRC?", "regulatory"),
    ("What is meant by Ongoing Managed Compliance Support in GRC services?", "regulatory"),
    ("What does Internal Audit and Certification Readiness cover in GRC?", "regulatory"),

    # ── SAMA Cybersecurity Framework (English) ────────────────────────────────
    ("What sectors does the SAMA Cybersecurity Framework regulate?", "regulatory"),
    ("What are the main focus areas of the SAMA Cybersecurity Framework?", "regulatory"),
    ("How does SAMA address third-party security compliance for financial institutions?", "regulatory"),
    ("What does the SAMA Cybersecurity Framework require regarding incident response?", "regulatory"),
    ("What is the role of risk management in the SAMA Cybersecurity Framework?", "regulatory"),
    ("How does the SAMA Cybersecurity Framework address security operations?", "regulatory"),

    # ── NCA / ECC (English) ───────────────────────────────────────────────────
    ("What is the purpose of the Essential Cybersecurity Controls (ECC) issued by NCA?", "regulatory"),
    ("Which entities must comply with the NCA Essential Cybersecurity Controls?", "regulatory"),
    ("What is the Cybersecurity Compliance Controls (CCC) framework issued by NCA?", "regulatory"),
    ("How does the NCA framework address governance in cybersecurity?", "regulatory"),
    ("What are the monitoring requirements under NCA cybersecurity controls?", "regulatory"),
    ("What does the NCA framework require for incident management?", "regulatory"),
    ("How does NCA define critical infrastructure for cybersecurity purposes?", "regulatory"),
    ("What risk control measures does NCA require from government entities?", "regulatory"),

    # ── PDPL (English) ────────────────────────────────────────────────────────
    ("What is the Personal Data Protection Law (PDPL) in Saudi Arabia?", "regulatory"),
    ("When did the Saudi Personal Data Protection Law come into effect?", "regulatory"),
    ("Which authority is responsible for enforcing the PDPL in Saudi Arabia?", "regulatory"),
    ("What role does SDAIA play in implementing the Personal Data Protection Law?", "regulatory"),
    ("What role does NDMO play under the Personal Data Protection Law?", "regulatory"),
    ("What types of personal data does the PDPL protect?", "regulatory"),
    ("What are the rights of data subjects under the Saudi PDPL?", "regulatory"),
    ("What obligations does the PDPL place on controllers of personal data?", "regulatory"),

    # ── ISO Standards (English) ───────────────────────────────────────────────
    ("What is ISO 27001 and what does it certify?", "regulatory"),
    ("What is the difference between ISO 27001 and ISO 27701?", "regulatory"),
    ("What does ISO 27701 address in terms of privacy information management?", "regulatory"),
    ("What is ISO/IEC 27400 and what does it cover?", "regulatory"),
    ("What is ISO/IEC 27403 and how does it relate to IoT security?", "regulatory"),
    ("What is ISO 20000 and what service management processes does it cover?", "regulatory"),
    ("What is ISO 22301 and what does it require for business continuity?", "regulatory"),
    ("What is ISO 23200 and how does it apply to blockchain technology?", "regulatory"),
    ("What is ISO 42001 and what does it govern for artificial intelligence?", "regulatory"),
    ("What is an Information Security Management System (ISMS) under ISO 27001?", "regulatory"),

    # ── Aramco CCC / Third Party Cybersecurity (Arabic) ──────────────────────
    ("ما هو برنامج شهادة الامتثال للأمن السيبراني (CCC) المطلوب من موردي أرامكو؟", "arabic"),
    ("ما الفرق بين مستوى CCC القياسي ومستوى CCC+ في تقييمات أرامكو؟", "arabic"),
    ("من يقوم بالتحقق عن بُعد في تقييم CCC القياسي لشركة أرامكو؟", "arabic"),
    ("ما هو معيار SACS-002 الذي يجب على موردي أرامكو الامتثال له؟", "arabic"),
    ("ما المقصود بالتقييم الذاتي للامتثال في برنامج أرامكو للأمن السيبراني؟", "arabic"),
    ("ما الأطراف الخارجية التي تخضع لتقييم CCC+ الأكثر صرامة؟", "arabic"),

    # ── GRC Services (Arabic) ─────────────────────────────────────────────────
    ("ما هي خدمات تقييم الفجوات ومراجعة النضج في إطار حوكمة المخاطر والامتثال؟", "arabic"),
    ("ما الذي تشمله خدمة تطوير السياسات والأطر في مجال GRC؟", "arabic"),
    ("ما هو دور تقييم المخاطر وتطبيق الضوابط في خدمات GRC؟", "arabic"),
    ("ما المقصود بالمواءمة مع SOC والامتثال في خدمات حوكمة المخاطر؟", "arabic"),
    ("ما الذي تتضمنه خدمة دعم الامتثال المُدار المستمر في إطار GRC؟", "arabic"),

    # ── SAMA Cybersecurity Framework (Arabic) ────────────────────────────────
    ("ما القطاعات التي ينظمها إطار الأمن السيبراني الصادر عن البنك المركزي السعودي؟", "arabic"),
    ("ما متطلبات إدارة المخاطر في إطار الأمن السيبراني لمؤسسة النقد العربي السعودي؟", "arabic"),
    ("كيف يعالج إطار الأمن السيبراني لساما متطلبات الاستجابة للحوادث؟", "arabic"),
    ("ما متطلبات امتثال الأطراف الثالثة وفق إطار الأمن السيبراني لساما؟", "arabic"),
    ("ما دور عمليات الأمن السيبراني في الإطار الصادر عن البنك المركزي السعودي؟", "arabic"),

    # ── NCA / ECC (Arabic) ───────────────────────────────────────────────────
    ("ما هي الضوابط الأساسية للأمن السيبراني (ECC) الصادرة عن الهيئة الوطنية للأمن السيبراني؟", "arabic"),
    ("ما الجهات الملزمة بالامتثال لضوابط ECC الصادرة عن الهيئة الوطنية للأمن السيبراني؟", "arabic"),
    ("ما الذي تتضمنه ضوابط الامتثال للأمن السيبراني CCC الصادرة عن الهيئة الوطنية؟", "arabic"),
    ("كيف تعرّف الهيئة الوطنية للأمن السيبراني البنية التحتية الحيوية؟", "arabic"),
    ("ما متطلبات الحوكمة في الضوابط الأساسية للأمن السيبراني الصادرة عن NCA؟", "arabic"),
    ("ما متطلبات إدارة الحوادث وفق إطار الهيئة الوطنية للأمن السيبراني؟", "arabic"),
    ("ما ضوابط إدارة المخاطر المطلوبة من الجهات الحكومية وفق معايير NCA؟", "arabic"),

    # ── PDPL (Arabic) ─────────────────────────────────────────────────────────
    ("ما هو نظام حماية البيانات الشخصية في المملكة العربية السعودية؟", "arabic"),
    ("متى دخل نظام حماية البيانات الشخصية حيز التنفيذ في المملكة العربية السعودية؟", "arabic"),
    ("ما الجهة المسؤولة عن تطبيق وإنفاذ نظام حماية البيانات الشخصية في المملكة؟", "arabic"),
    ("ما دور هيئة البيانات والذكاء الاصطناعي SDAIA في تطبيق نظام حماية البيانات؟", "arabic"),
    ("ما دور المكتب الوطني لإدارة البيانات NDMO في الإشراف على حماية البيانات؟", "arabic"),
    ("ما حقوق أصحاب البيانات بموجب نظام حماية البيانات الشخصية السعودي؟", "arabic"),
    ("ما الالتزامات المفروضة على المتحكمين في البيانات بموجب نظام PDPL؟", "arabic"),

    # ── ISO Standards (Arabic) ────────────────────────────────────────────────
    ("ما هو معيار ISO 27001 وما الذي يشهد عليه؟", "arabic"),
    ("ما الفرق بين معيار ISO 27001 ومعيار ISO 27701 في مجال إدارة المعلومات؟", "arabic"),
    ("ما الذي يغطيه معيار ISO 27701 في إدارة معلومات الخصوصية؟", "arabic"),
    ("ما هو معيار ISO 20000 وما عمليات إدارة الخدمات التي يغطيها؟", "arabic"),
    ("ما هو معيار ISO 22301 وما متطلباته لاستمرارية الأعمال؟", "arabic"),
    ("ما هو معيار ISO 42001 وكيف يحكم أنظمة الذكاء الاصطناعي؟", "arabic"),
    ("ما المقصود بنظام إدارة أمن المعلومات ISMS وفق معيار ISO 27001؟", "arabic"),
]

DIVIDER     = "═" * 80
SUB_DIVIDER = "─" * 80

METHOD_LABELS = {
    "generative":   "GENERATIVE  (LLM called — answer from context)",
    "out_of_scope": "OUT OF SCOPE  (rejected — not a regulatory question)",
    "not_found":    "NOT FOUND  (low confidence — chunks not relevant enough)",
    "cached":       "CACHED  (semantic cache hit — no LLM call)",
    "none":         "NO RESULT",
    "unknown":      "UNKNOWN",
}

EXPECTED = {
    0:  "correct",       # What is SAMA?
    1:  "not_found",     # What is NORA? — not in DB
    2:  "correct",       # Capital adequacy
    3:  "correct",       # Who cannot open account
    4:  "correct",       # AML requirements
    5:  "correct",       # KYC requirements
    6:  "correct",       # Rules for opening accounts
    7:  "correct",       # Minimum capital for license
    8:  "trick",         # King Abdullah — should say not found
    9:  "correct",       # KYC (cached)
    10: "out_of_scope",  # Weather
    11: "out_of_scope",  # CEO of Apple
    12: "correct",       # Arabic central bank
    13: "not_found",     # Arabic capital requirements
    **{i: "correct" for i in range(14, 200)},
}

NOT_FOUND_PHRASES = [
    "does not contain", "cannot find", "not found in",
    "لا تتوفر", "لم أجد",
]

def _is_not_found_answer(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in NOT_FOUND_PHRASES)

def _log(msg: str) -> None:
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are a strict grounding evaluator for a regulatory RAG system.

Your job: determine if every factual claim in the ANSWER is explicitly present in the CONTEXT SNIPPETS.

Definitions:
- GROUNDED: every fact, number, and claim in the answer is directly traceable to the snippets
- UNGROUNDED: the answer contains facts, numbers, or claims NOT found anywhere in the snippets  
- PARTIAL: some claims are grounded, but at least one claim adds detail not present in the snippets

Important: paraphrasing is acceptable — the answer doesn't need to copy verbatim. But added specifics (numbers, percentages, names, conditions) that don't appear in the snippets = UNGROUNDED or PARTIAL.

Reply with ONLY one of: GROUNDED / UNGROUNDED / PARTIAL
Then on the next line: one concise sentence explaining which specific claim is ungrounded (if any).

CONTEXT SNIPPETS:
{snippets}

ANSWER TO EVALUATE:
{answer}"""

def llm_judge(answer: str, sources: list[dict]) -> tuple[str, str]:
    """Returns (verdict, reason) where verdict is GROUNDED/UNGROUNDED/PARTIAL."""
    if not answer or not sources:
        return "SKIP", "no answer or no sources to evaluate"
    if _is_not_found_answer(answer):
        return "SKIP", "answer is not-found — nothing to verify"

    snippets = "\n\n".join([
        f"[{s.get('document_name','?')} p{s.get('page_start','?')}]\n{s.get('snippet','')}"
        for s in sources[:5]  # snippets now 500 chars each = ~2500 chars total context
    ])

    try:
        resp = _judge_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                snippets=snippets, answer=answer
            )}],
            temperature=0,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
        lines = raw.split("\n", 1)
        verdict = lines[0].strip().upper()
        reason  = lines[1].strip() if len(lines) > 1 else ""
        if verdict not in ("GROUNDED", "UNGROUNDED", "PARTIAL"):
            verdict = "UNKNOWN"
        return verdict, reason
    except Exception as e:
        return "ERROR", str(e)


# ── Source phrase verification ────────────────────────────────────────────────

def verify_phrases_in_sources(answer: str, sources: list[dict]) -> tuple[float, list[str]]:
    """
    Extract key phrases from answer and check if they appear in source snippets.
    Returns (match_ratio, matched_phrases).
    A high ratio means the answer text is directly traceable to retrieved chunks.
    """
    if not answer or not sources or _is_not_found_answer(answer):
        return 0.0, []

    # Combine all snippets into one searchable string
    all_snippets = " ".join([
        (s.get("snippet") or "") + " " + (s.get("document_name") or "")
        for s in sources
    ]).lower()

    # Extract meaningful phrases from the answer (3-6 word n-grams)
    words = re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', answer)
    phrases = []
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i+3]).lower()
        if len(phrase) > 10:  # skip very short phrases
            phrases.append(phrase)

    if not phrases:
        return 0.0, []

    matched = [p for p in phrases if p in all_snippets]
    # Deduplicate
    matched = list(dict.fromkeys(matched))[:5]
    ratio = len(matched) / len(phrases) if phrases else 0.0
    return round(ratio, 2), matched


# ── Method verdict ────────────────────────────────────────────────────────────

def _score_method(idx: int, method: str, answer: str, timed_out: bool) -> tuple[str, str]:
    expected = EXPECTED.get(idx, "correct")
    if timed_out:
        return "TIMEOUT", "no response within timeout"
    if expected == "out_of_scope":
        if method == "out_of_scope":
            return "CORRECT", "correctly rejected as out of scope"
        return "WRONG", f"should be out_of_scope, got {method}"
    if expected == "not_found":
        if method == "not_found" or _is_not_found_answer(answer):
            return "CORRECT", "correctly returned not found"
        return "HALLUCINATED", "should say not found but returned an answer"
    if expected == "trick":
        if method == "not_found" or _is_not_found_answer(answer):
            return "CORRECT", "correctly said not found for trick question"
        return "HALLUCINATED", "trick question — should say not found"
    # correct
    if method == "out_of_scope":
        return "WRONG", "incorrectly rejected"
    if method == "not_found" or _is_not_found_answer(answer):
        return "WRONG", "returned not found but answer exists in DB"
    if method in ("generative", "cached"):
        return "CORRECT", f"returned answer via {method}"
    return "UNKNOWN", f"unexpected method={method}"


# ── Per-question test ─────────────────────────────────────────────────────────

def test_question(idx: int, total: int, q: str, category: str) -> dict:
    _log(f"\n{DIVIDER}")
    _log(f"  [{idx+1}/{total}] [{category.upper()}] QUESTION")
    _log(f"  {q}")
    _log(DIVIDER)

    start     = time.perf_counter()
    timed_out = False
    answer    = ""
    sources   = []
    method    = "unknown"

    try:
        resp = requests.post(API_URL, json={"query": q, "debug": True}, timeout=TIMEOUT)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data    = resp.json()
        answer  = (data.get("answer") or "").strip()
        sources = data.get("sources") or []
        cached  = data.get("cached", False)
        method  = data.get("method", "cached" if cached else "unknown")
        # New fields added by hybrid+reranker pipeline
        candidate_count = data.get("candidate_count", None)
        reranker_top_score = data.get("reranker_top_score", None)

        method_verdict, method_reason = _score_method(idx, method, answer, False)

        # LLM-as-judge (only for generative/cached answers with sources)
        if method in ("generative", "cached") and sources and not _is_not_found_answer(answer):
            _log(f"\n  [Verifying answer with LLM judge...]")
            judge_verdict, judge_reason = llm_judge(answer, sources)
            phrase_ratio, matched_phrases = verify_phrases_in_sources(answer, sources)
        else:
            judge_verdict, judge_reason = "SKIP", "not applicable"
            phrase_ratio, matched_phrases = 0.0, []

        label = METHOD_LABELS.get(method, method)
        _log(f"\n  METHOD       : {label}")
        _log(f"  TIME         : {elapsed:.2f}s")
        if candidate_count is not None:
            _log(f"  CANDIDATES   : {candidate_count} hybrid candidates fetched")
        if reranker_top_score is not None:
            _log(f"  RERANKER     : top score = {reranker_top_score:.3f}")
        _log(f"  METHOD CHECK : {method_verdict}  ({method_reason})")
        _log(f"  LLM JUDGE    : {judge_verdict}  —  {judge_reason}")
        _log(f"  PHRASE MATCH : {phrase_ratio:.0%} of answer phrases found in retrieved chunks")
        if matched_phrases:
            _log(f"  MATCHED      : {' | '.join(matched_phrases[:3])}")

        _log(f"\n  ANSWER")
        _log(SUB_DIVIDER)
        _log(answer)

        _log(f"\n  SOURCES  ({len(sources)} returned)")
        _log(SUB_DIVIDER)
        if not sources:
            _log("  (no sources)")
        else:
            for i, s in enumerate(sources, 1):
                doc   = s.get("document_name") or "Unknown"
                p_s   = s.get("page_start", "?")
                p_e   = s.get("page_end", "?")
                title = s.get("section_title") or ""
                sim   = s.get("similarity", 0)
                snip  = (s.get("snippet") or "").strip()
                _log(f"\n  [{i}]")
                _log(f"      Document  : {doc}")
                _log(f"      Pages     : {p_s} – {p_e}")
                if title: _log(f"      Section   : {title}")
                _log(f"      Similarity: {sim}")
                if snip:
                    _log(f"      Snippet   :")
                    for line in snip.splitlines():
                        _log(f"          {line}")

        return {
            "q": q, "category": category, "method": method,
            "elapsed": round(elapsed, 2),
            "method_verdict": method_verdict, "method_reason": method_reason,
            "judge_verdict": judge_verdict, "judge_reason": judge_reason,
            "phrase_match_ratio": phrase_ratio, "matched_phrases": matched_phrases,
            "answer": answer,
            "candidate_count": candidate_count,
            "reranker_top_score": reranker_top_score,
            "sources": [{"doc": s.get("document_name"), "sim": s.get("similarity")} for s in sources],
            "timed_out": False,
        }

    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start
        _log(f"\n  [TIMEOUT] No response after {elapsed:.0f}s.")
        return {
            "q": q, "category": category, "method": "timeout",
            "elapsed": round(elapsed, 2),
            "method_verdict": "TIMEOUT", "method_reason": "no response within timeout",
            "judge_verdict": "SKIP", "judge_reason": "",
            "phrase_match_ratio": 0.0, "matched_phrases": [],
            "answer": "", "sources": [], "timed_out": True,
        }

    except requests.exceptions.ConnectionError:
        _log("\n  [CONNECTION ERROR] api.py not running?")
        return {
            "q": q, "category": category, "method": "error",
            "elapsed": 0, "method_verdict": "ERROR", "method_reason": "connection refused",
            "judge_verdict": "SKIP", "judge_reason": "",
            "phrase_match_ratio": 0.0, "matched_phrases": [],
            "answer": "", "sources": [], "timed_out": False,
        }

    except Exception as e:
        elapsed = time.perf_counter() - start
        _log(f"\n  [ERROR after {elapsed:.2f}s] {e}")
        return {
            "q": q, "category": category, "method": "error",
            "elapsed": round(elapsed, 2), "method_verdict": "ERROR", "method_reason": str(e),
            "judge_verdict": "SKIP", "judge_reason": "",
            "phrase_match_ratio": 0.0, "matched_phrases": [],
            "answer": "", "sources": [], "timed_out": False,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total       = len(QUESTIONS)
    total_start = time.perf_counter()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"SAMA Chatbot Test Run\n")
        f.write(f"Timestamp  : {RUN_TIMESTAMP}\n")
        f.write(f"API URL    : {API_URL}\n")
        f.write(f"Questions  : {total}\n")
        f.write(f"Scoring    : method check + LLM-as-judge + phrase matching\n")
        f.write(f"Log file   : {LOG_FILE}\n")
        f.write(f"JSON file  : {JSON_FILE}\n")
        f.write("=" * 80 + "\n\n")

    print(f"\nTesting {total} questions against {API_URL}")
    print(f"Timeout per question: {TIMEOUT}s")
    print(f"Logs saving to: {LOG_FILE}")
    print("Scoring: method check + LLM-as-judge + phrase matching\n")

    all_results = []
    for i, (q, cat) in enumerate(QUESTIONS):
        r = test_question(i, total, q, cat)
        all_results.append(r)

    total_elapsed = time.perf_counter() - total_start

    # ── Tally scores ──────────────────────────────────────────────────────────
    method_counts  = {"CORRECT": 0, "HALLUCINATED": 0, "WRONG": 0, "TIMEOUT": 0, "ERROR": 0, "UNKNOWN": 0}
    judge_counts   = {"GROUNDED": 0, "UNGROUNDED": 0, "PARTIAL": 0, "SKIP": 0, "ERROR": 0, "UNKNOWN": 0}
    phrase_ratios  = []

    for r in all_results:
        v = r["method_verdict"]
        method_counts[v] = method_counts.get(v, 0) + 1
        j = r["judge_verdict"]
        judge_counts[j] = judge_counts.get(j, 0) + 1
        if r["phrase_match_ratio"] > 0:
            phrase_ratios.append(r["phrase_match_ratio"])

    avg_phrase = sum(phrase_ratios) / len(phrase_ratios) if phrase_ratios else 0.0
    judged     = [r for r in all_results if r["judge_verdict"] not in ("SKIP", "ERROR", "UNKNOWN")]
    grounded_pct = (judge_counts["GROUNDED"] / len(judged) * 100) if judged else 0.0

    # Reranker stats
    reranked    = [r for r in all_results if r.get("reranker_top_score") is not None]
    avg_rerank  = sum(r["reranker_top_score"] for r in reranked) / len(reranked) if reranked else 0.0
    avg_cands   = sum(r["candidate_count"] for r in all_results if r.get("candidate_count")) / max(1, len([r for r in all_results if r.get("candidate_count")]))

    # Save JSON
    json_data = {
        "run_timestamp": RUN_TIMESTAMP,
        "api_url": API_URL,
        "total_questions": total,
        "total_time_seconds": round(total_elapsed, 1),
        "method_counts": method_counts,
        "judge_counts": judge_counts,
        "method_accuracy_pct": round(method_counts["CORRECT"] / total * 100, 1),
        "grounded_pct": round(grounded_pct, 1),
        "avg_phrase_match": round(avg_phrase, 2),
        "reranker_questions": len(reranked),
        "avg_reranker_top_score": round(avg_rerank, 3),
        "avg_hybrid_candidates": round(avg_cands, 1),
        "results": all_results,
    }
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Print summary
    _log(f"\n{DIVIDER}")
    _log(f"  SUMMARY  —  total time: {total_elapsed:.1f}s")
    _log(DIVIDER)
    _log(f"\n  {'Q':<5} {'METHOD':<14} {'JUDGE':<12} {'PHRASES':>8}  {'TIME':>7}  QUESTION")
    _log(f"  {'-'*5} {'-'*14} {'-'*12} {'-'*8}  {'-'*7}  {'-'*30}")

    syms = {"CORRECT":"✓","HALLUCINATED":"✗","WRONG":"!","TIMEOUT":"T","ERROR":"E","UNKNOWN":"?"}
    jsyms = {"GROUNDED":"✓","UNGROUNDED":"✗","PARTIAL":"~","SKIP":"-","ERROR":"E","UNKNOWN":"?"}

    for i, r in enumerate(all_results):
        mv  = r["method_verdict"]
        jv  = r["judge_verdict"]
        ms  = syms.get(mv, "?")
        js  = jsyms.get(jv, "?")
        pr  = f"{r['phrase_match_ratio']:.0%}" if r["phrase_match_ratio"] > 0 else "  -"
        t   = f"{r['elapsed']:.1f}s"
        q_s = r["q"][:40] + ("..." if len(r["q"]) > 40 else "")
        _log(f"  [{i+1:>2}] {ms} {mv:<13} {js} {jv:<11} {pr:>8}  {t:>7}  {q_s}")

    _log(f"\n  {'─'*60}")
    _log(f"  METHOD CHECK  ✓ CORRECT: {method_counts['CORRECT']}/{total}  "
         f"✗ HALLUCINATED: {method_counts['HALLUCINATED']}  "
         f"! WRONG: {method_counts['WRONG']}")
    _log(f"  LLM JUDGE     ✓ GROUNDED: {judge_counts['GROUNDED']}  "
         f"✗ UNGROUNDED: {judge_counts['UNGROUNDED']}  "
         f"~ PARTIAL: {judge_counts['PARTIAL']}  "
         f"- SKIP: {judge_counts['SKIP']}")
    _log(f"  PHRASE MATCH  avg {avg_phrase:.0%} of answer phrases traceable to retrieved chunks")
    if reranked:
        _log(f"  RERANKER      {len(reranked)} questions reranked | avg top score: {avg_rerank:.3f} | avg candidates: {avg_cands:.1f}")
    _log(f"\n  Method accuracy : {method_counts['CORRECT']}/{total} = {method_counts['CORRECT']/total*100:.0f}%")
    _log(f"  Grounded answers: {judge_counts['GROUNDED']}/{len(judged)} judged = {grounded_pct:.0f}%")
    _log(f"  {'─'*60}\n")
    _log(f"  Log file : {LOG_FILE}")
    _log(f"  JSON file: {JSON_FILE}\n")
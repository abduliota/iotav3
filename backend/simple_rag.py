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
CACHE_BACKEND        = os.getenv("CACHE_BACKEND", "memory")
CACHE_SIM_THRESH     = float(os.getenv("CACHE_SIMILARITY_THRESH", "0.95"))
CACHE_TTL_SECONDS    = int(os.getenv("CACHE_TTL_SECONDS", "2592000"))
REDIS_URL            = os.getenv("REDIS_URL", "")

NOT_FOUND = (
    "The provided SAMA/regulatory documentation does not contain a clear answer "
    "to this question. Please consult sama.gov.sa or a qualified compliance officer."
)

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

# [FIX] Added negation/restriction awareness rule to SYSTEM_PROMPT.
# Root cause of the "who cannot create a bank account" failure:
# GPT-4o-mini was paraphrasing "shall not" restriction clauses as affirmative
# statements or treating them as irrelevant to the question.
# The new rule explicitly instructs the model to surface restrictive language
# accurately, including the Arabic equivalents (لا يجوز / يُحظر / غير مؤهل).
SYSTEM_PROMPT = """You are a strict regulatory assistant for Saudi Arabian banking regulations.

Answer using ONLY the text explicitly provided in <context>. Every sentence you write must be directly supported by a specific passage.

STRICT RULES:
- Write 2-3 natural sentences maximum. Stop immediately after 3 sentences.
- Add inline citations after each sentence: (Document Name, Page X)
- Use the exact document name and page number from the passage header.
- Do NOT add details, numbers, or conditions that are not explicitly written in the passages.
- Do NOT make inferences or reasonable assumptions — only state what the text says.
- If the user writes in Arabic, answer in Arabic using the same rules.
- Pay close attention to restrictive language: "shall not", "not permitted", "prohibited", "not eligible", "not allowed", "لا يجوز", "يُحظر", "غير مؤهل" — these indicate restrictions and must be reported accurately and completely.
- If the answer is not explicitly stated in the context, write ONLY: "The provided documentation does not contain a clear answer to this question."

FORBIDDEN:
- Do not use phrases like: generally speaking, typically, in most cases, overall, in summary, additionally.
- Do not invent organization names, regulation numbers, or SAR amounts not present in the context.
- Do not add a concluding sentence that generalizes beyond the context."""

OUT_OF_SCOPE_PATTERNS = [
    r"\bweather\b", r"\brecipe\b", r"\bsports\b", r"\bsong\b", r"\bmovie\b",
    r"who is the president\b", r"who is the prime minister\b", r"\bstock price\b",
    r"who is the ceo of", r"who is the founder of", r"who invented",
    r"\bnetflix\b", r"\bgoogle\b", r"\bamazon\b", r"\bmicrosoft\b",
    r"\bapple inc\b", r"\bfacebook\b", r"\btwitter\b",
]

def _is_out_of_scope(query: str) -> bool:
    q = query.strip().lower()
    return any(re.search(p, q) for p in OUT_OF_SCOPE_PATTERNS)

def _is_arabic(text: str) -> bool:
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return arabic_chars > len(text) * 0.3

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
    "edd":    "enhanced due diligence high risk customers",
    "hqla":   "high quality liquid assets liquidity coverage ratio",
    "ltv":    "loan to value ratio risk weight residential real estate",
    "retail": "retail customers individual natural persons resident bank account",

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

    # ── Bank account — multi-word keys (NEW) ──────────────────────────────────
    # Without these, a short query like "who cannot create a bank account" only
    # fires the "retail" single-word expansion which drifts toward general
    # account-opening procedures rather than restriction clauses.
    # These multi-word keys anchor retrieval to SAMA EN 1644 specifically.
    "bank account opening":          "bank account opening rules requirements procedures eligibility SAMA EN 1644",
    "open bank account":             "bank account opening requirements procedures eligibility SAMA EN 1644",
    "bank account rules":            "bank account opening rules regulations SAMA EN 1644 requirements",
    "open an account":               "bank account opening rules eligibility requirements procedures SAMA EN 1644",
    "create bank account":           "bank account opening eligibility requirements restrictions SAMA EN 1644",
    "create a bank account":         "bank account opening eligibility requirements restrictions prohibited SAMA EN 1644",

    # ── Restriction / negation queries (NEW) ──────────────────────────────────
    # Fix for "who cannot / not allowed / prohibited" questions.
    # The embedding model does not understand negation — "shall not open" and
    # "shall open" produce nearly identical vectors because the surrounding
    # regulatory language is identical. These expansions explicitly add
    # "shall not / prohibited / restrictions" to the query vector so it points
    # toward restriction clauses rather than procedure clauses.
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

    # ── Restriction queries — Arabic (NEW) ────────────────────────────────────
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
}


# [FIX] Case-insensitive substring match for long keys prevents silent misses
def _expand_query(query: str) -> str:
    q = query.strip()
    expansions = []
    q_lower = q.lower()
    for key, expansion in QUERY_EXPANSIONS.items():
        key_lower = key.lower()
        if len(key) <= 20 and re.search(rf"\b{re.escape(key_lower)}\b", q_lower):
            expansions.append(expansion)
        elif len(key) > 20 and key_lower in q_lower:
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
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading {EMBEDDING_MODEL}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print("[reranker] Loading BAAI/bge-reranker-base ...")
            _reranker = CrossEncoder("BAAI/bge-reranker-base")
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
                sim = float(np.dot(q, np.array(json.loads(raw))))
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

def fetch_chunks(query_vec: list[float], limit: int | None = None) -> list[dict]:
    resp = _get_supabase().rpc("match_chunks", {
        "query_embedding": query_vec,
        "match_threshold": SIMILARITY_THRESHOLD,
        "match_count":     limit or TOP_K,
    }).execute()
    return resp.data or []

def fetch_chunks_keyword(query: str, limit: int = 10) -> list[dict]:
    try:
        resp = _get_supabase().rpc("keyword_search_chunks", {
            "search_query": query,
            "match_count":  limit,
        }).execute()
        results = resp.data or []
        for r in results:
            if "similarity" not in r:
                r["similarity"] = 0.75
        return results
    except Exception as e:
        print(f"[hybrid] Keyword search unavailable: {e}")
        return []

def fetch_chunks_hybrid(query: str, query_vec: list[float], limit: int = 15) -> list[dict]:
    vector_results  = fetch_chunks(query_vec, limit=limit)
    keyword_results = fetch_chunks_keyword(query, limit=limit) if HYBRID_SEARCH else []
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

def _user_prompt(context_text: str, query: str, session_summary: str = "") -> str:
    summary_block = (
        f"<conversation_context>\n{session_summary}\n</conversation_context>\n\n"
        if session_summary else ""
    )
    if _is_arabic(query):
        instruction = "Answer in Arabic, max 2-3 sentences, include citations. If not found: لا تتوفر إجابة في الوثائق المقدمة"
    else:
        instruction = "Answer in natural sentences, max 3 sentences, include inline citations."
    return f"{summary_block}<context>\n{context_text}\n</context>\n\nQuestion: {query}\n\n{instruction}\n\nAnswer:"

_DRIFT_SIGNALS = [
    "in many countries", "it is important to note", "it should be noted",
    "generally speaking", "in general", "typically", "in most cases",
    "it is worth noting", "by adhering to", "this ensures that",
    "overall,", "in summary,", "in conclusion,", "furthermore, banks",
    "moreover, banks", "additionally, banks must", "international monetary fund",
    "world bank", "central bank of saudi arabia (cba)",
]

def _truncate_at_drift(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    for sent in sentences:
        if any(s in sent.lower() for s in _DRIFT_SIGNALS):
            break
        kept.append(sent)
        if len(kept) >= 3:
            break
    return " ".join(kept).strip() if kept else text

def _clean_output(text: str, query: str) -> str:
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
        text = _truncate_at_drift(text)
    return text

def _generate_qwen(ctx: str, query: str, on_chunk: Optional[Callable] = None,
                   session_summary: str = "") -> str:
    pipe = _get_qwen()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _user_prompt(ctx, query, session_summary)},
    ]
    full_input = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = pipe(
        full_input,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
        temperature=None,
        top_p=None,
    )
    answer = _clean_output(out[0]["generated_text"], query)
    if on_chunk: on_chunk(answer)
    return answer

def _generate_openai(ctx: str, query: str, on_chunk: Optional[Callable] = None,
                     session_summary: str = "") -> str:
    import openai
    stream = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": _user_prompt(ctx, query, session_summary)}],
        temperature=0.1, max_tokens=512, stream=True,
    )
    answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        answer += delta
        if on_chunk and delta: on_chunk(delta)
    return answer

def _generate_azure(ctx: str, query: str, on_chunk: Optional[Callable] = None,
                    session_summary: str = "") -> str:
    import openai
    stream = openai.AzureOpenAI(
        api_key=AZURE_OPENAI_KEY, azure_endpoint=AZURE_ENDPOINT, api_version="2024-02-01",
    ).chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": _user_prompt(ctx, query, session_summary)}],
        temperature=0.1, max_tokens=512, stream=True,
    )
    answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        answer += delta
        if on_chunk and delta: on_chunk(delta)
    return answer

def _generate(ctx: str, query: str, on_chunk: Optional[Callable] = None,
              session_summary: str = "") -> str:
    if LLM_BACKEND == "openai":
        return _generate_openai(ctx, query, on_chunk, session_summary)
    if LLM_BACKEND == "azure":
        return _generate_azure(ctx, query, on_chunk, session_summary)
    return _generate_qwen(ctx, query, on_chunk, session_summary)

def answer_query(
    user_query: str,
    top_k: int | None = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    debug: bool = False,
    session_summary: str = "",
    **kwargs,
) -> dict:
    if not user_query or not user_query.strip():
        return {"answer": "Please provide a question.", "sources": [], "cached": False, "method": "none"}

    query = user_query.strip()

    if _is_out_of_scope(query):
        answer = "This question is outside the scope of SAMA/banking regulatory documentation."
        if on_chunk: on_chunk(answer)
        return {"answer": answer, "sources": [], "cached": False, "method": "out_of_scope"}

    expanded  = _expand_query(query)
    query_vec = _embed(expanded)

    cached = _cache_lookup(query_vec)
    if cached:
        if on_chunk: on_chunk(cached["answer"])
        return {**cached, "cached": True, "method": "cached"}

    final_top_k = top_k or TOP_K
    candidates  = fetch_chunks_hybrid(expanded, query_vec, limit=RERANK_FETCH_K)

    if debug:
        print(f"\n[pipeline] {len(candidates)} hybrid candidates for: '{query}'")
        print(f"[pipeline] expanded: '{expanded[:120]}...'")
        for i, c in enumerate(candidates[:5]):
            print(f"  [{i+1}] sim={c.get('similarity',0):.4f} | {c.get('document_name','?')} p{c.get('page_start','?')}")

    if not candidates:
        if on_chunk: on_chunk(NOT_FOUND)
        return {"answer": NOT_FOUND, "sources": [], "cached": False, "method": "not_found"}

    chunks, reranker_top_score = rerank_chunks(query, candidates, top_n=final_top_k)
    top_sim = max(float(c.get("similarity", 0)) for c in candidates)

    if top_sim < LOW_CONF_THRESHOLD:
        if debug: print(f"[pipeline] low confidence ({top_sim:.4f} < {LOW_CONF_THRESHOLD})")
        if on_chunk: on_chunk(NOT_FOUND)
        return {"answer": NOT_FOUND, "sources": [], "cached": False, "method": "not_found"}

    answer = _generate(build_context(chunks), query, on_chunk, session_summary=session_summary)
    answer = _strip_trailing_not_found(answer)

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
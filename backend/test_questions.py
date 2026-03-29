"""
test_questions.py — Test the SAMA chatbot API

Scoring (3 independent KPIs — no single PASS/FAIL):
  KPI-1  Retrieval success   — did the system retrieve and attempt an answer?
                               SUCCESS = method in (generative, cached)
                               FAIL    = not_found / out_of_scope / timeout / error
  KPI-2  Grounding quality   — is the answer grounded in retrieved chunks?
                               GROUNDED   = 1.0  (PASS)
                               PARTIAL    = 0.5  (SOFT_PASS — counts as half a pass)
                               UNGROUNDED = 0.0  (FAIL)
                               SKIP       = excluded from grounding denominator
  KPI-3  Answer utility      — does the system route/behave correctly?
                               CORRECT / HALLUCINATED / WRONG (same as before)

Failure buckets (printed at end of every run):
  Bucket A — Generation failures
             retrieved OK (method=generative) but answer is not_found or UNGROUNDED
             → prompt issue or LLM refuses despite good chunks
  Bucket B — Grounding drift
             PARTIAL answers where top source similarity >= 0.79
             → answer adds claims beyond what chunks say; judge context mismatch
  Bucket C — Retrieval failures
             method=not_found OR top_sim < threshold OR out_of_scope when it shouldn't be
             → expansion gap, data coverage gap, or threshold too strict

Run while api.py is running:
    python test_questions.py

IMPORTANT — clear Redis cache before each benchmark run:
    POST http://localhost:8000/admin/cache/clear
    (stale cached answers will not reflect recent code/data changes)
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

# LOW_CONF_THRESHOLD mirrored here for bucket classification
# Keep in sync with .env / simple_rag.py
LOW_CONF_THRESHOLD = float(os.getenv("LOW_CONF_THRESHOLD", "0.72"))

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
    # ── Core SAMA knowledge ───────────────────────────────────────────────────
    ("What is SAMA?",                                                               "regulatory"),
    ("What is NORA?",                                                               "regulatory"),

    # ── Regulation content ────────────────────────────────────────────────────
    ("What are the minimum capital adequacy requirements for banks under SAMA?",     "regulatory"),
    ("Who cannot open a bank account in Saudi Arabia?",                             "regulatory"),
    ("What are the AML requirements under SAMA?",                                   "regulatory"),
    ("What are the know your customer requirements for retail customers?",           "regulatory"),
    ("What are the rules for opening bank accounts in Saudi Arabia?",               "regulatory"),
    ("What is the minimum capital requirement for a new bank license?",             "regulatory"),

    # ── Tricky / hallucination tests ──────────────────────────────────────────
    ("Did King Abdullah sign or write any SAMA documents?",                         "trick"),
    ("What are the KYC requirements for retail customers?",                         "regulatory"),

    # ── Out of scope ──────────────────────────────────────────────────────────
    ("What is the weather in Riyadh?",                                              "out_of_scope"),
    ("Who is the CEO of Apple?",                                                    "out_of_scope"),

    # ── Arabic ────────────────────────────────────────────────────────────────
    ("ما هو البنك المركزي السعودي؟",                                               "arabic"),
    ("ما هي متطلبات رأس المال للبنوك؟",                                            "arabic"),

    # ── Auto-generated from chunk content ─────────────────────────────────────
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

    # ── Additional English ────────────────────────────────────────────────────
    ("What is the scope of application of the Saudi Personal Data Protection Law?", "regulatory"),
    ("What penalties does the Saudi PDPL impose for violations?", "regulatory"),
    ("How does ISO 23200 address blockchain and distributed ledger technologies?", "regulatory"),
    ("What is the relationship between NCA and SAMA regarding cybersecurity frameworks?", "regulatory"),

    # ── Additional Arabic ─────────────────────────────────────────────────────
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

    # ── Aramco CCC (English) ──────────────────────────────────────────────────
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

    # ── Aramco CCC (Arabic) ───────────────────────────────────────────────────
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

    # ── NCA / ECC (Arabic) ────────────────────────────────────────────────────
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

    # ════════════════════════════════════════════════════════════════════════════
    # NEW QUESTIONS — added from SAMA Rulebook and NCA official documents
    # Indices 153–212
    # ════════════════════════════════════════════════════════════════════════════

    # ── SAMA Counter-Fraud Framework (English) ────────────────────────────────
    # Source: rulebook.sama.gov.sa/en/counter-fraud-framework + PDF Oct 2022 v1.0
    ("What are the four main domains of the SAMA Counter-Fraud Framework?",         "regulatory"),
    ("How does SAMA define fraud in the Counter-Fraud Framework?",                  "regulatory"),
    ("What is the minimum Counter-Fraud maturity level that Member Organisations must achieve under SAMA?", "regulatory"),
    ("How many maturity levels does the SAMA Counter-Fraud Maturity Model distinguish?", "regulatory"),
    ("What must a bank include in its Counter-Fraud Policy under the SAMA framework?", "regulatory"),
    ("What is the role of the Counter-Fraud Governance Committee (CFGC) under the SAMA Counter-Fraud Framework?", "regulatory"),
    ("How should Member Organisations handle a situation where a Counter-Fraud control requirement cannot be implemented?", "regulatory"),
    ("What training obligations do Member Organisations have for their Counter-Fraud department staff?", "regulatory"),
    ("How does SAMA use the Counter-Fraud Framework to assess compliance at Member Organisations?", "regulatory"),
    ("In what way must the SAMA Counter-Fraud Framework be used together with the SAMA Cybersecurity Framework?", "regulatory"),

    # ── SAMA Counter-Fraud Framework (Arabic) ────────────────────────────────
    ("ما هي المجالات الأربعة الرئيسية لإطار مكافحة الاحتيال الصادر عن ساما؟",    "arabic"),
    ("كيف يعرّف إطار مكافحة الاحتيال الصادر عن ساما مفهوم الاحتيال؟",           "arabic"),
    ("ما الحد الأدنى لمستوى النضج المطلوب من المنظمات الأعضاء في إطار مكافحة الاحتيال؟", "arabic"),
    ("ما التزامات المنظمات الأعضاء في التدريب على ضوابط مكافحة الاحتيال؟",       "arabic"),
    ("ما الذي يجب أن تتضمنه سياسة مكافحة الاحتيال لدى المنظمات الأعضاء وفق إطار ساما؟", "arabic"),

    # ── SAMA Cyber Resilience Fundamental Requirements — CRFR (English) ──────
    # Source: rulebook.sama.gov.sa/en/cyber-resilience-fundamental-requirements-crfr
    ("What is the purpose of the SAMA Cyber Resilience Fundamental Requirements (CRFR)?", "regulatory"),
    ("Which types of entities are in scope of the SAMA CRFR framework?",            "regulatory"),
    ("How does the CRFR differ from the SAMA Cyber Security Framework (CSF)?",      "regulatory"),
    ("What self-assessment obligations does the CRFR impose on regulated entities?", "regulatory"),
    ("What are the four main pillars of the SAMA CRFR framework?",                  "regulatory"),
    ("What Business Continuity and Disaster Recovery requirements does the SAMA CRFR impose?", "regulatory"),
    ("How does SAMA verify compliance with the Cyber Resilience Fundamental Requirements?", "regulatory"),

    # ── SAMA CRFR (Arabic) ────────────────────────────────────────────────────
    ("ما الغرض من متطلبات الصمود السيبراني الأساسية (CRFR) الصادرة عن ساما؟",     "arabic"),
    ("ما الجهات التي تندرج ضمن نطاق إطار CRFR الصادر عن البنك المركزي السعودي؟", "arabic"),
    ("ما التزامات التقييم الذاتي التي يفرضها إطار CRFR على الجهات المرخصة؟",      "arabic"),
    ("ما متطلبات استمرارية الأعمال والتعافي من الكوارث ضمن إطار CRFR؟",           "arabic"),

    # ── SAMA Cybersecurity Framework — Maturity & Strategy (English) ─────────
    # Source: rulebook.sama.gov.sa + SAMA CSF PDF v1.0 May 2017
    ("What maturity level must Member Organisations operate at as a minimum under the SAMA CSF?", "regulatory"),
    ("What does a Cyber Security Strategy under the SAMA CSF need to be aligned with?", "regulatory"),
    ("What does the SAMA CSF require regarding an annual internal audit of cybersecurity compliance?", "regulatory"),
    ("What are the four domains of the SAMA Cyber Security Framework?",             "regulatory"),
    ("What does the SAMA CSF require organisations to do when they conduct a gap assessment?", "regulatory"),

    # ── SAMA Cybersecurity Framework — Maturity & Strategy (Arabic) ──────────
    ("ما مستوى النضج الأدنى المطلوب من المنظمات الأعضاء وفق إطار الأمن السيبراني لساما؟", "arabic"),
    ("ما الذي يجب أن تكون استراتيجية الأمن السيبراني متوافقة معه وفق إطار SAMA CSF؟", "arabic"),
    ("ما المجالات الأربعة الرئيسية لإطار الأمن السيبراني الصادر عن ساما؟",        "arabic"),

    # ── SAMA Open Banking Framework (English) ─────────────────────────────────
    # Source: sama.gov.sa Open Banking Policy + second release Sep 2024
    ("What is the SAMA Open Banking Framework and what is its purpose?",            "regulatory"),
    ("What is a Payment Initiation Service (PIS) under the SAMA Open Banking Framework?", "regulatory"),
    ("When was the SAMA Open Banking Framework officially launched?",               "regulatory"),
    ("Who is permitted to use Open Banking APIs to access customer financial data under the SAMA framework?", "regulatory"),
    ("What are the key components of the SAMA Open Banking Framework?",             "regulatory"),
    ("How does the SAMA Open Banking Framework relate to Saudi Vision 2030?",       "regulatory"),

    # ── SAMA Open Banking Framework (Arabic) ──────────────────────────────────
    ("ما إطار الخدمات المصرفية المفتوحة الصادر عن ساما وما هدفه؟",                "arabic"),
    ("ما خدمة بدء الدفع (PIS) ضمن إطار الخدمات المصرفية المفتوحة؟",              "arabic"),
    ("من المخوّل باستخدام واجهات برمجة التطبيقات المفتوحة للوصول إلى البيانات المالية للعملاء؟", "arabic"),

    # ── SAMA Financial Sector Cyber Threat Intelligence (English) ────────────
    # Source: SAMA CTI Principles PDF March 2022 v1.0
    ("What document did SAMA issue to guide financial sector Cyber Threat Intelligence practices?", "regulatory"),
    ("What are the four types of CTI principles defined in the SAMA CTI document?", "regulatory"),
    ("What taxonomy does SAMA recommend for classifying threat actor Tactics, Techniques, and Procedures?", "regulatory"),
    ("Who is the SAMA Cyber Threat Intelligence Principles document intended for?", "regulatory"),

    # ── SAMA CTI (Arabic) ─────────────────────────────────────────────────────
    ("ما الوثيقة التي أصدرتها ساما لتوجيه ممارسات استخبارات التهديدات السيبرانية في القطاع المالي؟", "arabic"),
    ("ما الأنواع الأربعة لمبادئ استخبارات التهديدات السيبرانية المحددة في وثيقة ساما؟", "arabic"),

    # ── NCA ECC-2:2024 (English) ──────────────────────────────────────────────
    # Source: nca.gov.sa ECC-2:2024 PDF
    ("What is the ECC-2:2024 and how does it differ from ECC-1:2018?",             "regulatory"),
    ("What does ECC-2:2024 require regarding the establishment of a cybersecurity department?", "regulatory"),
    ("Under ECC-2:2024, what must a cybersecurity strategy include?",               "regulatory"),
    ("What does ECC-2:2024 require for email service protection?",                  "regulatory"),
    ("Under NCA ECC-2:2024, what network security controls are mandatory?",         "regulatory"),
    ("What does ECC-2:2024 require for application security and secure coding?",    "regulatory"),
    ("What are the ICS/OT-specific additional requirements under ECC-2:2024?",      "regulatory"),

    # ── NCA ECC-2:2024 (Arabic) ───────────────────────────────────────────────
    ("ما هي الضوابط الأساسية للأمن السيبراني ECC-2:2024 وكيف تختلف عن إصدار 2018؟", "arabic"),
    ("ما الذي تشترطه ECC-2:2024 بشأن إنشاء وحدة متخصصة للأمن السيبراني؟",        "arabic"),
    ("ما ضوابط أمن الشبكات الإلزامية بموجب ضوابط ECC-2:2024 الصادرة عن NCA؟",    "arabic"),

    # ── NCA Cloud Cybersecurity Controls CCC-2:2024 (English) ────────────────
    # Source: nca.gov.sa CCC-2:2024 PDF
    ("What is the NCA Cloud Cybersecurity Controls (CCC-2:2024) and what does it govern?", "regulatory"),
    ("Who must comply with the NCA CCC-2:2024?",                                   "regulatory"),
    ("How does the NCA CCC relate to the Essential Cybersecurity Controls (ECC)?", "regulatory"),
    ("What data localization requirements does NCA CCC-2:2024 introduce?",          "regulatory"),
    ("What are the multi-factor authentication requirements for cloud users under NCA CCC-2:2024?", "regulatory"),
    ("What isolation requirements does NCA CCC-2:2024 impose on Cloud Service Providers?", "regulatory"),

    # ── NCA CCC-2:2024 (Arabic) ───────────────────────────────────────────────
    ("ما ضوابط الأمن السيبراني للحوسبة السحابية CCC-2:2024 الصادرة عن الهيئة الوطنية؟", "arabic"),
    ("من الملزم بالامتثال لضوابط CCC-2:2024 الصادرة عن هيئة الأمن السيبراني؟",   "arabic"),
    ("ما متطلبات توطين البيانات التي أدخلتها ضوابط CCC-2:2024؟",                  "arabic"),

    # ── NCA Operational Technology Cybersecurity Controls OTCC-1:2022 (English)
    # Source: nca.gov.sa OTCC-1:2022 PDF
    ("What is the NCA OTCC-1:2022 and which organizations must comply with it?",   "regulatory"),
    ("How does the OTCC-1:2022 relate to the NCA Essential Cybersecurity Controls?", "regulatory"),
    ("What are the three criticality levels defined for facilities under OTCC-1:2022?", "regulatory"),
    ("What network segmentation requirements does OTCC-1:2022 impose on OT/ICS environments?", "regulatory"),
    ("What access control requirements does OTCC-1:2022 specify for OT/ICS systems?", "regulatory"),
    ("What logging and monitoring requirements does OTCC-1:2022 impose?",           "regulatory"),
    ("What third-party cybersecurity requirements does OTCC-1:2022 place on organizations?", "regulatory"),
    ("What are the four cybersecurity pillars addressed by the OTCC-1:2022 framework?", "regulatory"),

    # ── NCA OTCC-1:2022 (Arabic) ──────────────────────────────────────────────
    ("ما هي ضوابط الأمن السيبراني للتقنيات التشغيلية OTCC-1:2022 وعلى من تنطبق؟", "arabic"),
    ("كيف ترتبط ضوابط OTCC-1:2022 بالضوابط الأساسية للأمن السيبراني ECC؟",        "arabic"),
    ("ما متطلبات تجزئة الشبكة المفروضة على بيئات OT/ICS وفق OTCC-1:2022؟",       "arabic"),
    ("ما متطلبات التسجيل والمراقبة المفروضة بموجب ضوابط OTCC-1:2022؟",            "arabic"),
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
    1:  "correct",       # What is NORA?
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
    **{i: "correct" for i in range(14, 300)},
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

Important: paraphrasing is acceptable. But added specifics (numbers, percentages, names, conditions)
not in the snippets = UNGROUNDED or PARTIAL.

Reply with ONLY one of: GROUNDED / UNGROUNDED / PARTIAL
Then on the next line: one concise sentence explaining which specific claim is ungrounded (if any).

CONTEXT SNIPPETS:
{snippets}

ANSWER TO EVALUATE:
{answer}"""

def llm_judge(answer: str, sources: list[dict]) -> tuple[str, str]:
    if not answer or not sources:
        return "SKIP", "no answer or no sources to evaluate"
    if _is_not_found_answer(answer):
        return "SKIP", "answer is not-found — nothing to verify"

    snippets = "\n\n".join([
        f"[{s.get('document_name','?')} p{s.get('page_start','?')}]\n{s.get('snippet','')}"
        for s in sources[:5]
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
    if not answer or not sources or _is_not_found_answer(answer):
        return 0.0, []

    all_snippets = " ".join([
        (s.get("snippet") or "") + " " + (s.get("document_name") or "")
        for s in sources
    ]).lower()

    words = re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', answer)
    phrases = []
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i+3]).lower()
        if len(phrase) > 10:
            phrases.append(phrase)

    if not phrases:
        return 0.0, []

    matched = [p for p in phrases if p in all_snippets]
    matched = list(dict.fromkeys(matched))[:5]
    ratio = len(matched) / len(phrases) if phrases else 0.0
    return round(ratio, 2), matched


# ── KPI-3: Answer utility (method routing correctness) ───────────────────────

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
    # expected == "correct"
    if method == "out_of_scope":
        return "WRONG", "incorrectly rejected"
    if method == "not_found" or _is_not_found_answer(answer):
        return "WRONG", "returned not found but answer exists in DB"
    if method in ("generative", "cached"):
        return "CORRECT", f"returned answer via {method}"
    return "UNKNOWN", f"unexpected method={method}"


# ── KPI-1: Retrieval success ──────────────────────────────────────────────────

def _retrieval_success(method: str, answer: str, timed_out: bool, category: str) -> str:
    """
    SUCCESS = the system attempted to answer (method=generative/cached) AND
              for expected correct questions it didn't return not_found.
    EXPECTED_SKIP = out_of_scope or trick — retrieval not applicable.
    FAIL = method=not_found, or timed out, or errored.
    """
    expected = "correct"  # default
    if timed_out:
        return "FAIL"
    if method == "out_of_scope" and category == "out_of_scope":
        return "EXPECTED_SKIP"
    if method == "not_found" or _is_not_found_answer(answer):
        return "FAIL"
    if method in ("generative", "cached"):
        return "SUCCESS"
    return "FAIL"


# ── KPI-2: Grounding score (with SOFT_PASS for PARTIAL) ──────────────────────

GROUNDING_SCORE = {
    "GROUNDED":   1.0,   # PASS
    "PARTIAL":    0.5,   # SOFT_PASS — counts as half a pass
    "UNGROUNDED": 0.0,   # FAIL
    "SKIP":       None,  # excluded from denominator
    "ERROR":      None,
    "UNKNOWN":    None,
}

GROUNDING_LABEL = {
    "GROUNDED":   "PASS",
    "PARTIAL":    "SOFT_PASS (0.5)",
    "UNGROUNDED": "FAIL",
    "SKIP":       "SKIP",
    "ERROR":      "ERROR",
    "UNKNOWN":    "UNKNOWN",
}


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
    top_sim   = 0.0

    try:
        resp = requests.post(API_URL, json={"query": q, "debug": True}, timeout=TIMEOUT)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data    = resp.json()
        answer  = (data.get("answer") or "").strip()
        sources = data.get("sources") or []
        cached  = data.get("cached", False)
        method  = data.get("method", "cached" if cached else "unknown")
        candidate_count    = data.get("candidate_count", None)
        reranker_top_score = data.get("reranker_top_score", None)

        # Top similarity from sources (used for bucket classification)
        if sources:
            top_sim = max(float(s.get("similarity", 0)) for s in sources)

        # ── KPI-3: Answer utility ─────────────────────────────────────────────
        method_verdict, method_reason = _score_method(idx, method, answer, False)

        # ── KPI-1: Retrieval success ──────────────────────────────────────────
        ret_success = _retrieval_success(method, answer, False, category)

        # ── KPI-2: Grounding quality ──────────────────────────────────────────
        if method in ("generative", "cached") and sources and not _is_not_found_answer(answer):
            _log(f"\n  [Verifying answer with LLM judge...]")
            judge_verdict, judge_reason = llm_judge(answer, sources)
            phrase_ratio, matched_phrases = verify_phrases_in_sources(answer, sources)
        else:
            judge_verdict, judge_reason = "SKIP", "not applicable"
            phrase_ratio, matched_phrases = 0.0, []

        grounding_score = GROUNDING_SCORE.get(judge_verdict, None)
        grounding_label = GROUNDING_LABEL.get(judge_verdict, "UNKNOWN")

        label = METHOD_LABELS.get(method, method)
        _log(f"\n  ── KPI-1  RETRIEVAL   : {ret_success}")
        _log(f"  ── KPI-2  GROUNDING   : {judge_verdict}  →  {grounding_label}")
        _log(f"  ── KPI-3  UTILITY     : {method_verdict}  ({method_reason})")
        _log(f"\n  METHOD       : {label}")
        _log(f"  TIME         : {elapsed:.2f}s")
        if candidate_count is not None:
            _log(f"  CANDIDATES   : {candidate_count}")
        if reranker_top_score is not None:
            _log(f"  RERANKER     : top score = {reranker_top_score:.3f}")
        if sources:
            _log(f"  TOP SIM      : {top_sim:.4f}")
        _log(f"  JUDGE REASON : {judge_reason}")
        _log(f"  PHRASE MATCH : {phrase_ratio:.0%}")
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
            # KPI-1
            "retrieval_success": ret_success,
            # KPI-2
            "judge_verdict": judge_verdict,
            "judge_reason": judge_reason,
            "grounding_score": grounding_score,
            "grounding_label": grounding_label,
            # KPI-3
            "method_verdict": method_verdict,
            "method_reason": method_reason,
            # Supporting data
            "phrase_match_ratio": phrase_ratio,
            "matched_phrases": matched_phrases,
            "answer": answer,
            "top_sim": top_sim,
            "candidate_count": candidate_count,
            "reranker_top_score": reranker_top_score,
            "sources": [{"doc": s.get("document_name"), "sim": s.get("similarity"), "snippet": s.get("snippet","")} for s in sources],
            "timed_out": False,
        }

    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start
        _log(f"\n  [TIMEOUT] No response after {elapsed:.0f}s.")
        return {
            "q": q, "category": category, "method": "timeout",
            "elapsed": round(elapsed, 2),
            "retrieval_success": "FAIL",
            "judge_verdict": "SKIP", "judge_reason": "timeout",
            "grounding_score": None, "grounding_label": "SKIP",
            "method_verdict": "TIMEOUT", "method_reason": "no response within timeout",
            "phrase_match_ratio": 0.0, "matched_phrases": [],
            "answer": "", "top_sim": 0.0, "sources": [], "timed_out": True,
            "candidate_count": None, "reranker_top_score": None,
        }

    except requests.exceptions.ConnectionError:
        _log("\n  [CONNECTION ERROR] api.py not running?")
        return {
            "q": q, "category": category, "method": "error",
            "elapsed": 0,
            "retrieval_success": "FAIL",
            "judge_verdict": "SKIP", "judge_reason": "connection error",
            "grounding_score": None, "grounding_label": "SKIP",
            "method_verdict": "ERROR", "method_reason": "connection refused",
            "phrase_match_ratio": 0.0, "matched_phrases": [],
            "answer": "", "top_sim": 0.0, "sources": [], "timed_out": False,
            "candidate_count": None, "reranker_top_score": None,
        }

    except Exception as e:
        elapsed = time.perf_counter() - start
        _log(f"\n  [ERROR after {elapsed:.2f}s] {e}")
        return {
            "q": q, "category": category, "method": "error",
            "elapsed": round(elapsed, 2),
            "retrieval_success": "FAIL",
            "judge_verdict": "SKIP", "judge_reason": str(e),
            "grounding_score": None, "grounding_label": "SKIP",
            "method_verdict": "ERROR", "method_reason": str(e),
            "phrase_match_ratio": 0.0, "matched_phrases": [],
            "answer": "", "top_sim": 0.0, "sources": [], "timed_out": False,
            "candidate_count": None, "reranker_top_score": None,
        }


# ── Bucket classifier ─────────────────────────────────────────────────────────

def _classify_bucket(r: dict) -> str | None:
    """
    Bucket A — Generation failure:
        Retrieved OK (method=generative) but answer is not_found OR judge=UNGROUNDED.
        → LLM refuses despite good chunks, or hallucinates then gets caught.
    Bucket B — Grounding drift:
        judge=PARTIAL and top_sim >= LOW_CONF_THRESHOLD.
        → Answer adds claims beyond what chunks say.
    Bucket C — Retrieval failure:
        method=not_found, or top_sim < LOW_CONF_THRESHOLD, or wrong out_of_scope routing.
        → Expansion gap, data coverage gap, or threshold too strict.
    Returns None if the question passed cleanly (CORRECT + GROUNDED).
    """
    method  = r.get("method", "")
    verdict = r.get("judge_verdict", "")
    top_sim = r.get("top_sim", 0.0) or 0.0
    mv      = r.get("method_verdict", "")
    answer  = r.get("answer", "")
    cat     = r.get("category", "")

    # Clean pass — no bucket
    if mv == "CORRECT" and verdict in ("GROUNDED", "SKIP"):
        return None

    # Expected failures — no bucket (out_of_scope / trick / not_found correctly handled)
    if mv == "CORRECT":
        return None

    # Bucket A: retrieved but generation failed
    if method == "generative" and (
        _is_not_found_answer(answer) or verdict == "UNGROUNDED"
    ):
        return "A"

    # Bucket B: partial grounding with good retrieval
    if verdict == "PARTIAL" and top_sim >= LOW_CONF_THRESHOLD:
        return "B"

    # Bucket C: retrieval failed
    if method in ("not_found", "timeout", "error", "unknown"):
        return "C"
    if method == "out_of_scope" and cat not in ("out_of_scope",):
        return "C"
    if top_sim < LOW_CONF_THRESHOLD and top_sim > 0:
        return "C"

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total       = len(QUESTIONS)
    total_start = time.perf_counter()

    # ── Cache-clear reminder ──────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  ⚠  BEFORE RUNNING: clear Redis cache to avoid stale answers")
    print("     POST http://localhost:8000/admin/cache/clear")
    print(f"{'═'*60}\n")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"SAMA Chatbot Test Run — 3-KPI Scoring\n")
        f.write(f"Timestamp  : {RUN_TIMESTAMP}\n")
        f.write(f"API URL    : {API_URL}\n")
        f.write(f"Questions  : {total}\n")
        f.write(f"KPI-1  Retrieval success  (SUCCESS / FAIL / EXPECTED_SKIP)\n")
        f.write(f"KPI-2  Grounding quality  (GROUNDED=1.0 / PARTIAL=0.5 / UNGROUNDED=0.0)\n")
        f.write(f"KPI-3  Answer utility     (CORRECT / HALLUCINATED / WRONG)\n")
        f.write(f"Buckets: A=generation fail / B=grounding drift / C=retrieval fail\n")
        f.write("=" * 80 + "\n\n")

    print(f"Testing {total} questions against {API_URL}")
    print(f"Timeout per question: {TIMEOUT}s")
    print(f"Logs saving to: {LOG_FILE}\n")

    all_results = []
    for i, (q, cat) in enumerate(QUESTIONS):
        r = test_question(i, total, q, cat)
        all_results.append(r)

    total_elapsed = time.perf_counter() - total_start

    # ── KPI-1: Retrieval tallies ──────────────────────────────────────────────
    ret_success  = sum(1 for r in all_results if r["retrieval_success"] == "SUCCESS")
    ret_fail     = sum(1 for r in all_results if r["retrieval_success"] == "FAIL")
    ret_skip     = sum(1 for r in all_results if r["retrieval_success"] == "EXPECTED_SKIP")
    ret_eligible = total - ret_skip   # questions where retrieval was expected

    # ── KPI-2: Grounding tallies ──────────────────────────────────────────────
    judge_counts = {"GROUNDED": 0, "PARTIAL": 0, "UNGROUNDED": 0, "SKIP": 0, "ERROR": 0, "UNKNOWN": 0}
    grounding_score_sum   = 0.0
    grounding_denominator = 0
    for r in all_results:
        jv = r["judge_verdict"]
        judge_counts[jv] = judge_counts.get(jv, 0) + 1
        score = r["grounding_score"]
        if score is not None:
            grounding_score_sum   += score
            grounding_denominator += 1

    weighted_grounding_pct = (grounding_score_sum / grounding_denominator * 100) if grounding_denominator > 0 else 0.0
    strict_grounded_pct    = (judge_counts["GROUNDED"] / grounding_denominator * 100) if grounding_denominator > 0 else 0.0

    # ── KPI-3: Utility tallies ────────────────────────────────────────────────
    method_counts = {"CORRECT": 0, "HALLUCINATED": 0, "WRONG": 0, "TIMEOUT": 0, "ERROR": 0, "UNKNOWN": 0}
    for r in all_results:
        v = r["method_verdict"]
        method_counts[v] = method_counts.get(v, 0) + 1

    # ── Supporting stats ──────────────────────────────────────────────────────
    phrase_ratios  = [r["phrase_match_ratio"] for r in all_results if r["phrase_match_ratio"] > 0]
    avg_phrase     = sum(phrase_ratios) / len(phrase_ratios) if phrase_ratios else 0.0

    sims           = [r["top_sim"] for r in all_results if r.get("top_sim", 0) > 0]
    avg_sim        = sum(sims) / len(sims) if sims else 0.0

    cands          = [r["candidate_count"] for r in all_results if r.get("candidate_count")]
    avg_cands      = sum(cands) / len(cands) if cands else 0.0

    reranked       = [r for r in all_results if r.get("reranker_top_score") is not None]
    avg_rerank     = sum(r["reranker_top_score"] for r in reranked) / len(reranked) if reranked else 0.0

    # ── Failure bucket report ─────────────────────────────────────────────────
    bucket_a, bucket_b, bucket_c = [], [], []
    for r in all_results:
        b = _classify_bucket(r)
        if b == "A": bucket_a.append(r)
        elif b == "B": bucket_b.append(r)
        elif b == "C": bucket_c.append(r)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_data = {
        "run_timestamp": RUN_TIMESTAMP,
        "api_url": API_URL,
        "total_questions": total,
        "total_time_seconds": round(total_elapsed, 1),
        # KPI-1
        "kpi1_retrieval": {
            "success": ret_success,
            "fail": ret_fail,
            "expected_skip": ret_skip,
            "eligible": ret_eligible,
            "success_rate_pct": round(ret_success / ret_eligible * 100, 1) if ret_eligible else 0,
        },
        # KPI-2
        "kpi2_grounding": {
            "grounded": judge_counts["GROUNDED"],
            "partial":  judge_counts["PARTIAL"],
            "ungrounded": judge_counts["UNGROUNDED"],
            "skip": judge_counts["SKIP"],
            "judged": grounding_denominator,
            "weighted_score_pct": round(weighted_grounding_pct, 1),
            "strict_grounded_pct": round(strict_grounded_pct, 1),
            "note": "weighted = GROUNDED*1.0 + PARTIAL*0.5 / judged",
        },
        # KPI-3
        "kpi3_utility": {
            "correct": method_counts["CORRECT"],
            "hallucinated": method_counts["HALLUCINATED"],
            "wrong": method_counts["WRONG"],
            "timeout": method_counts["TIMEOUT"],
            "error": method_counts["ERROR"],
            "accuracy_pct": round(method_counts["CORRECT"] / total * 100, 1),
        },
        # Supporting
        "avg_phrase_match": round(avg_phrase, 2),
        "avg_top_similarity": round(avg_sim, 4),
        "avg_hybrid_candidates": round(avg_cands, 1),
        "avg_reranker_top_score": round(avg_rerank, 3),
        # Buckets
        "bucket_A_count": len(bucket_a),
        "bucket_B_count": len(bucket_b),
        "bucket_C_count": len(bucket_c),
        "results": all_results,
    }
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # ── Per-question summary table ────────────────────────────────────────────
    _log(f"\n{DIVIDER}")
    _log(f"  RESULTS TABLE  —  total time: {total_elapsed:.1f}s")
    _log(DIVIDER)
    _log(f"\n  {'Q':<5} {'RET':>4} {'GROUND':>10} {'UTIL':<14} {'SIM':>6}  {'TIME':>6}  QUESTION")
    _log(f"  {'-'*5} {'-'*4} {'-'*10} {'-'*14} {'-'*6}  {'-'*6}  {'-'*35}")

    ret_syms  = {"SUCCESS": "✓", "FAIL": "✗", "EXPECTED_SKIP": "-"}
    util_syms = {"CORRECT": "✓", "HALLUCINATED": "✗", "WRONG": "!", "TIMEOUT": "T", "ERROR": "E", "UNKNOWN": "?"}
    g_syms    = {"GROUNDED": "✓", "PARTIAL": "~", "UNGROUNDED": "✗", "SKIP": "-", "ERROR": "E", "UNKNOWN": "?"}

    for i, r in enumerate(all_results):
        rs  = ret_syms.get(r["retrieval_success"], "?")
        gs  = g_syms.get(r["judge_verdict"], "?")
        gl  = r["judge_verdict"]
        us  = util_syms.get(r["method_verdict"], "?")
        uv  = r["method_verdict"]
        sim = f"{r['top_sim']:.3f}" if r.get("top_sim", 0) > 0 else "  -  "
        t   = f"{r['elapsed']:.1f}s"
        q_s = r["q"][:38] + ("…" if len(r["q"]) > 38 else "")
        bkt = _classify_bucket(r)
        bkt_str = f" [B{bkt}]" if bkt else ""
        _log(f"  [{i+1:>3}] {rs} {gs} {gl:<10} {us} {uv:<13} {sim}  {t:>6}  {q_s}{bkt_str}")

    # ── KPI summary ───────────────────────────────────────────────────────────
    _log(f"\n{'═'*80}")
    _log(f"  3-KPI SUMMARY")
    _log(f"{'═'*80}")

    _log(f"\n  KPI-1  RETRIEVAL SUCCESS")
    _log(f"         ✓ SUCCESS : {ret_success}/{ret_eligible} eligible = {ret_success/ret_eligible*100:.0f}%" if ret_eligible else "         no eligible questions")
    _log(f"         ✗ FAIL    : {ret_fail}   (not_found / timeout / error)")
    _log(f"         - SKIP    : {ret_skip}  (out_of_scope / trick — expected non-answer)")

    _log(f"\n  KPI-2  GROUNDING QUALITY  (judged: {grounding_denominator})")
    _log(f"         ✓ GROUNDED   : {judge_counts['GROUNDED']}  (PASS = 1.0)")
    _log(f"         ~ PARTIAL    : {judge_counts['PARTIAL']}  (SOFT_PASS = 0.5)")
    _log(f"         ✗ UNGROUNDED : {judge_counts['UNGROUNDED']}  (FAIL = 0.0)")
    _log(f"         - SKIP       : {judge_counts['SKIP']}")
    _log(f"         Weighted score : {weighted_grounding_pct:.1f}%  (GROUNDED*1 + PARTIAL*0.5 / judged)")
    _log(f"         Strict score   : {strict_grounded_pct:.1f}%  (GROUNDED only)")

    _log(f"\n  KPI-3  ANSWER UTILITY")
    _log(f"         ✓ CORRECT      : {method_counts['CORRECT']}/{total} = {method_counts['CORRECT']/total*100:.0f}%")
    _log(f"         ✗ HALLUCINATED : {method_counts['HALLUCINATED']}")
    _log(f"         ! WRONG        : {method_counts['WRONG']}")

    _log(f"\n  SUPPORTING STATS")
    _log(f"         Avg top similarity   : {avg_sim:.4f}")
    _log(f"         Avg hybrid candidates: {avg_cands:.1f}")
    _log(f"         Avg phrase match     : {avg_phrase:.0%}")
    if reranked:
        _log(f"         Avg reranker score   : {avg_rerank:.3f}  ({len(reranked)} questions reranked)")

    # ── Failure bucket report ─────────────────────────────────────────────────
    _log(f"\n{'═'*80}")
    _log(f"  FAILURE BUCKET REPORT")
    _log(f"{'═'*80}")

    _log(f"\n  Bucket A — Generation failures  ({len(bucket_a)} questions)")
    _log(f"  Retrieved OK but LLM said not_found or answer was UNGROUNDED.")
    _log(f"  Fix: tune SYSTEM_PROMPT, raise TOP_K, or check chunk fragmentation.")
    if bucket_a:
        for r in bucket_a:
            _log(f"    [{r['judge_verdict']:10}] sim={r.get('top_sim',0):.3f}  {r['q'][:65]}")
    else:
        _log(f"    (none)")

    _log(f"\n  Bucket B — Grounding drift  ({len(bucket_b)} questions)")
    _log(f"  PARTIAL answers where retrieval was good (sim >= {LOW_CONF_THRESHOLD}).")
    _log(f"  Fix: tighten SYSTEM_PROMPT, reduce max_tokens, check snippet length for judge.")
    if bucket_b:
        for r in bucket_b:
            _log(f"    [{r['judge_verdict']:10}] sim={r.get('top_sim',0):.3f}  {r['q'][:65]}")
            if r.get("judge_reason"):
                _log(f"      reason: {r['judge_reason']}")
    else:
        _log(f"    (none)")

    _log(f"\n  Bucket C — Retrieval failures  ({len(bucket_c)} questions)")
    _log(f"  method=not_found, low similarity, or incorrectly out_of_scoped.")
    _log(f"  Fix: add query expansion keys, ingest missing docs, or lower LOW_CONF_THRESHOLD.")
    if bucket_c:
        for r in bucket_c:
            _log(f"    [{r['method']:12}] sim={r.get('top_sim',0):.3f}  {r['q'][:65]}")
    else:
        _log(f"    (none)")

    _log(f"\n{'═'*80}")
    _log(f"  % not_found   : {sum(1 for r in all_results if r['method']=='not_found')/total*100:.0f}%  ({sum(1 for r in all_results if r['method']=='not_found')}/{total})")
    _log(f"  % partial     : {judge_counts['PARTIAL']/max(grounding_denominator,1)*100:.0f}%  ({judge_counts['PARTIAL']}/{grounding_denominator} judged)")
    _log(f"  % ungrounded  : {judge_counts['UNGROUNDED']/max(grounding_denominator,1)*100:.0f}%  ({judge_counts['UNGROUNDED']}/{grounding_denominator} judged)")
    _log(f"{'═'*80}")
    _log(f"\n  Log  : {LOG_FILE}")
    _log(f"  JSON : {JSON_FILE}\n")
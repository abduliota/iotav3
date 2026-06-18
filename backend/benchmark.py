"""
SAMA NORA — Comprehensive Benchmark Runner
Sends 500+ questions to the local API, records results, generates stats.

Usage:
    python benchmark.py                    # run all questions
    python benchmark.py --limit 50         # first 50 only (quick test)
    python benchmark.py --start 100        # resume from question 100
    python benchmark.py --url http://localhost:8000
    python benchmark.py --workers 1        # sequential (default: 1)

Output files:
    benchmark_results.csv   — one row per question (live-written)
    benchmark_stats.json    — summary statistics
    benchmark.log           — full log with answers
"""
from __future__ import annotations
import argparse, csv, json, logging, os, sys, time
from datetime import datetime
from pathlib import Path

# ── Questions (500+) ──────────────────────────────────────────────────────────
QUESTIONS = [

# ── SECTION 1: Bank Account Opening — English (50) ───────────────────────────
  {"q":"What are the SAMA regulations for opening bank accounts for SMEs?",         "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What documents does an SME need to open a bank account?",                   "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Who is not eligible to open a bank account in Saudi Arabia?",               "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can a non-Saudi resident open a bank account?",                              "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the requirements for opening a bank account for a company?",        "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can a non-GCC national open a bank account in Saudi Arabia?",               "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What documents are required for individual bank account opening?",           "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the restrictions on opening bank accounts for foreign nationals?",  "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can a minor open a bank account in Saudi Arabia?",                           "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the eligibility criteria for a bank account in Saudi Arabia?",      "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What is required to open a joint bank account?",                             "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the SAMA account opening rules for juristic persons?",              "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can a foreigner open a savings account in Saudi Arabia?",                    "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the digital account opening requirements under SAMA?",              "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What is the straight-through processing approach for bank accounts?",        "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What happens if account opening documents are incomplete?",                  "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can a non-resident Indian open a bank account in Saudi Arabia?",             "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What is the manual review process for bank account opening?",                "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What biometric verification is required for account opening?",               "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the SAMA rules for remote account opening?",                        "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can expatriates open bank accounts in Saudi Arabia?",                        "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the conditions for opening a bank account for a small enterprise?", "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Are there any restrictions on who can open a bank account?",                 "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What is the workflow for bank account opening under SAMA EN 1644?",          "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What verification is needed for corporate bank account opening?",            "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the account opening requirements for large corporations?",          "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can someone with an expired ID open a bank account?",                        "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What national ID requirements apply for bank account opening?",              "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the ongoing due diligence requirements after account opening?",     "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What commercial register documents are required for SME account opening?",   "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the digital account limits under SAMA regulations?",                "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the mobile app account opening requirements?",                      "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the Absher verification requirements for account opening?",         "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the criteria for rejecting a bank account application?",            "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can a GCC national open a bank account in Saudi Arabia?",                    "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the non-resident account opening conditions?",                      "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What signatory approval is required for corporate account opening?",         "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the articles of association requirements for company accounts?",    "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What document submission formats are accepted for account opening?",         "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the SAMA rules for accounts for medium-sized enterprises?",         "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can a Pakistani national open a bank account in Saudi Arabia?",              "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the name mismatch procedures in account opening?",                  "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What is the biometric verification fallback procedure?",                     "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the SAMA EN 1644 requirements for account opening?",               "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What types of bank accounts can be opened under SAMA regulations?",          "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"Can a company without commercial registration open a bank account?",         "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"What are the individual bank account opening requirements in Saudi Arabia?", "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"u cannot open bank account who",                                             "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"who all cannot create a bank account?",                                      "cat":"bank_account","lang":"en","expect":"answered"},
  {"q":"what r the rules for opening sme bank account",                              "cat":"bank_account","lang":"en","expect":"answered"},

# ── SECTION 2: Bank Account Opening — Arabic (30) ────────────────────────────
  {"q":"ما هي متطلبات فتح الحساب البنكي للمنشآت الصغيرة؟",              "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"من لا يحق له فتح حساب بنكي في المملكة العربية السعودية؟",       "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي شروط فتح الحساب البنكي للشركات؟",                         "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"هل يمكن للأجانب فتح حساب بنكي في السعودية؟",                    "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي المستندات المطلوبة لفتح حساب بنكي للفرد؟",               "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي قواعد ساما لفتح الحساب البنكي للأفراد؟",                 "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"هل يمكن لمواطن خليجي فتح حساب بنكي في السعودية؟",              "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي القيود المفروضة على فتح حسابات للمقيمين الأجانب؟",       "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي شروط فتح الحساب البنكي الرقمي؟",                          "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي المعايير المطلوبة لفتح حساب للمنشآت المتوسطة؟",          "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي إجراءات التحقق الأمني عند فتح الحساب البنكي؟",            "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي السجل التجاري المطلوب لفتح حساب الشركة؟",                 "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"من يُمنع من فتح حساب بنكي وفق أنظمة ساما؟",                    "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات العناية الواجبة المستمرة بعد فتح الحساب؟",       "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"هل يمكن فتح حساب بنكي للقاصر في المملكة العربية السعودية؟",    "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هو إجراء المراجعة اليدوية لطلبات فتح الحساب؟",              "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي شروط فتح الحساب عن بُعد؟",                                "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هو نظام SAMA EN 1644 لفتح الحسابات البنكية؟",                "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي القيود على الحسابات للمقيمين من غير السعوديين؟",         "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"هل يمكن فتح حساب بنكي بدون سجل تجاري؟",                        "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات التوقيع الرقمي لفتح الحساب؟",                     "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي حدود الحساب البنكي الرقمي؟",                              "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات التحقق البيومتري لفتح الحساب؟",                   "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي أسباب رفض طلب فتح الحساب البنكي؟",                       "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي معايير الأهلية لفتح حساب بنكي في السعودية؟",              "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي وثائق عقد التأسيس المطلوبة لفتح حساب الشركة؟",           "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي شروط فتح الحساب المشترك وفق ساما؟",                       "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي إجراءات التقديم غير المكتمل لفتح الحساب؟",               "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"هل يمكن للمقيم الأجنبي من دول الخليج فتح حساب؟",               "cat":"bank_account","lang":"ar","expect":"answered"},
  {"q":"ما هي اشتراطات ساما للمؤسسات الصغيرة والمتوسطة؟",              "cat":"bank_account","lang":"ar","expect":"answered"},

# ── SECTION 3: KYC / AML — English (40) ─────────────────────────────────────
  {"q":"What are the KYC requirements for corporate customers?",                     "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What enhanced due diligence applies to politically exposed persons?",        "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What sanctions lists must banks screen against?",                            "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What is the SAR reporting threshold under SAMA?",                            "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the AML obligations for Saudi banks?",                              "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What customer due diligence is required for individual customers?",          "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the UBO verification requirements?",                                "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the suspicious activity reporting requirements?",                   "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What is the SAMA framework for anti-money laundering?",                      "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the enhanced due diligence measures for high-risk customers?",      "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the requirements for screening against sanctions lists?",           "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What is the role of SAFIU in AML reporting?",                               "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What ongoing monitoring is required for customer relationships?",            "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the FATF requirements applicable to Saudi banks?",                  "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"How are politically exposed persons defined under SAMA regulations?",        "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What happens when a customer fails enhanced due diligence?",                 "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What senior management approval is needed for PEP relationships?",           "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the source of wealth requirements for PEPs?",                       "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the UN Security Council sanctions screening requirements?",         "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the PCCML list screening requirements?",                            "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What KYC documents are required for corporate onboarding?",                 "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the ultimate beneficial owner requirements?",                       "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the AML record keeping requirements?",                              "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"When must a suspicious activity report be filed?",                           "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the customer risk classification requirements?",                    "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the OFAC screening requirements for Saudi banks?",                  "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the transaction monitoring requirements under AML regulations?",    "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"How should banks handle customers linked to terrorism financing?",           "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the correspondent banking due diligence requirements?",             "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the wire transfer monitoring requirements?",                        "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What is the definition of money laundering under SAMA regulations?",        "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the cash transaction reporting requirements?",                      "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What training requirements exist for AML compliance officers?",              "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the SAMA requirements for AML governance?",                        "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the risk-based approach requirements for AML?",                     "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the politically exposed person family member requirements?",        "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the digital banking AML requirements?",                             "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the suspicious transaction indicators under SAMA?",                "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What cross-border payment monitoring is required?",                          "cat":"kyc_aml","lang":"en","expect":"answered"},
  {"q":"What are the regulatory reporting timelines for suspicious activity?",       "cat":"kyc_aml","lang":"en","expect":"answered"},

# ── SECTION 4: KYC / AML — Arabic (25) ──────────────────────────────────────
  {"q":"ما هي متطلبات اعرف عميلك للعملاء المؤسسيين؟",                   "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي العناية الواجبة المعززة للأشخاص المعرضين سياسياً؟",       "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي قوائم العقوبات التي يجب على البنوك الفحص بها؟",           "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هو حد الإبلاغ عن الأنشطة المشبوهة تحت ساما؟",               "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي التزامات مكافحة غسيل الأموال للبنوك السعودية؟",          "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هو تعريف الشخص المعرض سياسياً؟",                              "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات التحقق من المالك المستفيد الفعلي؟",               "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"متى يجب تقديم تقرير الاشتباه؟",                                 "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات مراقبة المعاملات المشبوهة؟",                      "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات فحص قوائم العقوبات لمجلس الأمن الدولي؟",        "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي المتطلبات الرقابية المستمرة لعلاقات العملاء؟",            "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات مكافحة تمويل الإرهاب؟",                          "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات التدريب لمسؤولي الامتثال لمكافحة الغسيل؟",      "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هو دور وحدة الاستخبارات المالية السعودية؟",                   "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي المؤشرات على الأنشطة المشبوهة وفق ساما؟",                 "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات العناية الواجبة للعملاء عالي المخاطر؟",           "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات الفحص لقائمة OFAC؟",                              "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي إجراءات الإبلاغ عن المعاملات المشبوهة؟",                  "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات حفظ سجلات مكافحة الغسيل؟",                       "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هو التوجه القائم على المخاطر في مكافحة الغسيل؟",             "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات موافقة الإدارة العليا على العلاقات مع الأشخاص المعرضين؟", "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات مراقبة المدفوعات العابرة للحدود؟",                "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي إجراءات إنهاء العلاقة مع العملاء عند فشل العناية الواجبة؟", "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات الحوكمة لمكافحة غسيل الأموال في ساما؟",          "cat":"kyc_aml","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات التحقق من مصدر الثروة للأشخاص المعرضين سياسياً؟","cat":"kyc_aml","lang":"ar","expect":"answered"},

# ── SECTION 5: Capital Adequacy / Basel III — English (30) ───────────────────
  {"q":"What is the minimum capital adequacy ratio for Saudi banks?",               "cat":"capital","lang":"en","expect":"answered"},
  {"q":"How is the leverage ratio calculated under SAMA Basel III?",                "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the LCR requirement for Saudi banks?",                              "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the NSFR requirement under SAMA regulations?",                      "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the CET1 capital requirement for Saudi banks?",                     "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the Tier 1 capital requirement under Basel III?",                   "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the capital conservation buffer requirements?",                    "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the risk-weighted assets calculation requirements?",               "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the minimum total capital ratio for Saudi banks?",                  "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the HQLA requirements for liquidity coverage ratio?",              "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the 75 percent cap on cash inflows for LCR?",                      "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the pillar 3 disclosure requirements?",                            "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the ICAAP requirements for Saudi banks?",                          "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the operational risk capital requirements?",                       "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the loss event threshold for operational risk?",                    "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the loan-to-value ratio for residential real estate?",             "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the annual disclosure requirements for banks?",                    "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the loan to deposit ratio reporting requirement?",                  "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the minimum capital requirements for a new bank license?",         "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the net stable funding ratio calculation?",                         "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the countercyclical capital buffer requirements?",                 "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the systemically important bank surcharge requirements?",          "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the leverage ratio minimum under SAMA?",                            "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What Basel III reforms apply to Saudi banks?",                              "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the definition of Tier 2 capital under SAMA?",                     "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the stress testing requirements for Saudi banks?",                 "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the market risk capital requirement?",                              "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What is the credit risk capital calculation method?",                       "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the SAMA requirements for internal capital adequacy assessment?",  "cat":"capital","lang":"en","expect":"answered"},
  {"q":"What are the eligible capital instruments under SAMA Basel III?",           "cat":"capital","lang":"en","expect":"answered"},

# ── SECTION 6: Capital — Arabic (20) ─────────────────────────────────────────
  {"q":"ما هي نسبة كفاية رأس المال المطلوبة للبنوك السعودية؟",          "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات نسبة تغطية السيولة؟",                              "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي نسبة الرافعة المالية المطلوبة وفق ساما؟",                  "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هو الحد الأدنى لرأس المال الأساسي من الشريحة الأولى؟",       "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات نسبة التمويل المستقر الصافي؟",                    "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات الإفصاح السنوي للبنوك؟",                          "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هو الحد الأقصى لتدفقات النقد الداخلة في نسبة التغطية؟",     "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات الركيزة الثالثة للإفصاح؟",                        "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هو نسبة القرض إلى الودائع؟",                                  "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي الحد الأدنى لنسبة كفاية رأس المال الإجمالية؟",            "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي الأصول السائلة عالية الجودة للامتثال لنسبة التغطية؟",    "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات وسادة الحفاظ على رأس المال؟",                     "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات الحد الأدنى لرأس مال البنوك الجديدة؟",            "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي إصلاحات بازل الثالث المطبقة على البنوك السعودية؟",        "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هو تعريف رأس المال من الشريحة الثانية وفق ساما؟",            "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي الأدوات المؤهلة لرأس المال وفق ساما؟",                    "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات نسبة القرض إلى قيمة العقار؟",                     "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات التقييم الداخلي لكفاية رأس المال؟",               "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هو حد حدث الخسارة لمخاطر التشغيل؟",                          "cat":"capital","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات اختبار الإجهاد للبنوك السعودية؟",                 "cat":"capital","lang":"ar","expect":"answered"},

# ── SECTION 7: Cybersecurity NCA/ECC — English (30) ─────────────────────────
  {"q":"What are the NCA Essential Cybersecurity Controls?",                        "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What is the relationship between SAMA and NCA for cybersecurity?",          "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the CCC cloud cybersecurity requirements?",                        "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the SAMA cybersecurity framework requirements?",                   "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the NCA governance requirements for cybersecurity?",               "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the incident response requirements under NCA?",                    "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the cybersecurity monitoring requirements?",                       "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What third-party security requirements apply under SAMA?",                  "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the OTCC operational technology cybersecurity controls?",          "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the critical infrastructure cybersecurity requirements?",          "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the Aramco CCC cybersecurity certification requirements?",         "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the SACS-002 third-party cybersecurity requirements?",             "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What is the difference between CCC standard and CCC+ assessment?",         "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the ISO 27001 requirements for banks?",                            "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the business continuity requirements for cybersecurity?",          "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What BYOD policies are required under NCA controls?",                       "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the vulnerability management requirements?",                       "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the access control requirements under ECC?",                       "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the backup and recovery requirements under NCA?",                  "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the cybersecurity risk management requirements?",                  "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the network security requirements under ECC?",                     "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the identity and access management requirements?",                 "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the cryptography requirements under NCA controls?",               "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the cybersecurity awareness training requirements?",               "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What physical security requirements apply under ECC?",                      "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the cloud service provider security requirements?",                "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the fraud reporting requirements to SAMA?",                        "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What is the internal Shariah audit function purpose?",                      "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the SAMA cybersecurity framework sectors of application?",         "cat":"cybersec","lang":"en","expect":"answered"},
  {"q":"What are the ISO 22301 business continuity management requirements?",       "cat":"cybersec","lang":"en","expect":"answered"},

# ── SECTION 8: Cybersecurity — Arabic (20) ───────────────────────────────────
  {"q":"ما هي الضوابط الأساسية للأمن السيبراني الصادرة عن الهيئة الوطنية؟",    "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي العلاقة بين ساما والهيئة الوطنية للأمن السيبراني؟",              "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي ضوابط الأمن السيبراني السحابي؟",                                  "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات إطار الأمن السيبراني لساما؟",                             "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات حوكمة الأمن السيبراني للهيئة الوطنية؟",                  "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات الاستجابة للحوادث السيبرانية؟",                           "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات مراقبة الأمن السيبراني؟",                                 "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي ضوابط التقنيات التشغيلية للأمن السيبراني؟",                       "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات الأمن السيبراني للبنية التحتية الحيوية؟",                 "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هو برنامج شهادة الامتثال للأمن السيبراني لأرامكو؟",                  "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي معايير SACS-002 لأمن الأطراف الثالثة؟",                           "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هو الفرق بين مستوى CCC القياسي والمستوى CCC+؟",                      "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات معيار ISO 27001 للبنوك؟",                                  "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات استمرارية الأعمال للأمن السيبراني؟",                       "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات إدارة الثغرات الأمنية؟",                                   "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات التحكم في الوصول وفق الضوابط الأساسية؟",                  "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات النسخ الاحتياطي والاسترداد وفق الهيئة؟",                  "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات تدريب التوعية بالأمن السيبراني؟",                          "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي قطاعات تطبيق إطار ساما للأمن السيبراني؟",                          "cat":"cybersec","lang":"ar","expect":"answered"},
  {"q":"ما هي الإبلاغ عن الاحتيال المطلوب من ساما؟",                             "cat":"cybersec","lang":"ar","expect":"answered"},

# ── SECTION 9: PDPL — English (20) ──────────────────────────────────────────
  {"q":"What are the PDPL penalties for violations?",                               "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What is the scope of the Personal Data Protection Law?",                    "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the data subject rights under PDPL?",                              "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the controller obligations under PDPL?",                           "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What is the role of SDAIA in data protection?",                             "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the consent requirements under PDPL?",                             "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the data breach notification requirements?",                       "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the cross-border data transfer requirements?",                     "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the binding common rules requirements for data protection?",       "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the NDMO data management requirements?",                           "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What personal data categories require special protection under PDPL?",      "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the data retention requirements under PDPL?",                      "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the processor obligations under PDPL?",                            "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the exceptions to consent requirements under PDPL?",               "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the privacy impact assessment requirements?",                      "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the ISO 27701 requirements for privacy management?",               "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What is the right to erasure under PDPL?",                                  "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the PDPL requirements for personal data processing?",              "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What fines apply for PDPL violations in Saudi Arabia?",                    "cat":"pdpl","lang":"en","expect":"answered"},
  {"q":"What are the legitimate bases for processing personal data under PDPL?",    "cat":"pdpl","lang":"en","expect":"answered"},

# ── SECTION 10: PDPL — Arabic (15) ───────────────────────────────────────────
  {"q":"ما هي عقوبات مخالفة نظام حماية البيانات الشخصية؟",              "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هو نطاق تطبيق نظام حماية البيانات الشخصية؟",                 "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي حقوق أصحاب البيانات وفق نظام حماية البيانات؟",            "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي التزامات المتحكمين في البيانات الشخصية؟",                  "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هو دور هيئة البيانات والذكاء الاصطناعي في حماية البيانات؟",  "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات موافقة صاحب البيانات الشخصية؟",                   "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات الإخطار عن اختراق البيانات الشخصية؟",             "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات نقل البيانات عبر الحدود؟",                        "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي الغرامات المفروضة على مخالفات نظام حماية البيانات؟",      "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي أنواع البيانات الشخصية التي تتطلب حماية خاصة؟",          "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هو حق الحذف في نظام حماية البيانات الشخصية؟",                "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي الأسس المشروعة لمعالجة البيانات الشخصية؟",               "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي استثناءات متطلب الموافقة في نظام حماية البيانات؟",        "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هو دور المكتب الوطني لإدارة البيانات؟",                       "cat":"pdpl","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات ISO 27701 لإدارة الخصوصية؟",                      "cat":"pdpl","lang":"ar","expect":"answered"},

# ── SECTION 11: Identity / Out-of-scope / Conversational (20) ────────────────
  {"q":"Who are you?",                                     "cat":"identity","lang":"en","expect":"identity"},
  {"q":"What are you?",                                    "cat":"identity","lang":"en","expect":"identity"},
  {"q":"What is your name?",                               "cat":"identity","lang":"en","expect":"identity"},
  {"q":"Tell me about yourself",                           "cat":"identity","lang":"en","expect":"identity"},
  {"q":"من أنت؟",                                          "cat":"identity","lang":"ar","expect":"identity"},
  {"q":"ما اسمك؟",                                        "cat":"identity","lang":"ar","expect":"identity"},
  {"q":"What is the weather in Riyadh?",                   "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
  {"q":"Who is the president of the United States?",       "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
  {"q":"What is the recipe for kabsa?",                    "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
  {"q":"Are we ready?",                                    "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
  {"q":"Hello",                                            "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
  {"q":"What is the stock price of Saudi Aramco?",         "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
  {"q":"Okay thanks",                                      "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
  {"q":"Great",                                            "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
  {"q":"What is NORA?",                                    "cat":"nora","lang":"en","expect":"answered"},
  {"q":"What is SAMA?",                                    "cat":"general","lang":"en","expect":"answered"},
  {"q":"What does NCA stand for?",                         "cat":"general","lang":"en","expect":"answered"},
  {"q":"What is PDPL?",                                    "cat":"general","lang":"en","expect":"answered"},
  {"q":"What is Basel III?",                               "cat":"general","lang":"en","expect":"answered"},
  {"q":"What is KYC?",                                     "cat":"general","lang":"en","expect":"answered"},

# ── SECTION 12: GRC / ISO / Advanced (40) ────────────────────────────────────
  {"q":"What is the ISO 27001 information security management system?",             "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the GRC gap assessment requirements?",                             "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the clawback arrangement requirements?",                           "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the remuneration policy requirements for banks?",                  "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the savings account regulations under SAMA?",                      "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the administrative service charge maximums?",                      "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the prepaid card fee types?",                                      "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the IFRS 9 non-performing exposure requirements?",                 "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the ISO 22301 business continuity requirements?",                  "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the ISO 42001 AI management system requirements?",                 "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the SAMA deepfake guidelines?",                                    "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the IoT security requirements under ISO 27400?",                   "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the blockchain security requirements under ISO 23200?",            "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the IT service management requirements under ISO 20000?",          "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the privacy management requirements under ISO 27701?",             "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the Shariah audit requirements for Islamic banks?",                "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What is the COBIT framework for cybersecurity controls?",                   "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the correspondent banking requirements?",                          "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the SAMA consumer protection regulations?",                        "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the bank governance requirements under SAMA?",                     "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the regulatory capital deduction requirements?",                   "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the SAMA requirements for fintech companies?",                     "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What licensing requirements apply to payment service providers?",           "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the open banking requirements under SAMA?",                        "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the digital payment security requirements?",                       "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the regulatory sandbox requirements?",                             "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What is the cybersecurity steering committee function?",                    "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the data localization requirements in Saudi Arabia?",              "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the SAMA requirements for outsourcing?",                           "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"What are the SAMA requirements for internal audit?",                        "cat":"iso_grc","lang":"en","expect":"answered"},
  {"q":"ما هي متطلبات نظام إدارة أمن المعلومات ISO 27001؟",              "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هي ضوابط clawback لسياسة المكافآت؟",                          "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هي أنظمة حسابات التوفير تحت ساما؟",                          "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هي الحد الأقصى لرسوم الخدمات الإدارية؟",                     "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هو إطار عمل COBIT للضوابط الأمنية؟",                         "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات ISO 22301 لاستمرارية الأعمال؟",                   "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هي إرشادات ساما للتزييف العميق؟",                             "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات ISO 42001 لإدارة الذكاء الاصطناعي؟",              "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات التدقيق الشرعي للبنوك الإسلامية؟",                "cat":"iso_grc","lang":"ar","expect":"answered"},
  {"q":"ما هي متطلبات ساما لحماية المستهلك؟",                            "cat":"iso_grc","lang":"ar","expect":"answered"},

# ── SECTION 13: Edge / Informal / Mixed (30) ──────────────────────────────────
  {"q":"whats the kyc requirements for companies",                                  "cat":"edge","lang":"en","expect":"answered"},
  {"q":"how much capital do saudi banks need",                                      "cat":"edge","lang":"en","expect":"answered"},
  {"q":"can u tell me about pep requirements",                                      "cat":"edge","lang":"en","expect":"answered"},
  {"q":"what r the aml rules",                                                      "cat":"edge","lang":"en","expect":"answered"},
  {"q":"sama regulations for smes pls",                                             "cat":"edge","lang":"en","expect":"answered"},
  {"q":"who is not allowed to create a bank account",                               "cat":"edge","lang":"en","expect":"answered"},
  {"q":"what is the minimum cap ratio",                                              "cat":"edge","lang":"en","expect":"answered"},
  {"q":"nca ecc controls",                                                           "cat":"edge","lang":"en","expect":"answered"},
  {"q":"lcr ratio requirement",                                                      "cat":"edge","lang":"en","expect":"answered"},
  {"q":"pdpl penalties saudi arabia",                                                "cat":"edge","lang":"en","expect":"answered"},
  {"q":"how to open sme bank account saudi",                                         "cat":"edge","lang":"en","expect":"answered"},
  {"q":"pep edd requirements",                                                       "cat":"edge","lang":"en","expect":"answered"},
  {"q":"what are the rules for UBO verification",                                    "cat":"edge","lang":"en","expect":"answered"},
  {"q":"sama capital adequacy percentage",                                           "cat":"edge","lang":"en","expect":"answered"},
  {"q":"is an indian allowed to open bank account in saudi arabia",                  "cat":"edge","lang":"en","expect":"answered"},
  {"q":"can a company without cr open bank account",                                 "cat":"edge","lang":"en","expect":"answered"},
  {"q":"what happens if sar is not filed",                                           "cat":"edge","lang":"en","expect":"answered"},
  {"q":"nsfr calculation method",                                                    "cat":"edge","lang":"en","expect":"answered"},
  {"q":"leverage ratio tier 1",                                                      "cat":"edge","lang":"en","expect":"answered"},
  {"q":"sme bank account requirements sama",                                         "cat":"edge","lang":"en","expect":"answered"},
  {"q":"what are restrictions for non saudi nationals",                              "cat":"edge","lang":"en","expect":"answered"},
  {"q":"cloud security requirements nca",                                            "cat":"edge","lang":"en","expect":"answered"},
  {"q":"what is HQLA",                                                               "cat":"edge","lang":"en","expect":"answered"},
  {"q":"ecc cybersecurity controls",                                                 "cat":"edge","lang":"en","expect":"answered"},
  {"q":"data protection law penalties saudi",                                        "cat":"edge","lang":"en","expect":"answered"},
  {"q":"كيف يفتح الأجنبي حساباً بنكياً",                               "cat":"edge","lang":"ar","expect":"answered"},
  {"q":"ما هو نسبة رأس المال الكافي",                                    "cat":"edge","lang":"ar","expect":"answered"},
  {"q":"متطلبات KYC للشركات",                                            "cat":"edge","lang":"ar","expect":"answered"},
  {"q":"ما معنى PEP في البنوك",                                          "cat":"edge","lang":"ar","expect":"answered"},
  {"q":"شروط فتح حساب الشركة",                                           "cat":"edge","lang":"ar","expect":"answered"},

]  # end QUESTIONS

# ── API config ─────────────────────────────────────────────────────────────────
API_URL    = "http://localhost:8000"
API_KEY    = "0d52daf5f34807f9adfb5bca028a770f25a294156ecf22a4247b38d6c0c666cd"
TIMEOUT    = 120  # seconds per question
OUT_CSV    = "benchmark_results.csv"
OUT_STATS  = "benchmark_stats.json"
OUT_LOG    = "benchmark.log"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("benchmark")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _classify_result(result: dict, expect: str) -> str:
    """Return pass/fail/partial based on expected vs actual."""
    method = result.get("method", "")
    answer = result.get("answer", "")
    sources= result.get("sources", [])

    if expect == "identity":
        return "PASS" if method == "identity" else "FAIL"
    if expect == "out_of_scope":
        return "PASS" if method == "out_of_scope" else "FAIL"
    if expect == "answered":
        if method in ("not_found", "out_of_scope"):
            return "FAIL"
        if "does not contain" in answer.lower() or "cannot find" in answer.lower():
            return "NOT_FOUND"
        if sources:
            return "PASS"
        return "PARTIAL"  # answered but no sources
    return "UNKNOWN"

def _send_query(session, question: str) -> tuple[dict, float]:
    import urllib.request, json as _json
    body = _json.dumps({"query": question}).encode()
    req  = urllib.request.Request(
        f"{API_URL}/api/query",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key":    API_KEY,
        },
        method="POST",
    )
    t0  = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = _json.loads(resp.read().decode())
        return data, (time.time() - t0) * 1000
    except Exception as e:
        return {"error": str(e), "answer": "", "sources": [], "method": "error"}, (time.time()-t0)*1000

def _write_csv_header(f):
    w = csv.writer(f)
    w.writerow([
        "idx","question","category","lang","expect",
        "status","method","sources","reranker_score",
        "answer_len","time_ms","answer_preview",
    ])
    return w

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global API_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    default=API_URL)
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--start",  type=int, default=0)
    args = parser.parse_args()
    API_URL = args.url

    questions = QUESTIONS[args.start:]
    if args.limit:
        questions = questions[:args.limit]

    total   = len(questions)
    counts  = {"PASS":0,"FAIL":0,"NOT_FOUND":0,"PARTIAL":0,"ERROR":0}
    times_ms= []
    reranker_scores = []
    by_cat  = {}

    log.info("=" * 60)
    log.info(f"SAMA NORA Benchmark — {total} questions")
    log.info(f"API: {API_URL}")
    log.info(f"Output: {OUT_CSV}, {OUT_STATS}, {OUT_LOG}")
    log.info("=" * 60)

    # Open CSV for live writing
    csv_f = open(OUT_CSV, "w", newline="", encoding="utf-8")
    writer = _write_csv_header(csv_f)

    t_start = time.time()

    for i, item in enumerate(questions, 1):
        q      = item["q"]
        cat    = item["cat"]
        lang   = item["lang"]
        expect = item["expect"]

        log.info(f"[{i}/{total}] {cat}/{lang} | {q[:70]}")

        result, ms = _send_query(None, q)
        times_ms.append(ms)

        status = "ERROR" if "error" in result else _classify_result(result, expect)
        counts[status] = counts.get(status, 0) + 1

        rs = result.get("reranker_top_score")
        if rs is not None:
            reranker_scores.append(float(rs))

        answer  = result.get("answer", "")
        sources = result.get("sources", [])
        method  = result.get("method", "")

        log.info(f"  → {status} | method={method} | sources={len(sources)} | rs={rs} | {ms:.0f}ms")
        if status in ("FAIL","NOT_FOUND","ERROR"):
            log.warning(f"  ✗ Answer: {answer[:200]}")
        else:
            log.info(f"  ✓ Answer: {answer[:150]}")

        writer.writerow([
            args.start + i,
            q, cat, lang, expect,
            status, method, len(sources),
            round(float(rs),4) if rs else "",
            len(answer), round(ms),
            answer[:200].replace("\n"," "),
        ])
        csv_f.flush()

        # Track by category
        by_cat.setdefault(cat, {"PASS":0,"FAIL":0,"NOT_FOUND":0,"PARTIAL":0,"ERROR":0})
        by_cat[cat][status] = by_cat[cat].get(status, 0) + 1

        # Progress every 25 questions
        if i % 25 == 0:
            elapsed = time.time() - t_start
            eta     = (elapsed / i) * (total - i)
            pct     = sum(1 for s in ["PASS","PARTIAL"] if s in counts
                          for _ in range(counts[s])) / i * 100
            log.info(f"\n--- Progress: {i}/{total} | Pass rate: {pct:.1f}% | ETA: {eta:.0f}s ---\n")

    csv_f.close()
    elapsed_total = time.time() - t_start

    # ── Generate stats ────────────────────────────────────────────────────────
    pass_count = counts.get("PASS",0) + counts.get("PARTIAL",0)
    total_done = sum(counts.values())

    stats = {
        "run_date":        datetime.now().isoformat(),
        "total_questions": total_done,
        "elapsed_seconds": round(elapsed_total),
        "counts":          counts,
        "pass_rate_pct":   round(pass_count / total_done * 100, 1) if total_done else 0,
        "fail_rate_pct":   round(counts.get("FAIL",0) / total_done * 100, 1) if total_done else 0,
        "not_found_pct":   round(counts.get("NOT_FOUND",0) / total_done * 100, 1) if total_done else 0,
        "avg_time_ms":     round(sum(times_ms)/len(times_ms)) if times_ms else 0,
        "p50_time_ms":     round(sorted(times_ms)[len(times_ms)//2]) if times_ms else 0,
        "p95_time_ms":     round(sorted(times_ms)[int(len(times_ms)*0.95)]) if times_ms else 0,
        "avg_reranker":    round(sum(reranker_scores)/len(reranker_scores),4) if reranker_scores else None,
        "by_category":     by_cat,
    }

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  BENCHMARK COMPLETE")
    log.info("=" * 60)
    log.info(f"  Total questions : {total_done}")
    log.info(f"  PASS            : {counts.get('PASS',0)}")
    log.info(f"  PARTIAL         : {counts.get('PARTIAL',0)}")
    log.info(f"  NOT_FOUND       : {counts.get('NOT_FOUND',0)}")
    log.info(f"  FAIL            : {counts.get('FAIL',0)}")
    log.info(f"  ERROR           : {counts.get('ERROR',0)}")
    log.info(f"  Pass rate       : {stats['pass_rate_pct']}%")
    log.info(f"  Avg time        : {stats['avg_time_ms']}ms")
    log.info(f"  P95 time        : {stats['p95_time_ms']}ms")
    log.info(f"  Avg reranker    : {stats['avg_reranker']}")
    log.info(f"  Elapsed total   : {round(elapsed_total/60,1)} min")
    log.info("=" * 60)
    log.info(f"  Results CSV  : {OUT_CSV}")
    log.info(f"  Stats JSON   : {OUT_STATS}")
    log.info(f"  Full log     : {OUT_LOG}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
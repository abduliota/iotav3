"""
generate_test_questions_from_db.py

Read `sama_nora_chunks` from Supabase in batches, generate English regulatory
questions from each chunk, translate each question to Arabic, and append each
record to a JSONL output file incrementally.

Usage examples:
  python generate_test_questions_from_db.py
  python generate_test_questions_from_db.py --limit-rows 30 --batch-size 10
  python generate_test_questions_from_db.py --resume --export-json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

DIVIDER = "=" * 90
SUB_DIVIDER = "-" * 90
NOT_FOUND_PHRASES = [
    "does not contain",
    "cannot find",
    "not found in",
    "not available",
    "لا تتوفر",
    "لم أجد",
]


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _normalize_question(text: str) -> str:
    q = " ".join((text or "").strip().split())
    q = re.sub(r"[?.!]+$", "", q)
    return q.lower()


def _clean_question(text: str) -> str:
    q = " ".join((text or "").strip().split())
    q = re.sub(r"^[0-9]+[.)]\s*", "", q)
    return q.strip()


def _looks_arabic(text: str) -> bool:
    if not text:
        return False
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _short_preview(content: str, limit: int = 220) -> str:
    compact = " ".join((content or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _is_not_found_answer(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in NOT_FOUND_PHRASES)


def _log(log_path: Path | None, msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe)
    if log_path is None:
        return
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _iter_chunks(supabase, batch_size: int, limit_rows: int | None) -> Iterable[list[dict]]:
    offset = 0
    yielded = 0

    while True:
        remaining = (limit_rows - yielded) if limit_rows is not None else batch_size
        if limit_rows is not None and remaining <= 0:
            break
        this_batch = min(batch_size, remaining) if limit_rows is not None else batch_size

        resp = (
            supabase
            .table("sama_nora_chunks")
            .select("id, content, document_name, page_start, page_end")
            .order("id")
            .range(offset, offset + this_batch - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break

        yield rows
        got = len(rows)
        yielded += got
        offset += got

        if got < this_batch:
            break


def _extract_questions(payload: str) -> list[str]:
    """
    Parse model output expecting a JSON array, with safe fallbacks.
    """
    raw = (payload or "").strip()
    if not raw:
        return []

    # First attempt: strict JSON
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [_clean_question(str(x)) for x in data if str(x).strip()]
    except Exception:
        pass

    # Second attempt: find JSON array substring
    m = re.search(r"\[(.|\n|\r)*\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [_clean_question(str(x)) for x in data if str(x).strip()]
        except Exception:
            pass

    # Third attempt: numbered/bulleted lines
    lines = [ln.strip("-* ").strip() for ln in raw.splitlines() if ln.strip()]
    return [_clean_question(ln) for ln in lines if _clean_question(ln)]


def _verify_phrases_in_sources(answer: str, sources: list[dict]) -> tuple[float, list[str]]:
    if not answer or not sources or _is_not_found_answer(answer):
        return 0.0, []

    all_snippets = " ".join(
        ((s.get("snippet") or "") + " " + (s.get("document_name") or ""))
        for s in sources
    ).lower()

    words = re.findall(r"\b[a-zA-Z\u0600-\u06FF]{3,}\b", answer)
    phrases: list[str] = []
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i + 3]).lower()
        if len(phrase) > 10:
            phrases.append(phrase)
    if not phrases:
        return 0.0, []

    matched = [p for p in phrases if p in all_snippets]
    matched = list(dict.fromkeys(matched))[:8]
    ratio = len(matched) / len(phrases) if phrases else 0.0
    return round(ratio, 2), matched


def _llm_judge(client, model: str, answer: str, sources: list[dict]) -> tuple[str, str]:
    """
    Returns (verdict, reason), verdict in {GROUNDED, UNGROUNDED, PARTIAL, SKIP, ERROR, UNKNOWN}.
    """
    if not answer or not sources or _is_not_found_answer(answer):
        return "SKIP", "no answer/sources to evaluate or not-found response"

    judge_prompt = """You are a strict grounding evaluator for a regulatory RAG system.

Your job: determine if every factual claim in the ANSWER is explicitly present in the CONTEXT SNIPPETS.

Definitions:
- GROUNDED: every factual claim is directly traceable to snippets.
- UNGROUNDED: answer includes claims not found in snippets.
- PARTIAL: answer is partly grounded but contains at least one extra unsupported claim.

Reply with ONLY one of: GROUNDED / UNGROUNDED / PARTIAL
Then on the next line: one concise explanation.

CONTEXT SNIPPETS:
{snippets}

ANSWER TO EVALUATE:
{answer}
"""

    snippets = "\n\n".join(
        f"[{s.get('document_name','?')} p{s.get('page_start','?')}]\n{s.get('snippet','')}"
        for s in sources[:5]
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=120,
            messages=[{"role": "user", "content": judge_prompt.format(snippets=snippets, answer=answer)}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        lines = raw.split("\n", 1)
        verdict = lines[0].strip().upper() if lines else "UNKNOWN"
        reason = lines[1].strip() if len(lines) > 1 else ""
        if verdict not in {"GROUNDED", "UNGROUNDED", "PARTIAL"}:
            verdict = "UNKNOWN"
        return verdict, reason
    except Exception as e:
        return "ERROR", str(e)


def _score_combined(method: str, answer: str, judge_verdict: str, timed_out: bool, errored: bool) -> tuple[str, str]:
    """
    Combined correctness rule selected by user:
    PASS only when:
    - method is neither out_of_scope nor not_found
    - answer is not a not-found phrase
    - judge verdict is GROUNDED
    """
    if timed_out:
        return "TIMEOUT", "backend call timed out"
    if errored:
        return "ERROR", "backend call failed"

    if method in {"out_of_scope", "not_found"} or _is_not_found_answer(answer):
        return "FAIL", f"method/answer indicates non-answer ({method})"
    if judge_verdict == "GROUNDED":
        return "PASS", "method valid and answer grounded in sources"
    if judge_verdict == "PARTIAL":
        return "FAIL", "answer only partially grounded"
    if judge_verdict in {"UNGROUNDED", "UNKNOWN", "ERROR"}:
        return "FAIL", f"judge verdict is {judge_verdict}"
    return "FAIL", "judge skipped or unsupported verdict"


def _test_single_question(
    client,
    judge_model: str,
    question_record: dict,
    api_url: str,
    timeout_s: int,
) -> dict:
    import requests

    q_en = question_record.get("question_en", "")
    start = time.perf_counter()
    timed_out = False
    errored = False
    answer = ""
    method = "unknown"
    sources: list[dict] = []
    candidate_count = None
    reranker_top_score = None
    error_message = ""

    try:
        resp = requests.post(api_url, json={"query": q_en, "debug": True}, timeout=timeout_s)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()
        answer = (data.get("answer") or "").strip()
        sources = data.get("sources") or []
        cached = data.get("cached", False)
        method = data.get("method", "cached" if cached else "unknown")
        candidate_count = data.get("candidate_count", None)
        reranker_top_score = data.get("reranker_top_score", None)
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start
        timed_out = True
        error_message = "timeout"
    except Exception as e:
        elapsed = time.perf_counter() - start
        errored = True
        error_message = str(e)

    if timed_out or errored:
        judge_verdict, judge_reason = "SKIP", "backend error/timeout"
        phrase_ratio, matched_phrases = 0.0, []
    else:
        judge_verdict, judge_reason = _llm_judge(client, judge_model, answer, sources)
        phrase_ratio, matched_phrases = _verify_phrases_in_sources(answer, sources)

    overall_verdict, overall_reason = _score_combined(
        method=method,
        answer=answer,
        judge_verdict=judge_verdict,
        timed_out=timed_out,
        errored=errored,
    )

    return {
        "chunk_id": question_record.get("chunk_id"),
        "question_en": q_en,
        "question_ar": question_record.get("question_ar", ""),
        "elapsed": round(elapsed, 3),
        "method": method,
        "answer": answer,
        "sources_count": len(sources),
        "sources": sources,
        "candidate_count": candidate_count,
        "reranker_top_score": reranker_top_score,
        "judge_verdict": judge_verdict,
        "judge_reason": judge_reason,
        "phrase_match_ratio": phrase_ratio,
        "matched_phrases": matched_phrases,
        "overall_verdict": overall_verdict,
        "overall_reason": overall_reason,
        "timed_out": timed_out,
        "errored": errored,
        "error_message": error_message,
    }


def _init_cumulative_stats() -> dict:
    return {
        "tested": 0,
        "passed": 0,
        "failed": 0,
        "timeouts": 0,
        "errors": 0,
        "elapsed_sum": 0.0,
        "phrase_sum": 0.0,
        "phrase_count": 0,
        "method_counts": defaultdict(int),
        "judge_counts": defaultdict(int),
        "verdict_counts": defaultdict(int),
    }


def _update_stats(stats: dict, result: dict) -> None:
    stats["tested"] += 1
    if result["overall_verdict"] == "PASS":
        stats["passed"] += 1
    else:
        stats["failed"] += 1
    if result["timed_out"]:
        stats["timeouts"] += 1
    if result["errored"]:
        stats["errors"] += 1

    stats["elapsed_sum"] += float(result.get("elapsed", 0.0) or 0.0)
    if result.get("phrase_match_ratio", 0.0) > 0:
        stats["phrase_sum"] += result["phrase_match_ratio"]
        stats["phrase_count"] += 1

    stats["method_counts"][result.get("method", "unknown")] += 1
    stats["judge_counts"][result.get("judge_verdict", "UNKNOWN")] += 1
    stats["verdict_counts"][result.get("overall_verdict", "UNKNOWN")] += 1


def _append_batch_test_log(log_path: Path | None, batch_index: int, batch_results: list[dict], stats: dict) -> None:
    if log_path is None:
        return

    _log(log_path, f"\n{DIVIDER}")
    _log(log_path, f"BATCH TEST RESULTS  batch={batch_index}  questions_in_batch={len(batch_results)}")
    _log(log_path, DIVIDER)

    for i, r in enumerate(batch_results, start=1):
        _log(log_path, f"\n[{i}] chunk_id={r.get('chunk_id')}  verdict={r['overall_verdict']}  elapsed={r['elapsed']:.3f}s")
        _log(log_path, f"question_en: {r['question_en']}")
        _log(log_path, f"question_ar: {r['question_ar']}")
        _log(log_path, f"method={r['method']}  judge={r['judge_verdict']}  phrase_match={r['phrase_match_ratio']:.0%}")
        _log(log_path, f"overall_reason: {r['overall_reason']}")
        if r.get("judge_reason"):
            _log(log_path, f"judge_reason: {r['judge_reason']}")
        if r.get("candidate_count") is not None:
            _log(log_path, f"candidate_count={r['candidate_count']}")
        if r.get("reranker_top_score") is not None:
            _log(log_path, f"reranker_top_score={r['reranker_top_score']}")
        if r.get("error_message"):
            _log(log_path, f"error_message: {r['error_message']}")

        _log(log_path, "answer:")
        _log(log_path, SUB_DIVIDER)
        _log(log_path, r.get("answer") or "(empty)")
        _log(log_path, SUB_DIVIDER)

        _log(log_path, f"sources_count={r['sources_count']}")
        for s_idx, s in enumerate((r.get("sources") or [])[:5], start=1):
            _log(
                log_path,
                f"  source[{s_idx}] doc={s.get('document_name')} p={s.get('page_start')}-{s.get('page_end')} sim={s.get('similarity')}",
            )

    tested = stats["tested"]
    passed = stats["passed"]
    accuracy = (passed / tested * 100.0) if tested else 0.0
    avg_latency = (stats["elapsed_sum"] / tested) if tested else 0.0
    avg_phrase = (stats["phrase_sum"] / stats["phrase_count"]) if stats["phrase_count"] else 0.0

    _log(log_path, f"\n{DIVIDER}")
    _log(log_path, "CUMULATIVE SUMMARY (ALL TESTS SO FAR)")
    _log(log_path, DIVIDER)
    _log(log_path, f"total_tested={tested}")
    _log(log_path, f"total_passed={passed}")
    _log(log_path, f"total_failed={stats['failed']}")
    _log(log_path, f"total_timeouts={stats['timeouts']}")
    _log(log_path, f"total_errors={stats['errors']}")
    _log(log_path, f"accuracy_pct={accuracy:.2f}")
    _log(log_path, f"avg_latency_sec={avg_latency:.3f}")
    _log(log_path, f"avg_phrase_match={avg_phrase:.2f}")
    _log(log_path, f"method_counts={dict(stats['method_counts'])}")
    _log(log_path, f"judge_counts={dict(stats['judge_counts'])}")
    _log(log_path, f"verdict_counts={dict(stats['verdict_counts'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bilingual test questions from sama_nora_chunks.")
    parser.add_argument("--batch-size", type=int, default=10, help="Rows per DB fetch batch (default: 10)")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional cap on number of rows to process")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).with_name("generated_test_questions.jsonl")),
        help="Output JSONL path",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--export-json", action="store_true", help="Also export a JSON array file after generation")
    parser.add_argument("--test-backend", action="store_true", help="Test newly generated questions against backend API")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/api/query", help="Backend query API URL")
    parser.add_argument("--test-timeout", type=int, default=120, help="Per-question backend timeout in seconds")
    parser.add_argument("--log-file", type=str, default=None, help="Optional explicit test log file path")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.limit_rows is not None and args.limit_rows <= 0:
        raise ValueError("--limit-rows must be > 0 when provided")
    if args.test_timeout <= 0:
        raise ValueError("--test-timeout must be > 0")

    # Deferred imports so --help works even if deps are missing.
    from openai import OpenAI
    from supabase import create_client

    supabase_url = _require_env("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_KEY", "").strip()
    if not supabase_key:
        raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY).")
    openai_api_key = _require_env("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    supabase = create_client(supabase_url, supabase_key)
    client = OpenAI(api_key=openai_api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_path: Path | None = None
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cumulative_stats = _init_cumulative_stats()
    if args.test_backend:
        if args.log_file:
            log_path = Path(args.log_file)
        else:
            log_path = Path(__file__).with_name("test_logs") / f"generated_questions_test_{run_timestamp}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            f.write("")
        _log(log_path, "Generated Questions Batch Testing")
        _log(log_path, DIVIDER)
        _log(log_path, f"run_timestamp={run_timestamp}")
        _log(log_path, f"output_jsonl={output_path}")
        _log(log_path, f"resume={args.resume}")
        _log(log_path, f"openai_model={openai_model}")
        _log(log_path, f"batch_size={args.batch_size}")
        _log(log_path, f"limit_rows={args.limit_rows}")
        _log(log_path, f"api_url={args.api_url}")
        _log(log_path, f"test_timeout={args.test_timeout}")
        _log(log_path, "resume_behavior=only newly generated records in this run are tested")
        _log(log_path, DIVIDER)

    seen_en: set[str] = set()
    existing_records = 0
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                qn = _normalize_question(rec.get("question_en", ""))
                if qn:
                    seen_en.add(qn)
                existing_records += 1
        print(f"[resume] loaded {existing_records} existing records, {len(seen_en)} EN dedup keys.")

    mode = "a" if args.resume else "w"
    total_rows = 0
    total_saved = 0
    total_skipped_empty = 0
    total_skipped_dup = 0

    reached_row_limit = False

    with output_path.open(mode, encoding="utf-8") as out:
        for batch_index, rows in enumerate(_iter_chunks(supabase, args.batch_size, args.limit_rows), start=1):
            print(f"[batch {batch_index}] fetched {len(rows)} rows.")
            batch_records: list[dict] = []
            for row in rows:
                if args.limit_rows is not None and total_rows >= args.limit_rows:
                    reached_row_limit = True
                    break

                total_rows += 1
                chunk_id = row.get("id")
                content = (row.get("content") or "").strip()

                if not content:
                    total_skipped_empty += 1
                    continue

                # 1) Generate 3 English questions from the chunk.
                gen_prompt = (
                    "You generate concise regulatory test questions.\n"
                    "Using ONLY the provided chunk, produce exactly 3 distinct English questions.\n"
                    "Rules:\n"
                    "- Questions must be answerable directly from the chunk text.\n"
                    "- Keep each question clear and <= 20 words.\n"
                    "- Do not include answers or explanations.\n"
                    "- Return ONLY a JSON array of 3 strings.\n\n"
                    f"Chunk:\n{content}"
                )

                gen_resp = client.chat.completions.create(
                    model=openai_model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are a strict JSON-only generator."},
                        {"role": "user", "content": gen_prompt},
                    ],
                )
                generated_text = gen_resp.choices[0].message.content or ""
                en_questions = _extract_questions(generated_text)[:3]

                for q_en in en_questions:
                    q_en = _clean_question(q_en)
                    if not q_en:
                        continue
                    dedup_key = _normalize_question(q_en)
                    if dedup_key in seen_en:
                        total_skipped_dup += 1
                        continue

                    # 2) Translate each EN question to Arabic (same meaning).
                    tr_prompt = (
                        "Translate the following English question to Arabic.\n"
                        "Requirements:\n"
                        "- Preserve exact intent and scope.\n"
                        "- Output ONLY the Arabic question text.\n\n"
                        f"English question: {q_en}"
                    )
                    tr_resp = client.chat.completions.create(
                        model=openai_model,
                        temperature=0.1,
                        messages=[
                            {"role": "system", "content": "You are a precise EN->AR translator."},
                            {"role": "user", "content": tr_prompt},
                        ],
                    )
                    q_ar = _clean_question(tr_resp.choices[0].message.content or "")

                    if not q_ar or not _looks_arabic(q_ar):
                        # Skip low-quality translation output.
                        continue

                    record = {
                        "chunk_id": chunk_id,
                        "document_name": row.get("document_name"),
                        "page_start": row.get("page_start"),
                        "page_end": row.get("page_end"),
                        "question_en": q_en,
                        "question_ar": q_ar,
                        "source_preview": _short_preview(content),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()

                    seen_en.add(dedup_key)
                    total_saved += 1
                    batch_records.append(record)

            print(
                f"[batch {batch_index}] progress rows={total_rows} saved={total_saved} "
                f"skip_empty={total_skipped_empty} skip_dup={total_skipped_dup}"
            )

            if args.test_backend and batch_records:
                _log(log_path, f"\n{DIVIDER}")
                _log(log_path, f"START BATCH TEST batch={batch_index} generated_records={len(batch_records)}")
                _log(log_path, DIVIDER)
                batch_results: list[dict] = []
                for record in batch_records:
                    result = _test_single_question(
                        client=client,
                        judge_model=openai_model,
                        question_record=record,
                        api_url=args.api_url,
                        timeout_s=args.test_timeout,
                    )
                    batch_results.append(result)
                    _update_stats(cumulative_stats, result)

                _append_batch_test_log(log_path, batch_index, batch_results, cumulative_stats)

            if reached_row_limit:
                print(f"[stop] reached --limit-rows={args.limit_rows}")
                break

    print("\nGeneration complete.")
    print(f"  output      : {output_path}")
    print(f"  rows_seen   : {total_rows}")
    print(f"  records_saved: {total_saved}")
    print(f"  skipped_empty: {total_skipped_empty}")
    print(f"  skipped_dup  : {total_skipped_dup}")
    if args.test_backend and log_path is not None:
        print(f"  test_log     : {log_path}")

    if args.export_json:
        json_path = output_path.with_suffix(".json")
        records: list[dict] = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(records, jf, ensure_ascii=False, indent=2)
        print(f"  exported_json: {json_path}")


if __name__ == "__main__":
    main()


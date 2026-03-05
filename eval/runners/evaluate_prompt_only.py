"""Prompt-only evaluation (no retrieval).

Goal: test prompt techniques in isolation and log to MLflow.

- Uses the same prompt template versions as backend/app/rag_engine.py (v1/v2).
- Sends empty context to force guardrail behavior.

Run:
  PROMPT_VERSION=v1 python eval/runners/evaluate_prompt_only.py --eval eval/datasets/rag_eval_set.json
  PROMPT_VERSION=v2 python eval/runners/evaluate_prompt_only.py --eval eval/datasets/rag_eval_set.json

Notes:
- Requires GOOGLE_API_KEY.
- If mlflow is installed, logs params/metrics automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def _try_import_mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def _build_llm():
    """Use OpenRouter if configured; else fallback to Google Gemini."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "google/gemma-3-12b-instruct"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=openrouter_key,
            temperature=0.3,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000"),
                "X-Title": os.getenv("OPENROUTER_X_TITLE", "Pet Nutrition RAG (School Project)"),
            },
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing")

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)


def _prompt_template(version: str) -> str:
    # Keep in sync with backend/app/rag_engine.py
    v1 = """คุณเป็นผู้เชี่ยวชาญด้านโภชนาการสัตว์เลี้ยง ตอบโดยอิงจาก \"ข้อมูลจากฐานความรู้\" เท่านั้น

ข้อมูลจากฐานความรู้:
{context}

คำถามจากผู้ใช้: {question}

กติกา (สำคัญมาก):
- ห้ามใช้ความรู้จากที่อื่นนอกเหนือจาก {context}
- ทุกข้อสรุปต้องมีหลักฐานจาก {context} รองรับ
- ถ้า {context} ไม่พอ ให้ตอบว่า \"ไม่มีข้อมูลในฐานความรู้ ควรปรึกษาสัตวแพทย์\" และบอกว่าข้อมูลที่ขาดคืออะไร
- ถ้าเกี่ยวกับพิษ/อาการรุนแรง ให้แนะนำพบสัตวแพทย์ทันที

รูปแบบคำตอบ:
1) สรุปคำตอบ (1-3 บรรทัด)
2) รายละเอียด (bullet)
3) ข้อควรระวัง/เมื่อไหร่ควรพบสัตวแพทย์
4) แหล่งอ้างอิงจากฐานความรู้: (เช่น WSAVA/FEDIAF/CSV + หน้า ถ้ามี)

คำตอบ:"""

    v2 = """คุณเป็นผู้เชี่ยวชาญด้านโภชนาการสัตว์เลี้ยง ตอบโดยอิงจาก \"ข้อมูลจากฐานความรู้\" เท่านั้น

ข้อมูลจากฐานความรู้:
{context}

คำถามจากผู้ใช้: {question}

กติกา (สำคัญมาก):
- ห้ามใช้ความรู้จากที่อื่นนอกเหนือจาก {context}
- ถ้า {context} ว่าง/ไม่เพียงพอ ให้ *ไม่เดา* และให้ทำ 2 อย่าง:
  (1) ตอบว่า \"ไม่มีข้อมูลในฐานความรู้\" (2) ถามข้อมูลเพิ่มเติมที่จำเป็น 1-3 ข้อ (เช่น อายุ/น้ำหนัก/ระดับกิจกรรม/โรค/แพ้อาหาร)
- ถ้าเกี่ยวกับพิษ/อาการรุนแรง ให้แนะนำพบสัตวแพทย์ทันที

รูปแบบคำตอบ:
1) สรุปคำตอบ (1-3 บรรทัด)
2) รายละเอียด (bullet)
3) ข้อมูลที่ต้องการเพิ่ม (ถ้ามี)
4) ข้อควรระวัง/เมื่อไหร่ควรพบสัตวแพทย์
5) แหล่งอ้างอิงจากฐานความรู้: (เช่น WSAVA/FEDIAF/CSV + หน้า ถ้ามี)

คำตอบ:"""

    return v2 if version == "v2" else v1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=str, default=str(Path(__file__).resolve().parents[1] / "datasets" / "rag_eval_set.json"))
    ap.add_argument("--out", type=str, default="eval/results/prompt_only_results.json")
    args = ap.parse_args()

    prompt_version = os.getenv("PROMPT_VERSION", "v1")

    with open(args.eval, "r", encoding="utf-8") as f:
        items = json.load(f)

    llm = _build_llm()
    template = _prompt_template(prompt_version)

    rows = []
    latencies = []
    refusals = []

    for it in items:
        q = it["question"]
        prompt = template.format(context="", question=q)

        t0 = time.perf_counter()
        resp = llm.invoke(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        ans = getattr(resp, "content", str(resp))
        is_refusal = "ไม่มีข้อมูลในฐานความรู้" in (ans or "")

        rows.append({"id": it.get("id"), "question": q, "answer": ans, "latency_ms": round(latency_ms, 2), "refusal": is_refusal})
        latencies.append(latency_ms)
        refusals.append(1 if is_refusal else 0)

    summary = {
        "prompt_version": prompt_version,
        "n": len(rows),
        "refusal_rate": round(sum(refusals) / len(refusals), 3) if rows else 0.0,
        "latency_avg_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    print("=== Prompt-only Eval Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved: {args.out}")

    mlflow = _try_import_mlflow()
    if mlflow is not None:
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "pet-prompt-eval"))
        run_name = os.getenv("MLFLOW_RUN_NAME", f"prompt-{prompt_version}")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("prompt_version", prompt_version)
            mlflow.log_metric("refusal_rate", summary["refusal_rate"])
            mlflow.log_metric("latency_avg_ms", summary["latency_avg_ms"])
            mlflow.log_artifact(args.out)


if __name__ == "__main__":
    main()

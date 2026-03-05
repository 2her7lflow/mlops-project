# rag_engine.py
import os
import shutil
import time
import json
from pathlib import Path as _Path
from dotenv import load_dotenv

_RAG_IMPORT_ERROR = None
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.document_loaders import PyPDFLoader, CSVLoader
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever
except Exception as e:  # pragma: no cover
    _RAG_IMPORT_ERROR = e

# -----------------------------
# Env loading
# -----------------------------
BASE_DIR = _Path(__file__).resolve().parent
load_dotenv(override=False)

# Disable anonymized telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# -----------------------------
# Prompt Templates (Advanced Prompt Engineering)
# -----------------------------

# 1) Triage Prompt (Prompt Chaining)
TRIAGE_PROMPT_TEMPLATE = """You are a strict triage classifier for a veterinary assistant.

User input (may be Thai/English): "{question}"

Classify into ONE of these exact categories:
- EMERGENCY
- PROFILE_FACT
- CONSULT

Definitions (choose the safest option when uncertain):

1) EMERGENCY:
   Any life-threatening symptom OR high-risk ingestion/exposure OR major trauma.
   Examples (not exhaustive):
   - Breathing difficulty: หายใจลำบาก/หอบหนัก/ตัวเขียว/สำลัก
   - Neurologic: ชัก/หมดสติ/ซึมมาก/เดินเซเฉียบพลัน
   - Severe GI/abdomen: อาเจียนไม่หยุด, ถ่ายเป็นเลือด, ท้องบวมแข็ง/สงสัยบิดกระเพาะ
   - Bleeding/trauma: เลือดออกมาก, โดนรถชน, ตกจากที่สูง, แผลลึก
   - Heatstroke: ลิ้นม่วง, ตัวร้อนจัด, หอบมากหลังอยู่กลางแดด
   - Toxins/poisons/exposures: กินช็อกโกแลต, องุ่น/ลูกเกด, หอม/กระเทียมปริมาณมาก,
     ไซลิทอล, แอลกอฮอล์, ยาฆ่าแมลง/เหยื่อหนู, ยาคนทุกชนิด (พาราเซตามอล/ไอบูโพรเฟน ฯลฯ),
     ยานอนหลับ, สารทำความสะอาด, ยาเสพติด, พืชพิษ (เช่น ลิลลี่ในแมว), ของมีคม, กระดูกติดคอ
   If any EMERGENCY cue appears, output EMERGENCY even if the question also asks for advice.

2) PROFILE_FACT:
   ONLY if the user is explicitly asking for stored pet profile facts:
   name / age / breed / weight / sex / profile status.
   Examples: "how old is my dog", "น้องชื่ออะไร", "พันธุ์อะไร", "หนักกี่กิโล", "เพศอะไร"
   If the question contains advice-seeking beyond a simple fact (diet/health/behavior), do NOT choose PROFILE_FACT.

3) CONSULT:
   All other health, food, nutrition, diet, behavior, preventive care, general questions that are not EMERGENCY and not pure PROFILE_FACT.

Return ONLY the category name (EMERGENCY, PROFILE_FACT, or CONSULT). No extra text."""

# 2) Main RAG Prompt (V3)
RAG_PROMPT_TEMPLATE_V3 = """คุณคือผู้ช่วย AI สัตวแพทย์ที่เชี่ยวชาญและเป็นมิตร
คุณมีแหล่งข้อมูลอ้างอิง 2 แหล่ง เพื่อใช้ประกอบการตอบคำถาม:

1. ข้อมูลส่วนตัวสัตว์เลี้ยงของผู้ใช้ (Pet Profile):
{pet_context}

2. ฐานความรู้เชิงการแพทย์/โภชนาการ (Knowledge Base):
{context}

คำถามของผู้ใช้: {question}

คำสั่งในการตอบ (ปฏิบัติตามอย่างเคร่งครัด):
1. [กติกา Pet Profile สำคัญมาก] ใช้ข้อมูลจาก Pet Profile ได้ "เฉพาะ" ที่ปรากฏจริงเท่านั้น
   - ถ้าฟิลด์ใดเป็น "ไม่ระบุ" / ไม่มีข้อมูล: ห้ามเดา/ห้ามสร้างตัวเลขหรือชื่อขึ้นเอง
   - ถ้าไม่มีชื่อ ให้เรียกแบบทั่วไป เช่น "น้อง", "สัตว์เลี้ยงของคุณ"
2. ให้ใช้ข้อมูลจาก "Knowledge Base" เป็นหลักในการตอบ และนำ Pet Profile มาเชื่อมโยงคำนวณหรือให้คำแนะนำ (ถ้ามีข้อมูล)
3. หากคำถามเป็นเรื่องขอคำปรึกษา ให้จัดรูปแบบการตอบเป็น 4 หัวข้อ ดังนี้เสมอ:
      - **สรุปคำตอบ**: (ตอบคำถามหลักสั้นๆ ชัดเจน)
      - **รายละเอียด**: (อธิบายเหตุผลและรายละเอียดเชิงลึกโดยอิงจากฐานความรู้)
      - **ข้อควรระวัง/เมื่อไหร่ควรพบสัตวแพทย์**: (คำเตือนด้านความปลอดภัย หรือข้อแนะนำให้พบแพทย์)
      - **แหล่งอ้างอิง**: (อ้างอิงชื่อเอกสาร, บทที่, หรือโควตข้อความจากฐานความรู้)
4. หากไม่มีข้อมูลใน "Knowledge Base" ที่เกี่ยวข้องเลย ห้ามแต่งข้อมูลเองเด็ดขาด ให้ตอบว่า "จากฐานข้อมูลที่มีอยู่ ไม่พบข้อมูลที่เฉพาะเจาะจงสำหรับคำถามนี้" และแนะนำให้ปรึกษาสัตวแพทย์

ตอบเป็นภาษาไทยเสมอ ด้วยน้ำเสียงที่เข้าอกเข้าใจและเป็นมืออาชีพ
"""

# 3) Safety Check Prompt (Self-Feedback / Self-Critique)
SAFETY_PROMPT_TEMPLATE = """You are a strict veterinary safety reviewer and corrector.
Your job: check the draft answer for safety AND compliance with RAG-v3 rules (no hallucination, KB-grounded).

User Question: {question}

Pet Profile (may be empty / contain "ไม่ระบุ"):
{pet_context}

Knowledge Base Snippets (may be empty):
{context}

Draft Answer (to review):
{draft_answer}

Hard Rules (must enforce):
A) Pet Profile facts:
   - You may use ONLY facts that literally appear in Pet Profile above.
   - If a field is missing / "ไม่ระบุ": DO NOT guess. Do NOT invent name/age/weight/breed/sex or numbers.
   - If name is unknown, refer generically: "น้อง", "สัตว์เลี้ยงของคุณ"

B) Knowledge grounding:
   - For CONSULT questions, advice must be grounded in Knowledge Base Snippets above.
   - If the KB snippets do NOT contain relevant info for the question, you MUST NOT fabricate.
     In that case, respond exactly with:
     "จากฐานข้อมูลที่มีอยู่ ไม่พบข้อมูลที่เฉพาะเจาะจงสำหรับคำถามนี้"
     and recommend consulting a veterinarian.

C) Medical safety:
   - NEVER recommend human medicines or give dosing for any drug (Paracetamol/Tylenol/Ibuprofen/Aspirin, etc.).
   - Do not suggest dangerous home remedies or delaying care for severe symptoms.
   - Avoid definitive diagnosis; provide general guidance and when to see a vet.
   - If the question implies EMERGENCY (toxins, breathing trouble, seizures, unconsciousness, severe bleeding, severe bloat, repeated vomiting, etc.), rewrite to strongly urge immediate veterinary care.

Output rules:
1) If the draft answer is fully safe AND complies with A+B+C, return EXACTLY: SAFE
2) If UNSAFE or non-compliant, rewrite the answer in Thai to be safe and compliant:
   - Keep the 4-section format:
     **สรุปคำตอบ**
     **รายละเอียด**
     **ข้อควรระวัง/เมื่อไหร่ควรพบสัตวแพทย์**
     **แหล่งอ้างอิง** (cite only from Knowledge Base Snippets; if none relevant, follow rule B)
Return ONLY the rewritten answer or SAFE. No extra commentary.
"""

# -----------------------------
# Pet Profile helpers
# -----------------------------
def _get_field_with_key(pet_context: dict, keys: list[str]) -> tuple[str | None, object | None]:
    for k in keys:
        if k in pet_context:
            v = pet_context.get(k)
            if v is None:
                continue
            vv = str(v).strip()
            if vv == "":
                continue
            return k, v
    return None, None

def _safe_num_str(v: object) -> str:
    try:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return str(int(v))
        if isinstance(v, float):
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return str(v)
    except Exception:
        pass
    return str(v)

def _format_age(key: str | None, v: object) -> str:
    s = _safe_num_str(v)
    if key and "month" in key:
        return f"{s} เดือน"
    if any(x in (key or "") for x in ["age_years"]) or (key in ("age", "อายุ")):
        if isinstance(v, str) and any(u in v for u in ["ปี", "เดือน"]):
            return v.strip()
        return f"{s} ปี"
    if isinstance(v, str):
        return v.strip()
    return s

def _format_weight(key: str | None, v: object) -> str:
    s = _safe_num_str(v)
    if isinstance(v, str):
        if any(u in v.lower() for u in ["kg", "กก", "lbs", "lb"]):
            return v.strip()
    if key and "lb" in key:
        return f"{s} lb"
    return f"{s} กก."

def _normalize_pet_profile_for_llm(pet_context: dict | None) -> str:
    """Normalize pet profile to a stable format; missing core fields become 'ไม่ระบุ'."""
    if not pet_context:
        return "ไม่มีข้อมูลสัตว์เลี้ยง"

    name_k, name_v = _get_field_with_key(pet_context, ["name", "pet_name", "dog_name", "cat_name", "ชื่อ"])
    breed_k, breed_v = _get_field_with_key(pet_context, ["breed", "species_breed", "pet_breed", "สายพันธุ์", "พันธุ์"])
    age_k, age_v = _get_field_with_key(pet_context, ["age_years", "age_months", "age", "อายุ"])
    weight_k, weight_v = _get_field_with_key(pet_context, ["weight_kg", "weight_lbs", "weight", "น้ำหนัก"])
    sex_k, sex_v = _get_field_with_key(pet_context, ["sex", "gender", "เพศ"])
    allergy_k, allergy_v = _get_field_with_key(pet_context, ["allergies", "allergy", "แพ้"])
    cond_k, cond_v = _get_field_with_key(pet_context, ["health_conditions", "condition", "conditions", "โรค", "โรคประจำตัว"])

    lines = [
        f"- ชื่อ: {(str(name_v).strip() if name_v is not None else 'ไม่ระบุ')}",
        f"- สายพันธุ์: {(str(breed_v).strip() if breed_v is not None else 'ไม่ระบุ')}",
        f"- อายุ: {(_format_age(age_k, age_v) if age_v is not None else 'ไม่ระบุ')}",
        f"- น้ำหนัก: {(_format_weight(weight_k, weight_v) if weight_v is not None else 'ไม่ระบุ')}",
        f"- เพศ: {(str(sex_v).strip() if sex_v is not None else 'ไม่ระบุ')}",
    ]

    if allergy_v is not None:
        lines.append(f"- ประวัติแพ้: {str(allergy_v).strip()}")
    if cond_v is not None:
        lines.append(f"- โรคประจำตัว/ภาวะสุขภาพ: {str(cond_v).strip()}")

    used_keys = {k for k in [name_k, breed_k, age_k, weight_k, sex_k, allergy_k, cond_k] if k}
    for k, v in pet_context.items():
        if k in used_keys:
            continue
        if v is None:
            continue
        vv = str(v).strip()
        if vv == "":
            continue
        lines.append(f"- {k}: {vv}")

    return "\n".join(lines)

def _answer_from_pet_profile(question: str, pet_context: dict | None) -> str:
    """Return ONE natural Thai sentence using ONLY existing pet_context fields."""
    if not pet_context:
        return "ตอนนี้ยังไม่มีข้อมูล Pet Profile ของน้องในระบบค่ะ/ครับ"

    q = (question or "").lower()

    _, name_v = _get_field_with_key(pet_context, ["name", "pet_name", "dog_name", "cat_name", "ชื่อ"])
    breed_k, breed_v = _get_field_with_key(pet_context, ["breed", "species_breed", "pet_breed", "สายพันธุ์", "พันธุ์"])
    age_k, age_v = _get_field_with_key(pet_context, ["age_years", "age_months", "age", "อายุ"])
    weight_k, weight_v = _get_field_with_key(pet_context, ["weight_kg", "weight_lbs", "weight", "น้ำหนัก"])
    _, sex_v = _get_field_with_key(pet_context, ["sex", "gender", "เพศ"])

    def want_any(keys: list[str]) -> bool:
        return any(k in q for k in keys)

    parts: list[str] = []
    missing: list[str] = []

    if want_any(["ชื่อ", "name"]):
        if name_v is None:
            missing.append("ชื่อ")
        else:
            parts.append(f"น้องชื่อ {str(name_v).strip()}")
    if want_any(["สายพันธุ์", "พันธุ์", "breed"]):
        if breed_v is None:
            missing.append("สายพันธุ์")
        else:
            parts.append(f"น้องพันธุ์ {str(breed_v).strip()}")
    if want_any(["อายุ", "age", "how old"]):
        if age_v is None:
            missing.append("อายุ")
        else:
            parts.append(f"น้องอายุ {_format_age(age_k, age_v)}")
    if want_any(["น้ำหนัก", "weight"]):
        if weight_v is None:
            missing.append("น้ำหนัก")
        else:
            parts.append(f"น้องหนัก {_format_weight(weight_k, weight_v)}")
    if want_any(["เพศ", "sex", "gender"]):
        if sex_v is None:
            missing.append("เพศ")
        else:
            parts.append(f"น้องเป็นเพศ {str(sex_v).strip()}")

    if parts:
        # single natural sentence
        return ", ".join(parts) + " ค่ะ/ครับ"
    if missing:
        return "ใน Pet Profile ตอนนี้ยังไม่มีข้อมูล" + " / ".join(missing) + "ค่ะ/ครับ"
    # If it's a generic profile question
    if any(k in q for k in ["profile", "ข้อมูลสัตว์เลี้ยง", "ข้อมูล", "ประวัติ"]):
        summary = []
        if name_v is not None:
            summary.append(f"ชื่อ {str(name_v).strip()}")
        if breed_v is not None:
            summary.append(f"พันธุ์ {str(breed_v).strip()}")
        if age_v is not None:
            summary.append(f"อายุ {_format_age(age_k, age_v)}")
        if weight_v is not None:
            summary.append(f"น้ำหนัก {_format_weight(weight_k, weight_v)}")
        if summary:
            return "ตอนนี้ Pet Profile ของน้องมี " + ", ".join(summary) + " ค่ะ/ครับ"
    return "ตอนนี้มีข้อมูล Pet Profile ของน้องบางส่วนแล้วค่ะ/ครับ"

# -----------------------------
# Knowledge base paths
# -----------------------------
def _resolve_kb_dir() -> _Path:
    env = os.getenv("KNOWLEDGE_BASE_DIR")
    if env:
        p = _Path(env).expanduser().resolve()
        if p.exists():
            return p

    repo_root = BASE_DIR.parent
    candidates = [
        repo_root / "knowledge_base" / "knowledge_base",
        repo_root / "knowledge_base",
        repo_root / "backend" / "knowledge_base",
    ]
    for c in candidates:
        if c.exists():
            return c
    return repo_root / "knowledge_base"

KB_DIR = _resolve_kb_dir()
VECTORSTORE_DIR = KB_DIR / "processed" / "vectorstore"
PAGE_VECTORSTORE_DIR = KB_DIR / "processed" / "page_vectorstore"
CHUNKS_PATH = KB_DIR / "processed" / "chunks.jsonl"
PAGES_PATH = KB_DIR / "processed" / "pages.jsonl"

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _normalize_text(s: str) -> str:
    return " ".join((s or "").split())

def _rrf_fuse(rankings: list[list[Document]], k: int, rrf_k: int = 60) -> list[Document]:
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in rankings:
        for rank, d in enumerate(docs, start=1):
            key = f"{d.metadata.get('source','')}|{d.metadata.get('page', '')}|{d.metadata.get('row', '')}|{hash(d.page_content)}"
            by_key[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [by_key[key] for key, _ in best]

def _safe_similarity_search(store, query: str, k: int, meta_filter: dict | None = None) -> list[Document]:
    if meta_filter is None:
        return store.similarity_search(query, k=k)
    try:
        return store.similarity_search(query, k=k, filter=meta_filter)  # type: ignore[arg-type]
    except TypeError:
        pass
    try:
        return store.similarity_search(query, k=k, where=meta_filter)  # type: ignore[arg-type]
    except TypeError:
        pass
    except Exception:
        return []
    return []

class PetNutritionRAG:
    """
    Retrieval strategy:
      - Dense chunk retrieval (Chroma)
      - Keyword retrieval (BM25 on cached chunks)
      - Page/row-first narrowing (dense page/row store + scoped chunk search)
      - Fuse with RRF
    """

    def __init__(self):
        if _RAG_IMPORT_ERROR is not None:
            raise RuntimeError(
                "RAG dependencies are not installed. Install requirements.txt to enable RAG. "
                f"Import error: {_RAG_IMPORT_ERROR}"
            )
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")

        # embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=self.api_key,
        )

        # chat model
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("RAG_LLM_MODEL", "gemini-2.5-flash"),
            temperature=_env_float("RAG_TEMPERATURE", 0.3),
            google_api_key=self.api_key,
        )

        # stores
        self.persist_directory = str(VECTORSTORE_DIR)
        self.page_persist_directory = str(PAGE_VECTORSTORE_DIR)
        self.vectorstore = None
        self.page_store = None

        # prompt
        self.prompt = None

        # retrieval knobs
        self.k_vector = _env_int("RAG_K_VECTOR", 4)
        self.k_bm25 = _env_int("RAG_K_BM25", 6)
        self.k_pages = _env_int("RAG_K_PAGES", 4)
        self.k_per_page = _env_int("RAG_K_PER_PAGE", 2)
        self.k_final = _env_int("RAG_K_FINAL", 5)
        self.page_text_max_chars = _env_int("RAG_PAGE_TEXT_MAX_CHARS", 2000)

        # guardrail
        self.min_relevance = _env_float("RAG_MIN_RELEVANCE", 0.35)

        # internal retrievers
        self._bm25_chunks = None
        self._bm25_pages = None

        # info
        self.retriever_mode = "page-first+hybrid"

    # -----------------------------
    # KB loading / indexing
    # -----------------------------
    def _sanitize_metadata(self, metadata: dict) -> dict:
        if not metadata:
            return {}
        clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, _Path):
                clean[k] = str(v)
            elif isinstance(v, (list, dict, tuple)):
                try:
                    clean[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    clean[k] = str(v)
            else:
                clean[k] = str(v)
        return clean

    def _iter_kb_files(self) -> list[_Path]:
        if not KB_DIR.exists():
            return []
        files: list[_Path] = []
        for ext in ("*.pdf", "*.csv"):
            files.extend(KB_DIR.rglob(ext))

        def _is_processed(p: _Path) -> bool:
            parts = {x.lower() for x in p.parts}
            return "processed" in parts or "vectorstore" in parts or "page_vectorstore" in parts

        out = [p for p in files if p.is_file() and not _is_processed(p)]
        return sorted(out, key=lambda x: str(x).lower())

    def _load_file(self, p: _Path) -> list[Document]:
        section = p.parent.name
        try:
            if p.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(p))
                docs = loader.load()
            elif p.suffix.lower() == ".csv":
                try:
                    loader = CSVLoader(str(p), encoding="utf-8-sig")
                    docs = loader.load()
                except Exception:
                    docs = self._load_csv_fallback(p)
            else:
                return []
        except Exception as e:
            print(f"⚠️ Error loading {p}: {e}")
            return []

        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", str(p))
            d.metadata.setdefault("kb_section", section)
            d.metadata.setdefault("kb_file", p.name)
        return docs

    def _load_csv_fallback(self, p: _Path) -> list[Document]:
        import csv
        encodings = ["utf-8", "utf-8-sig", "cp874", "windows-1252", "latin1"]
        delimiters = [",", ";", "\t", "|"]

        raw = None
        used_enc = None
        for enc in encodings:
            try:
                raw = p.read_text(encoding=enc, errors="strict")
                used_enc = enc
                break
            except Exception:
                continue

        if raw is None:
            raw = p.read_text(encoding="utf-8", errors="replace")
            used_enc = "utf-8(replace)"

        raw = raw.replace("\x00", "")
        sample = raw[:4096]

        used_delim = ","
        try:
            sniff = csv.Sniffer().sniff(sample, delimiters=delimiters)
            used_delim = sniff.delimiter
        except Exception:
            best = (",", -1)
            for d in delimiters:
                c = sample.count(d)
                if c > best[1]:
                    best = (d, c)
            used_delim = best[0]

        docs: list[Document] = []
        reader = csv.DictReader(raw.splitlines(), delimiter=used_delim)
        if reader.fieldnames is None:
            reader2 = csv.reader(raw.splitlines(), delimiter=used_delim)
            for i, row in enumerate(reader2):
                text = " | ".join(str(x).strip() for x in row if str(x).strip())
                docs.append(Document(page_content=text, metadata={"row": i, "csv_encoding": used_enc, "csv_delimiter": used_delim}))
            return docs

        for i, row in enumerate(reader):
            parts = []
            for k, v in (row or {}).items():
                if k is None:
                    continue
                vv = "" if v is None else str(v).strip()
                if vv == "":
                    continue
                parts.append(f"{str(k).strip()}: {vv}")
            text = " | ".join(parts)
            docs.append(Document(page_content=text, metadata={"row": i, "csv_encoding": used_enc, "csv_delimiter": used_delim}))
        return docs

    def _save_docs_jsonl(self, docs: list[Document], out_path: _Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for d in docs:
                meta = d.metadata or {}
                safe_meta = {
                    "source": str(meta.get("source", "")),
                    "page": meta.get("page", None),
                    "row": meta.get("row", None),
                    "kb_section": meta.get("kb_section", None),
                    "kb_file": meta.get("kb_file", None),
                    "granularity": meta.get("granularity", None),
                }
                rec = {"page_content": d.page_content, "metadata": safe_meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _index_documents_chroma(
        self,
        docs: list[Document],
        persist_dir: _Path,
        batch_size: int = 20,
        delay_seconds: int = 15,
    ):
        if not docs:
            return None

        for d in docs:
            d.metadata = self._sanitize_metadata(getattr(d, "metadata", {}) or {})

        def _filter_empty(batch_docs: list[Document]) -> list[Document]:
            clean = []
            for d in batch_docs:
                text = (getattr(d, "page_content", "") or "").strip()
                if not text:
                    continue
                d.page_content = text
                clean.append(d)
            return clean

        persist_dir.mkdir(parents=True, exist_ok=True)

        initial_batch = _filter_empty(docs[:batch_size])
        if not initial_batch:
            raise ValueError("All documents in the initial batch are empty. Cannot initialize Chroma.")

        store = Chroma.from_documents(
            documents=initial_batch,
            embedding=self.embeddings,
            persist_directory=str(persist_dir),
        )

        for i in range(batch_size, len(docs), batch_size):
            time.sleep(delay_seconds)
            batch = _filter_empty(docs[i : i + batch_size])
            if not batch:
                continue
            store.add_documents(batch)

        return store

    def load_knowledge_base(self) -> int:
        documents: list[Document] = []
        kb_files = self._iter_kb_files()
        if not kb_files:
            print(f"🚫 No KB files found under: {KB_DIR}")
            return 0

        for p in kb_files:
            docs = self._load_file(p)
            if docs:
                documents.extend(docs)

        if not documents:
            print("🚫 KB files found but no documents could be loaded.")
            return 0

        # page/row docs (no splitting)
        page_docs: list[Document] = []
        for d in documents:
            meta = d.metadata or {}
            text = _normalize_text(d.page_content or "")
            if self.page_text_max_chars and len(text) > self.page_text_max_chars:
                text = text[: self.page_text_max_chars]
            page_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": meta.get("source", ""),
                        "page": meta.get("page", None),
                        "row": meta.get("row", None),
                        "kb_section": meta.get("kb_section", None),
                        "kb_file": meta.get("kb_file", None),
                        "granularity": "page_or_row",
                    },
                )
            )

        self._save_docs_jsonl(page_docs, PAGES_PATH)

        # chunk docs (splitting)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, length_function=len)
        splits = text_splitter.split_documents(documents)
        self._save_docs_jsonl(splits, CHUNKS_PATH)

        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        PAGE_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        self.vectorstore = self._index_documents_chroma(splits, VECTORSTORE_DIR)
        self.page_store = self._index_documents_chroma(page_docs, PAGE_VECTORSTORE_DIR)

        # ready
        self.setup_qa_chain()

        # NOTE (Windows): chromadb's Rust-backed `collection.count()` has been observed to
        # hard-crash the interpreter ("Windows fatal exception: access violation") in some
        # environments. Prefer a pure-Python count from our cached JSONL instead.
        return int(len(splits))

    def rebuild_vectorstore(self) -> int:
        if VECTORSTORE_DIR.exists():
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
        if PAGE_VECTORSTORE_DIR.exists():
            shutil.rmtree(PAGE_VECTORSTORE_DIR, ignore_errors=True)
        return self.load_knowledge_base()

    def load_existing_vectorstore(self) -> int:
        # Disable Chroma telemetry to avoid background PostHog threads during evaluation.
        # (Also helps keep eval runs deterministic/quiet.)
        try:
            from chromadb.config import Settings
            _chroma_settings = Settings(anonymized_telemetry=False)
        except Exception:
            _chroma_settings = None

        if VECTORSTORE_DIR.exists():
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                client_settings=_chroma_settings,
            )
        else:
            self.vectorstore = None

        if PAGE_VECTORSTORE_DIR.exists():
            self.page_store = Chroma(
                persist_directory=self.page_persist_directory,
                embedding_function=self.embeddings,
                client_settings=_chroma_settings,
            )
        else:
            self.page_store = None

        # IMPORTANT: Avoid calling Chroma collection.count() on Windows because it can
        # crash the process (native access violation). Use cached chunks jsonl instead.
        chunk_count = 0
        if CHUNKS_PATH.exists():
            try:
                with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                    chunk_count = sum(1 for _ in f)
            except Exception:
                chunk_count = 0

        # If we have a persisted vectorstore but no JSONL cache, still set up QA chain.
        # (We can't safely count, but we can attempt retrieval later.)
        if chunk_count == 0 and self.vectorstore is not None and VECTORSTORE_DIR.exists():
            chunk_count = 1

        if chunk_count > 0:
            self.setup_qa_chain()
        return chunk_count

    # -----------------------------
    # QA chain Setup
    # -----------------------------
    def setup_qa_chain(self):
        if self.vectorstore is None:
            return

        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE_V3,
            input_variables=["context", "pet_context", "question"],
        )

        # BM25 over chunks/pages cached from jsonl (no extra API calls)
        self._bm25_chunks = None
        if CHUNKS_PATH.exists():
            docs: list[Document] = []
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    docs.append(Document(page_content=rec.get("page_content", ""), metadata=rec.get("metadata", {})))
            if docs:
                self._bm25_chunks = BM25Retriever.from_documents(docs)
                self._bm25_chunks.k = self.k_bm25

        self._bm25_pages = None
        if PAGES_PATH.exists():
            pdocs: list[Document] = []
            with open(PAGES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pdocs.append(Document(page_content=rec.get("page_content", ""), metadata=rec.get("metadata", {})))
            if pdocs:
                self._bm25_pages = BM25Retriever.from_documents(pdocs)
                self._bm25_pages.k = self.k_pages

    # -----------------------------
    # Retrieval
    # -----------------------------
    def _retrieve_vector(self, query: str) -> tuple[list[Document], float]:
        try:
            pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.k_vector)
            docs = [d for d, _ in pairs]
            best = max([s for _, s in pairs], default=0.0)
            return docs, float(best)
        except Exception:
            docs = self.vectorstore.similarity_search(query, k=self.k_vector)
            return docs, 1.0 if docs else 0.0

    def _retrieve_bm25_chunks(self, query: str) -> list[Document]:
        if not self._bm25_chunks:
            return []
        return self._bm25_chunks.invoke(_normalize_text(query))

    def _retrieve_pages(self, query: str) -> list[Document]:
        if self.page_store is not None:
            try:
                pairs = self.page_store.similarity_search_with_relevance_scores(query, k=self.k_pages)
                return [d for d, _ in pairs]
            except Exception:
                try:
                    return self.page_store.similarity_search(query, k=self.k_pages)
                except Exception:
                    pass

        if self._bm25_pages:
            return self._bm25_pages.invoke(_normalize_text(query))[: self.k_pages]

        return []

    def _retrieve_scoped_chunks(self, query: str, page_doc: Document) -> list[Document]:
        meta = page_doc.metadata or {}
        meta_filter: dict = {"source": meta.get("source", "")}
        if meta.get("page") is not None:
            meta_filter["page"] = meta.get("page")
        if meta.get("row") is not None:
            meta_filter["row"] = meta.get("row")
        try:
            return _safe_similarity_search(self.vectorstore, query, k=self.k_per_page, meta_filter=meta_filter)
        except Exception:
            return []

    def _retrieve_one(self, query: str) -> tuple[list[Document], float, dict]:
        vec_docs, best_rel = self._retrieve_vector(query)
        kw_docs = self._retrieve_bm25_chunks(query)

        page_docs = self._retrieve_pages(query)
        scoped_docs: list[Document] = []
        for pd in page_docs:
            scoped_docs.extend(self._retrieve_scoped_chunks(query, pd))

        fused = _rrf_fuse([scoped_docs, vec_docs, kw_docs], k=self.k_final)

        meta = {
            "best_relevance": float(best_rel),
            "page_index_hit": 1.0 if page_docs else 0.0,
            "num_contexts": float(len(fused)),
            "unique_sources": float(len({d.metadata.get("source") for d in fused if d.metadata})),
            "retrieval_strategy": self.retriever_mode,
        }
        return fused, best_rel, meta

    # -----------------------------
    # Core Ask Function (Chaining, RAG, Safety)
    # -----------------------------
    def _build_context_with_sources(self, docs: list[Document]) -> str:
        parts: list[str] = []
        for d in docs:
            meta = d.metadata or {}
            src_raw = meta.get("source", "Unknown") or "Unknown"
            try:
                src = _Path(str(src_raw)).name
            except Exception:
                src = str(src_raw)

            loc = []
            if meta.get("page") is not None:
                loc.append(f"page={meta.get('page')}")
            if meta.get("row") is not None:
                loc.append(f"row={meta.get('row')}")
            tag = src + (f" ({', '.join(loc)})" if loc else "")
            parts.append(f"[{tag}] {d.page_content}")
        return "\n\n".join(parts)

    def ask(self, question: str, pet_context: dict | None = None) -> dict:
        t0 = time.perf_counter()

        # Phase 1: triage
        triage_msg = PromptTemplate.from_template(TRIAGE_PROMPT_TEMPLATE).format(question=question)
        triage_res = (self.llm.invoke(triage_msg).content or "").strip().upper()

        if "EMERGENCY" in triage_res:
            return {
                "answer": "⚠️ คำถามนี้มีแนวโน้มเป็นภาวะฉุกเฉิน/อันตรายต่อชีวิตสัตว์เลี้ยง กรุณารีบพาน้องไปพบสัตวแพทย์ หรือโทรหาโรงพยาบาลสัตว์ใกล้บ้านทันทีค่ะ/ครับ",
                "sources": [],
                "_meta": {"mode": "emergency", "total_ms": float((time.perf_counter() - t0) * 1000.0)},
            }

        if "PROFILE_FACT" in triage_res:
            return {
                "answer": _answer_from_pet_profile(question, pet_context),
                "sources": [],
                "_meta": {"mode": "pet_profile", "total_ms": float((time.perf_counter() - t0) * 1000.0)},
            }

        # Phase 2: consult (RAG)
        if not self.vectorstore:
            self.load_existing_vectorstore()
            if not self.vectorstore:
                return {
                    "answer": "จากฐานข้อมูลที่มีอยู่ ไม่พบข้อมูลที่เฉพาะเจาะจงสำหรับคำถามนี้\n\nแนะนำให้ปรึกษาสัตวแพทย์ค่ะ/ครับ",
                    "sources": [],
                    "_meta": {"mode": "consult", "guardrail_not_indexed": 1.0, "note": "kb_not_indexed"},
                }

        # Retrieval query (only include hints if present)
        query_text = question
        if pet_context:
            b_k, breed = _get_field_with_key(pet_context, ["breed", "species_breed", "pet_breed", "สายพันธุ์", "พันธุ์"])
            a_k, age = _get_field_with_key(pet_context, ["age_years", "age_months", "age", "อายุ"])
            w_k, weight = _get_field_with_key(pet_context, ["weight_kg", "weight_lbs", "weight", "น้ำหนัก"])
            hints: list[str] = []
            if breed is not None:
                hints.append(f"สายพันธุ์: {str(breed).strip()}")
            if age is not None:
                hints.append(f"อายุ: {_format_age(a_k, age)}")
            if weight is not None:
                hints.append(f"น้ำหนัก: {_format_weight(w_k, weight)}")
            if hints:
                query_text = " | ".join(hints) + "\n" + question

        t_retr0 = time.perf_counter()
        docs, best_rel, meta = self._retrieve_one(query_text)
        retrieval_ms = (time.perf_counter() - t_retr0) * 1000.0

        meta = meta or {}
        meta["mode"] = "consult"
        meta["retrieval_ms"] = float(retrieval_ms)

        if (not docs) or (float(best_rel) < float(self.min_relevance)):
            meta["guardrail_no_relevant_kb"] = 1.0
            meta["total_ms"] = float((time.perf_counter() - t0) * 1000.0)
            return {
                "answer": "จากฐานข้อมูลที่มีอยู่ ไม่พบข้อมูลที่เฉพาะเจาะจงสำหรับคำถามนี้\n\nแนะนำให้ปรึกษาสัตวแพทย์ค่ะ/ครับ",
                "sources": [],
                "_meta": meta,
            }

        if not self.prompt:
            self.setup_qa_chain()

        context = self._build_context_with_sources(docs)
        pet_info_str = _normalize_pet_profile_for_llm(pet_context)

        t_llm0 = time.perf_counter()
        msg = self.prompt.format(context=context, pet_context=pet_info_str, question=question)
        draft_out = self.llm.invoke(msg)
        draft_answer = getattr(draft_out, "content", None) or str(draft_out)
        meta["draft_llm_ms"] = float((time.perf_counter() - t_llm0) * 1000.0)

        # Phase 3: safety (FIX: pass pet_context + context)
        safety_msg = PromptTemplate.from_template(SAFETY_PROMPT_TEMPLATE).format(
            question=question,
            pet_context=pet_info_str,
            context=context,
            draft_answer=draft_answer,
        )
        t_safety0 = time.perf_counter()
        safety_out = self.llm.invoke(safety_msg)
        safety_res = (getattr(safety_out, "content", None) or str(safety_out)).strip()
        meta["safety_llm_ms"] = float((time.perf_counter() - t_safety0) * 1000.0)

        if safety_res == "SAFE" or safety_res.startswith("SAFE"):
            final_answer = draft_answer
            meta["safety_rewritten"] = False
        else:
            final_answer = safety_res
            meta["safety_rewritten"] = True

        meta["total_ms"] = float((time.perf_counter() - t0) * 1000.0)

        # sources
        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None),
                "row": doc.metadata.get("row", None),
                "snippet": (doc.page_content or "")[:220],
            })

        dedup = {}
        for s in sources:
            key = (s.get("source"), s.get("page"), s.get("row"))
            dedup[key] = s

        return {"answer": final_answer, "sources": list(dedup.values()), "_meta": meta}

# Singleton / disable switch
rag_system = None

class _DummyRAG:
    def ask(self, question: str, pet_context: dict | None = None) -> dict:
        return {"answer": "RAG is disabled (DISABLE_RAG=true).", "sources": [], "_meta": {"disabled": 1.0}}

    def load_existing_vectorstore(self) -> int:
        return 0

    def setup_qa_chain(self):
        return None

    def rebuild_vectorstore(self) -> int:
        return 0

def get_rag():
    global rag_system
    if rag_system is None:
        if os.getenv("DISABLE_RAG", "false").lower() in {"1", "true", "yes"}:
            rag_system = _DummyRAG()
        else:
            rag_system = PetNutritionRAG()
    return rag_system

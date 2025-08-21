#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG Ratings Extractor (V4.2 - Deterministic Table Grades + Narrative From Text)

What changed vs V4.1:
- **Deterministic table parser**: extracts only grades and strict items from the "Credit Rating" table.
- **Grade vs Narrative split**: table fills "(Grade)" fields; LLM fills long prose fields from text.
- **STRICT override** now protects only table-backed fields (grades + rating/anchor/final/modifiers/CAP).
- Kept geometric anchoring & improved table boundary detection (gap=15px + horizontal line hints).
- Jupyter-friendly; same CLI + run_pipeline API.

Additional changes in this workspace version:
- CLI overrides for paths, DPI, languages, and model path.
- --table-only flag to skip LLM and output only STRICT table fields.
- Optional fallback to table-only if model initialization fails.

Guarantees:
- STRICT_TABLE_FIELDS are copied exactly from the table region (no LLM, no normalization beyond .strip()).
- LLM can fill narrative fields but can NEVER overwrite STRICT_TABLE_FIELDS.
"""

from __future__ import annotations
import os
import gc
import re
import json
import time
import zipfile
import argparse
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional

import torch
import pandas as pd
import numpy as np
import pymupdf as fitz  # PyMuPDF
import easyocr
from langdetect import detect, LangDetectException

# transformers & optional quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
    _HAVE_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAVE_BNB = False

# ----------------------------- Logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    torch.set_num_threads(min(8, os.cpu_count() or 8))
except Exception:
    pass

# ----------------------------- Paths / Config -----------------------------
MODEL_PATH = os.environ.get("ESG_MODEL_PATH", "/home/inghero/data/dqanalysis/llm/parc/qwen3_14B")
COMPANIES_INPUT_DIR = os.environ.get("ESG_COMPANIES_INPUT_DIR", "/home/inghero/data/dqanalysis/llm/parc/notebooks/ESG_Ratings_input")
COMPANIES_INPUT_DIR_FLD = os.environ.get("ESG_COMPANIES_INPUT_DIR_FLD", "/home/inghero/data/dqanalysis/llm/parc/notebooks/ESG_Ratings_input/company_outputs")
OUTPUT_CSV = os.environ.get("ESG_OUTPUT_CSV", "/home/inghero/data/dqanalysis/llm/parc/notebooks/ESG_Ratings_output/ESG_Ratings_output.csv")
ZIP_SOURCE_PATH = os.environ.get("ESG_ZIP_SOURCE_PATH", os.path.join(os.path.dirname(COMPANIES_INPUT_DIR), "company_outputs.zip"))

EXTRACTED_TEXT_DIR = os.environ.get("ESG_EXTRACTED_TEXT_DIR", "extracted_texts_esg_ratings")
CHUNK_OUTPUTS_DIR = os.environ.get("ESG_CHUNK_OUTPUTS_DIR", "chunk_outputs_esg_ratings")
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)
os.makedirs(CHUNK_OUTPUTS_DIR, exist_ok=True)

# OCR / rendering
DPI = int(os.environ.get("ESG_DPI", "300"))
LANGUAGES = [s for s in os.environ.get("ESG_LANGS", "en,nl,fr").split(",") if s]
MODEL_DIR = os.path.expanduser("~/.EasyOCR/model")
ROW_THRESHOLD = 40  # for grouping OCR words into lines

TABLE_HEADING_STRINGS = [
    "Credit Rating", "Credit Ratings", "Rating Summary",
    "Ratings Summary", "Issuer Rating", "Rating Action",
    "RATING SNAPSHOT"
]

# ----------------------------- Fields & Rules -----------------------------
# Narrative fields (LLM fills from text)
NARRATIVE_BASE = [
    "Business Risk Profile",
    "Industry Risk Assessment",
    "Industry's ESG",
    "Competitive Positioning",
    "Governance",
    "Financial Risk Profile",
    "Cash Flow and Leverage",
    "Capitalisation",
    "Company's ESG",
]

# Grade fields (table fills strictly)
def grade_key(k: str) -> str:
    return f"{k} (Grade)"

GRADE_FIELDS = [grade_key(k) for k in NARRATIVE_BASE]

# Full schema (order matters in CSV)
FIELDS = [
    "Entity name", "Entity type", "Action Date", "Action", "Rating", "Outlook",
    "Last rating date", "First rating date",
] + sum(([k, grade_key(k)] for k in NARRATIVE_BASE), []) + [
    "Overall Ratings", "Anchor Rating", "Final Rating", "Modifiers", "CAP",
    "Industry/sector", "Industry/sector heatmap score",
    "Effect on industry risk assessment", "Industry risk adjustment",
    "Description of industry ESG profile", "Entity/company ESG score",
    "Effect on financial profile assessment", "Financial profile adjustment",
    "Description of entity's ESG profile", "Presence of ESG controversies",
    "ESG controversies score"
]

# STRICT: table-only keys (cannot be overwritten by LLM)
STRICT_TABLE_FIELDS = GRADE_FIELDS + ["Rating", "Overall Ratings", "Anchor Rating", "Final Rating", "Modifiers", "CAP"]

ALLOWED_ENTITY_TYPES = {"Corporate", "Sovereign", "Financial Institution", "Public Sector", "Project Finance", "Other"}
ALLOWED_ACTIONS = {"Affirmation", "Upgrade", "Downgrade", "Initiation", "Withdrawal", "Suspension"}
ALLOWED_OUTLOOKS = {"Stable", "Positive", "Negative", "Developing", "Under review", "N/A"}

# ----------------------------- Utilities -----------------------------
def sanitize_filename(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    s = re.sub(r"_+", "_", s).strip("._")
    return s

def unzip_if_needed(zip_path: str, extract_dir: str) -> None:
    if not os.path.exists(zip_path):
        logger.warning(f"Zip file not found at '{zip_path}'.")
        return
    logger.info(f"Extracting '{zip_path}' to '{extract_dir}'...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        logger.info("Extraction complete.")
    except Exception as e:
        logger.error(f"Error during unzipping: {e}")

# ----------------------------- OCR init -----------------------------
_reader = None
def get_ocr_reader():
    global _reader
    if _reader is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            logger.info("Initializing EasyOCR reader...")
            _reader = easyocr.Reader(LANGUAGES, gpu=torch.cuda.is_available(), model_storage_directory=MODEL_DIR)
        except Exception as e:
            logger.warning(f"EasyOCR init with model dir failed ({e}); retrying with defaults...")
            _reader = easyocr.Reader(LANGUAGES, gpu=torch.cuda.is_available())
    return _reader

def pixmap_to_numpy(pix: fitz.Pixmap) -> np.ndarray:
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

# ----------------------------- Full-page OCR (for LLM context) -----------------------------
def ocr_page_to_lines(img_np: np.ndarray, row_threshold: int = ROW_THRESHOLD) -> List[str]:
    reader = get_ocr_reader()
    result = reader.readtext(img_np, detail=1)
    result.sort(key=lambda r: (round(r[0][0][1], -1), r[0][0][0]))
    lines: List[str] = []
    current_row: List[str] = []
    last_y = None
    for bbox, text, prob in result:
        y = bbox[0][1]
        t = (text or "").strip()
        if not t:
            continue
        if last_y is None or abs(y - last_y) < row_threshold:
            current_row.append(t)
        else:
            if current_row:
                lines.append("\t".join(current_row))
            current_row = [t]
        last_y = y
    if current_row:
        lines.append("\t".join(current_row))
    return lines

def extract_text_from_pdf(pdf_path: str, dpi: int = DPI, row_threshold: int = ROW_THRESHOLD) -> Optional[str]:
    try:
        start_time = time.time()
        logger.info(f"Extracting text from {os.path.basename(pdf_path)}")
        company_name_raw = os.path.basename(os.path.dirname(pdf_path))
        sanitized_company_name = sanitize_filename(company_name_raw)
        sanitized_pdf_name = sanitize_filename(os.path.splitext(os.path.basename(pdf_path))[0])
        text_file_name = f"{sanitized_company_name}__{sanitized_pdf_name}.txt"
        text_out_path = os.path.join(EXTRACTED_TEXT_DIR, text_file_name)
        if os.path.exists(text_out_path) and os.path.getmtime(text_out_path) >= os.path.getmtime(pdf_path):
            with open(text_out_path, "r", encoding="utf-8") as f:
                cached = f.read()
            if cached.strip():
                logger.info(f"Using cached text for {os.path.basename(pdf_path)}")
                return cached
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        parts: List[str] = []
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_np = pixmap_to_numpy(pix)
            logger.info(f"OCR page {page_num} of {os.path.basename(pdf_path)}")
            lines = ocr_page_to_lines(img_np, row_threshold=row_threshold)
            page_text = "\n".join(lines).strip()
            if page_text:
                parts.append(page_text)
        doc.close()
        full_text = "\n\n".join(p for p in parts if p).strip()
        if not full_text:
            logger.warning("No text extracted from entire PDF.")
            return None
        with open(text_out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info(f"Saved text to {text_out_path}")
        logger.info(f"Extraction for {os.path.basename(pdf_path)} took {time.time() - start_time:.2f}s")
        return full_text
    except Exception as e:
        logger.error(f"Error during PDF text extraction for {os.path.basename(pdf_path)}: {e}")
        return None

# ----------------------------- High-precision TABLE extraction -----------------------------
def find_table_anchor(page: fitz.Page) -> Optional[fitz.Rect]:
    blocks = page.get_text("blocks", sort=True)
    for b in blocks:
        block_text = b[4]
        for heading in TABLE_HEADING_STRINGS:
            if heading.lower() in (block_text or "").lower() and len(block_text or "") < len(heading) + 20:
                return fitz.Rect(b[:4])
    return None

def find_table_end_y(page: fitz.Page, start_y: float) -> float:
    # gap heuristic
    words = sorted([w for w in page.get_text("words") if w[1] > start_y], key=lambda w: w[1])
    if not words:
        return start_y + 300
    last_y1 = words[0][3]
    gap_based_end = last_y1
    for word in words[1:]:
        if word[1] - last_y1 > 15:  # lowered from 20
            gap_based_end = last_y1
            break
        last_y1 = word[3]
    else:
        gap_based_end = last_y1
    # horizontal line hint
    try:
        drawings = page.get_drawings()
        horiz_lines_y = [d['rect'].y1 for d in drawings
                         if d['rect'].y0 > start_y and abs(d['rect'].y1 - d['rect'].y0) < 5
                         and d['rect'].x1 - d['rect'].x0 > 100]
        if horiz_lines_y:
            max_horiz_y = max(horiz_lines_y)
            return min(gap_based_end, max_horiz_y + 5)
    except Exception:
        pass
    return gap_based_end

def ocr_and_reconstruct_text(pixmap: fitz.Pixmap, reader: easyocr.Reader) -> str:
    img_np = pixmap_to_numpy(pixmap)
    ocr_results = reader.readtext(img_np, detail=1)
    if not ocr_results:
        return ""
    lines = defaultdict(list)
    for bbox, text, _ in ocr_results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        lines[round(y_center / 10.0) * 10.0].append((bbox[0][0], text))
    reconstructed_lines = []
    image_midpoint = pixmap.width / 2
    for y_pos in sorted(lines.keys()):
        line_words = sorted(lines[y_pos], key=lambda t: t[0])
        label = " ".join([word for x, word in line_words if x < image_midpoint]).strip()
        value = " ".join([word for x, word in line_words if x >= image_midpoint]).strip()
        if label and value:
            reconstructed_lines.append(f"{label}: {value}")
    return "\n".join(reconstructed_lines)

# ---------- Deterministic table parsing (no LLM) ----------
TABLE_FIELD_PATTERNS = [
    (r"^Business\s+Risk\s+Profi[l1]e$", grade_key("Business Risk Profile")),
    (r"^Industry\s*risk\s*assessment$", grade_key("Industry Risk Assessment")),
    (r"^Industry'?s\s*ESG$", grade_key("Industry's ESG")),
    (r"^Competitive\s*Positioning$", grade_key("Competitive Positioning")),
    (r"^Governance$", grade_key("Governance")),
    (r"^Financial\s*Risk\s*Profi[l1]e$", grade_key("Financial Risk Profile")),
    (r"^Cash\s*flow\s*and\s*leverage$", grade_key("Cash Flow and Leverage")),
    (r"^Capitali[sz]ation$", grade_key("Capitalisation")),
    (r"^Company'?s\s*ESG$", grade_key("Company's ESG")),

    (r"^Anchor\s*Rating$", "Anchor Rating"),
    (r"^Modifiers?$", "Modifiers"),
    (r"^Rating\s+standalone.*$", "Overall Ratings"),
    (r"^CAP(\s|$).*", "CAP"),
    (r"^Final\s*Rating.*$", "Final Rating"),
    (r"^Rating$", "Rating"),
]
NORMALIZE_MAP = [(re.compile(p, re.I), k) for p, k in TABLE_FIELD_PATTERNS]

def _normalize_table_key(label: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", (label or "").strip())
    for rx, key in NORMALIZE_MAP:
        if rx.match(s):
            return key
    return None

def parse_table_pairs_deterministic(reconstructed_text: str) -> Dict[str, str]:
    table_dict = {}
    for raw in (reconstructed_text or "").splitlines():
        if ":" not in raw:
            continue
        label, value = raw.split(":", 1)
        key = _normalize_table_key(label)
        val = (value or "").strip()
        if key and val:
            table_dict[key] = val
    return table_dict

def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """Extracts strict table values (grades + rating items) deterministically."""
    reader = get_ocr_reader()
    doc = fitz.open(pdf_path)
    found_tables = []
    try:
        for page in doc:
            anchor_rect = find_table_anchor(page)
            if not anchor_rect:
                continue
            table_end_y = find_table_end_y(page, anchor_rect.y1)
            table_rect = fitz.Rect(page.rect.x0, anchor_rect.y0, page.rect.x1, table_end_y + 5)
            logger.info(f"Dynamically determined table area on page {page.number + 1}: {table_rect}")
            mat = fitz.Matrix(DPI / 72, DPI / 72)
            pix = page.get_pixmap(matrix=mat, clip=table_rect, alpha=False)
            reconstructed_text = ocr_and_reconstruct_text(pix, reader)

            parsed = parse_table_pairs_deterministic(reconstructed_text)
            if parsed:
                # Keep only strict keys we care about
                strict_only = {k: v for k, v in parsed.items() if k in STRICT_TABLE_FIELDS}
                if strict_only:
                    found_tables.append(strict_only)
    finally:
        doc.close()
    return found_tables

def extract_tables_from_pdfs(pdf_files: List[str]) -> Dict[str, str]:
    all_extractions = []
    for p in pdf_files:
        tables_from_pdf = extract_tables_from_pdf(p)
        if tables_from_pdf:
            all_extractions.extend(tables_from_pdf)
    if not all_extractions:
        return {}
    # simple consolidation: prefer most frequent non-null per key
    merged: Dict[str, str] = {}
    for k in STRICT_TABLE_FIELDS:
        vals = [d.get(k) for d in all_extractions if d.get(k)]
        if not vals:
            continue
        merged[k] = Counter(vals).most_common(1)[0][0]
    return merged

# ----------------------------- LLM helpers (for narrative) -----------------------------
def estimate_free_vram_gb(default_cpu: float = 16) -> float:
    if not torch.cuda.is_available():
        return default_cpu
    try:
        free_mem, _ = torch.cuda.mem_get_info()
        return free_mem / (1024 ** 3)
    except Exception:
        return 16.0

def split_text_into_chunks(text: str, tokenizer, overlap_tokens: int = 128) -> List[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n = len(token_ids)
    if not torch.cuda.is_available():
        max_chunk_size = 1500
    else:
        gb = estimate_free_vram_gb()
        if gb > 24:
            max_chunk_size = 7000
        elif gb > 16:
            max_chunk_size = 4500
        elif gb > 8:
            max_chunk_size = 2500
        else:
            max_chunk_size = 1200
    chunks: List[str] = []
    start = 0
    step = max_chunk_size - overlap_tokens
    while start < n:
        end = min(start + max_chunk_size, n)
        chunks.append(tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
        if end == n:
            break
        start += step
    return chunks

def decode_generations(outputs, inputs, tokenizer) -> List[str]:
    seqs = outputs
    attn = inputs["attention_mask"]
    input_lens = attn.sum(dim=1).tolist()
    texts: List[str] = []
    for i in range(seqs.size(0)):
        gen_tokens = seqs[i][input_lens[i]:]
        texts.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
    return texts

def _build_sys_prompt(fields_str: str) -> str:
    return (
        "You are an expert data analyst. You will be given 'known_table_values' (strict, trusted grades) "
        "and a 'text_chunk' (narrative). Return a single valid JSON object with keys: "
        f"{fields_str}.\n"
        "RULES:\n"
        "- PRIORITIZE STRICT KEYS: For any key present in 'known_table_values', use it as-is.\n"
        "- FILL NARRATIVE FIELDS from the text_chunk (concise, factual sentences). Do not invent.\n"
        "- Dates as dd/mm/yyyy. 'Entity type', 'Action', 'Outlook' must be from their allowed lists.\n"
        "- If unknown, use null. Output only the JSON."
    )

def _apply_chat_template_safe(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys = next((m["content"] for m in messages if m.get("role") == "system"), "")
        usr = next((m["content"] for m in messages if m.get("role") == "user"), "")
        return f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>"

def extract_report_info_from_batch_json(batch_chunks, model, tokenizer, languages, known_table_values: Dict[str, str]):
    fields_str = ", ".join([f'"{f}"' for f in FIELDS])
    sys_prompt = _build_sys_prompt(fields_str)
    table_str = json.dumps(known_table_values, ensure_ascii=False, indent=2)
    batch_messages = [[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": (
            f"**Trusted Table Data (strict):**\n```json\n{table_str}\n```\n\n"
            f"**Text Chunk (narrative context):**\n```text\n{chunk}\n```\n\n"
            f"Return ONLY the final JSON object with all keys."
        )}
    ] for chunk in batch_chunks]
    try:
        prompts = [_apply_chat_template_safe(tokenizer, m) for m in batch_messages]
        max_input_len = min(getattr(tokenizer, "model_max_length", 8192), 8192)
        with torch.inference_mode():
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len
            ).to(model.device if hasattr(model, "device") else "cpu")
            outputs = model.generate(
                **inputs, max_new_tokens=800, do_sample=False, num_beams=1, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
            )
        decoded_list = decode_generations(outputs, inputs, tokenizer)
        results = []
        for decoded in decoded_list:
            match = re.search(r"\{.*\}", decoded, re.DOTALL)
            if match:
                try:
                    results.append(json.loads(match.group(0)))
                except json.JSONDecodeError:
                    results.append({**{f: None for f in FIELDS}, "Analysis Comment": "JSON parsing error."})
            else:
                results.append({**{f: None for f in FIELDS}, "Analysis Comment": "No JSON object found."})
        return results
    except Exception as e:
        return [{**{f: None for f in FIELDS}, "Analysis Comment": f"Batch processing error: {e}"}] * len(batch_chunks)

# ----------------------------- Consolidation & Post-processing -----------------------------
def consolidate_by_votes(chunk_jsons: List[Dict]) -> Dict:
    if not chunk_jsons:
        return {**{f: None for f in FIELDS}, "Analysis Comment": "No chunks."}
    agg: Dict = {}

    keys_majority = [
        "Entity type", "Action Date", "Action", "Rating", "Outlook",
        "Last rating date", "First rating date", "Overall Ratings",
        "Anchor Rating", "Final Rating", "Industry/sector",
        "Industry/sector heatmap score", "Entity/company ESG score",
        "Effect on financial profile assessment", "Financial profile adjustment",
        "Presence of ESG controversies", "ESG controversies score"
    ] + GRADE_FIELDS

    for k in keys_majority:
        vals = [j.get(k) for j in chunk_jsons if j and j.get(k)]
        if not vals:
            agg[k] = None
            continue
        if k == "Action":
            norm = []
            for v in vals:
                s = str(v)
                if re.search(r"affirm", s, re.IGNORECASE):
                    norm.append("Affirmation")
                elif re.search(r"upgrade", s, re.IGNORECASE):
                    norm.append("Upgrade")
                elif re.search(r"downgrade", s, re.IGNORECASE):
                    norm.append("Downgrade")
                elif re.search(r"initiat", s, re.IGNORECASE):
                    norm.append("Initiation")
                elif re.search(r"withdraw", s, re.IGNORECASE):
                    norm.append("Withdrawal")
                elif re.search(r"suspend", s, re.IGNORECASE):
                    norm.append("Suspension")
                else:
                    norm.append(s)
            vals = norm
        if k == "Outlook":
            vals = [str(v).title() for v in vals]
        agg[k] = Counter(vals).most_common(1)[0][0]

    # Long-text narrative fields → choose longest/most informative
    long_text_fields = [*NARRATIVE_BASE,
                        "Description of industry ESG profile", "Description of entity's ESG profile"]
    for k in long_text_fields:
        candidates = [j.get(k) for j in chunk_jsons if j and j.get(k)]
        agg[k] = max(candidates, key=len) if candidates else None

    names = [j.get("Entity name") for j in chunk_jsons if j and j.get("Entity name")]
    agg["Entity name"] = names[0] if names else None
    return agg

def postprocess_with_table_values(final_json: Dict, table_values: Dict) -> Dict:
    j = dict(final_json)
    for field in STRICT_TABLE_FIELDS:
        if field in table_values and table_values[field] not in (None, ""):
            j[field] = table_values[field].strip()
    if j.get("Presence of ESG controversies"):
        sc = j.get("ESG controversies score")
        if isinstance(sc, str) and re.search(r"manageable", sc, re.IGNORECASE):
            j["ESG controversies score"] = "score between 1 and 3"
    if j.get("Action"):
        a = str(j["Action"])
        if re.search(r"affirm", a, re.IGNORECASE):
            j["Action"] = "Affirmation"
        elif re.search(r"upgrade", a, re.IGNORECASE):
            j["Action"] = "Upgrade"
        elif re.search(r"downgrade", a, re.IGNORECASE):
            j["Action"] = "Downgrade"
        elif re.search(r"initiat", a, re.IGNORECASE):
            j["Action"] = "Initiation"
        elif re.search(r"withdraw", a, re.IGNORECASE):
            j["Action"] = "Withdrawal"
        elif re.search(r"suspend", a, re.IGNORECASE):
            j["Action"] = "Suspension"
    return j

# ----------------------------- Document Analysis (LLM over full text) -----------------------------
def pick_batch_size(avg_prompt_len_tokens: int) -> int:
    if not torch.cuda.is_available():
        return 1
    gb = estimate_free_vram_gb()
    if gb > 22:
        return 6 if avg_prompt_len_tokens < 3500 else 4
    elif gb > 16:
        return 4 if avg_prompt_len_tokens < 3500 else 3
    elif gb > 10:
        return 3
    return 2

def fast_detect_lang(s: str) -> str:
    try:
        return detect(s[:5000])
    except LangDetectException:
        return "en"
    except Exception:
        return "en"

def analyze_document(text: str, model, tokenizer, languages, file_name: str, known_table_values: Dict[str, str]):
    if not text or not text.strip():
        return {"error": "No text provided"}
    base_file_name = sanitize_filename(os.path.splitext(file_name)[0])
    t0 = time.time()
    chunks = split_text_into_chunks(text, tokenizer)
    logger.info(f"Chunking took {time.time() - t0:.2f}s | Num chunks: {len(chunks)}")
    if not chunks:
        return {"error": "Text too short or empty"}
    tok_lens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in chunks[:8]]
    avg_len = int(sum(tok_lens) / max(1, len(tok_lens))) if tok_lens else 1500
    batch_size = pick_batch_size(avg_len)

    chunk_outputs: List[Dict] = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        b0 = time.time()
        batch_results = extract_report_info_from_batch_json(
            chunks[i:i + batch_size], model, tokenizer, languages, known_table_values
        )
        chunk_outputs.extend(batch_results)
        logger.info(f"Batch {i // batch_size + 1}/{total_batches} took {time.time() - b0:.2f}s")

    c0 = time.time()
    final_json = consolidate_by_votes(chunk_outputs)
    with open(os.path.join(CHUNK_OUTPUTS_DIR, f"{base_file_name}_final_raw.json"), "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    final_json = postprocess_with_table_values(final_json, known_table_values)
    with open(os.path.join(CHUNK_OUTPUTS_DIR, f"{base_file_name}_final_consolidated.json"), "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    logger.info(f"Consolidation took {time.time() - c0:.2f}s")
    return final_json

# ----------------------------- Company Processing -----------------------------
def process_company_folder(company_path: str, model, tokenizer, table_only: bool = False) -> Optional[pd.DataFrame]:
    company_name_original = os.path.basename(company_path)
    sanitized_company_name = sanitize_filename(company_name_original)
    logger.info(f"--- Starting processing for company: {company_name_original} ---")

    if not os.path.isdir(company_path):
        logger.error(f"Company path is not a directory: {company_path}")
        return None

    pdf_files = [os.path.join(company_path, f) for f in os.listdir(company_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDFs found in: {company_path}")
        return None

    # 1) Deterministic table extraction → strict values
    table_values = extract_tables_from_pdfs(pdf_files)
    if table_values:
        logger.info(f"TABLE (STRICT) RESULTS: {json.dumps(table_values, indent=2)}")

    if table_only:
        # Build minimal row from table values only
        row: Dict[str, str] = {"Entity name": company_name_original}
        for field in FIELDS:
            if field == "Entity name":
                continue
            if field in STRICT_TABLE_FIELDS:
                row[field] = table_values.get(field, "Not found") if table_values else "Not found"
            else:
                row[field] = "Not found"
        row["comment"] = "Table-only mode. No OCR narrative or LLM analysis."
        return pd.DataFrame([row])

    # 2) OCR full text → LLM for narrative/metadata
    t0 = time.time()
    max_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        texts = list(executor.map(extract_text_from_pdf, pdf_files))
    consolidated_text = "\n\n".join([t for t in texts if t])
    logger.info(f"Text extraction for company took {time.time() - t0:.2f}s")

    if not consolidated_text.strip():
        row = {"Entity name": company_name_original, "comment": "Failed to extract text."}
        for f in FIELDS:
            if f != "Entity name":
                row[f] = "Not found"
        return pd.DataFrame([row])

    try:
        sample_text = "\n\n".join(consolidated_text.split("\n\n")[:6])
        detected_langs = [fast_detect_lang(sample_text)] if sample_text else ["en"]
    except Exception:
        detected_langs = ["en"]
    detected_langs = list(set(detected_langs))

    t1 = time.time()
    parsed_info = analyze_document(consolidated_text, model, tokenizer, detected_langs, f"{sanitized_company_name}.log", table_values)
    logger.info(f"LLM analysis for company took {time.time() - t1:.2f}s")

    row = {"Entity name": company_name_original}
    if "error" in parsed_info:
        row["comment"] = f"LLM Analysis Failed: {parsed_info.get('error')}"
    else:
        for field in FIELDS:
            row[field] = parsed_info.get(field) or "Not found"
        row["comment"] = f"Languages: {', '.join(detected_langs)} | LLM Comment: {parsed_info.get('Analysis Comment', 'N/A')}"
    return pd.DataFrame([row])

def process_companies(
    companies_dir: str, model, tokenizer,
    only_company: Optional[str] = None, only_company_path: Optional[str] = None, all_companies: bool = False,
    table_only: bool = False,
) -> None:
    targets: List[str] = []
    if not all_companies:
        if only_company_path:
            if os.path.isdir(only_company_path):
                targets = [only_company_path]
            else:
                logger.error(f"--company-path not found: {only_company_path}")
                return
        elif only_company:
            candidate = os.path.join(companies_dir, only_company)
            if os.path.isdir(candidate):
                targets = [candidate]
            else:
                subs = [d for d in os.listdir(companies_dir) if os.path.isdir(os.path.join(companies_dir, d))]
                match = next((d for d in subs if d.lower() == only_company.lower()), None)
                if match:
                    targets = [os.path.join(companies_dir, match)]
                else:
                    logger.error(f"Company '{only_company}' not found in {companies_dir}")
                    return
    if all_companies or not targets:
        if not os.path.isdir(companies_dir):
            logger.error(f"Input directory not found: {companies_dir}")
            return
        targets = [
            os.path.join(companies_dir, d)
            for d in os.listdir(companies_dir)
            if os.path.isdir(os.path.join(companies_dir, d))
        ]

    header_written = False
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    open(OUTPUT_CSV, "w", encoding="utf-8").close()

    for path in targets:
        df = process_company_folder(path, model, tokenizer, table_only=table_only)
        if df is None or df.empty:
            continue
        cols = ["Entity name"] + [c for c in FIELDS if c != "Entity name"] + ["comment"]
        df = df.reindex(columns=cols)
        df.to_csv(OUTPUT_CSV, index=False, sep="|", mode="a", header=not header_written)
        header_written = True
        del df
        gc.collect()
    if header_written:
        logger.info(f"Processing complete. Results saved to {OUTPUT_CSV}")
    else:
        logger.warning("No data extracted.")

# ----------------------------- Model init -----------------------------
def initialize_model(model_path: str = MODEL_PATH):
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    attn_impl = "sdpa" if torch.cuda.is_available() else None
    quant_config = None
    if torch.cuda.is_available() and _HAVE_BNB:
        try:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        except Exception as e:
            logger.warning(f"BitsAndBytesConfig failed: {e}. No quantization.")
    try:
        kwargs = dict(device_map=device_map, trust_remote_code=True, low_cpu_mem_usage=True)
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        if quant_config:
            kwargs["quantization_config"] = quant_config
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Model ready (attn={attn_impl or 'default'}, quant={'4bit' if quant_config else 'none'})")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return None, None

# ----------------------------- Public API -----------------------------
def run_pipeline(
    all_companies: bool = False, company: Optional[str] = None,
    company_path: Optional[str] = None, skip_unzip: bool = True,
    table_only: bool = False, fallback_table_only: bool = True,
    model_path: Optional[str] = None,
    companies_input_dir: Optional[str] = None,
    companies_input_dir_fld: Optional[str] = None,
    output_csv: Optional[str] = None,
    zip_source_path: Optional[str] = None,
    dpi: Optional[int] = None,
    languages: Optional[List[str]] = None,
) -> None:
    logger.info("--- Starting ESG Ratings Analysis (run_pipeline, V4.2) ---")
    # Apply overrides to module-level settings for simplicity
    global MODEL_PATH, COMPANIES_INPUT_DIR, COMPANIES_INPUT_DIR_FLD, OUTPUT_CSV, ZIP_SOURCE_PATH, DPI, LANGUAGES
    if model_path:
        MODEL_PATH = model_path
    if companies_input_dir:
        COMPANIES_INPUT_DIR = companies_input_dir
    if companies_input_dir_fld:
        COMPANIES_INPUT_DIR_FLD = companies_input_dir_fld
    if output_csv:
        OUTPUT_CSV = output_csv
    if zip_source_path:
        ZIP_SOURCE_PATH = zip_source_path
    if isinstance(dpi, int) and dpi > 0:
        DPI = dpi
    if languages:
        LANGUAGES = languages

    if not skip_unzip:
        unzip_if_needed(ZIP_SOURCE_PATH, COMPANIES_INPUT_DIR)

    model, tokenizer = (None, None)
    if not table_only:
        model, tokenizer = initialize_model(MODEL_PATH)
        if (model is None or tokenizer is None) and fallback_table_only:
            logger.critical("Model/tokenizer init failed. Falling back to table-only mode.")
            table_only = True
        elif (model is None or tokenizer is None):
            logger.critical("Model and tokenizer failed to initialize and no fallback enabled. Exiting.")
            return

    try:
        process_companies(
            COMPANIES_INPUT_DIR_FLD, model, tokenizer,
            only_company=company, only_company_path=company_path,
            all_companies=all_companies or (not company and not company_path),
            table_only=table_only,
        )
    finally:
        try:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    logger.info("--- ESG Ratings Analysis Finished ---")

# ----------------------------- CLI -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ESG Ratings Extractor (V4.2 - Deterministic table grades + narrative)")
    p.add_argument("--all", action="store_true", help="Process ALL companies.")
    p.add_argument("--company", type=str, default=None, help="Process only this company by name.")
    p.add_argument("--company-path", type=str, default=None, help="Process only this specific company folder path.")
    p.add_argument("--skip-unzip", action="store_true", help="Skip unzipping.")
    p.add_argument("--row-threshold", type=int, default=ROW_THRESHOLD, help="Row grouping threshold (px) for OCR.")
    # Overrides
    p.add_argument("--model-path", type=str, default=None, help="Override model path.")
    p.add_argument("--companies-input-dir", type=str, default=None, help="Override companies input dir (root).")
    p.add_argument("--companies-input-dir-fld", type=str, default=None, help="Override companies subfolder with PDFs.")
    p.add_argument("--output-csv", type=str, default=None, help="Override output CSV path.")
    p.add_argument("--zip-source-path", type=str, default=None, help="Override ZIP source path for unzipping.")
    p.add_argument("--dpi", type=int, default=None, help="Rendering DPI for OCR (default 300).")
    p.add_argument("--languages", type=str, default=None, help="Comma-separated OCR languages (e.g., en,nl,fr).")
    # Modes
    p.add_argument("--table-only", action="store_true", help="Skip LLM and output only table-backed fields.")
    p.add_argument("--no-fallback-table-only", action="store_true", help="Do NOT fallback to table-only on model init failure.")
    return p

def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

if __name__ == "__main__":
    if _running_in_ipython():
        logger.info("Detected IPython/Jupyter. Use run_pipeline(...) to execute.")
    else:
        args = build_arg_parser().parse_args()
        ROW_THRESHOLD = int(getattr(args, "row_threshold", ROW_THRESHOLD))
        # Gather overrides
        langs = None
        if getattr(args, "languages", None):
            langs = [s for s in str(args.languages).split(',') if s]
        fallback = not bool(getattr(args, "no_fallback_table_only", False))
        logger.info("--- Starting ESG Ratings Analysis Script (V4.2) ---")
        run_pipeline(
            all_companies=bool(args.all),
            company=args.company,
            company_path=args.company_path,
            skip_unzip=bool(args.skip_unzip),
            table_only=bool(args.table_only),
            fallback_table_only=fallback,
            model_path=args.model_path,
            companies_input_dir=args.companies_input_dir,
            companies_input_dir_fld=args.companies_input_dir_fld,
            output_csv=args.output_csv,
            zip_source_path=args.zip_source_path,
            dpi=args.dpi,
            languages=langs,
        )
        logger.info("--- ESG Ratings Analysis Script Finished ---")


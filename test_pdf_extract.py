"""
Korean PDF parsing comparison — pages 26-30 of data/solidworks_manual.pdf.

Path A  PaddleOCR (korean, PP-OCRv4)
        → fix 3: image preprocessing (grayscale + Otsu binarization + sharpen)
        → fix 4: layout analysis (mask figure/diagram regions before OCR)
        → fix 1: confidence filter (drop items < 0.85)
        → fix 2: character validity filter (drop garbage from graphic misreads)
        → kiwipiepy spacing correction
        → 600-char chunks

Path B  PyMuPDF renders page as JPEG → Qwen2.5-VL-32B vision LLM → 600-char chunks

Output:
  data/pdf_extract_output/page{N}_parse_compare.json   — per-page
  data/pdf_extract_output/pages26_30_parse_compare.json — combined
"""

import os
import re
import json
import base64
import math
import warnings
import logging
import time
import unicodedata
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image

os.environ["FLAGS_use_mkldnn"] = "0"   # disable oneDNN (crashes on Windows CPU)

import cv2
import fitz
from paddleocr import PaddleOCR
from kiwipiepy import Kiwi
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH   = Path("data/solidworks_manual.pdf")
PAGE_RANGE = range(25, 30)   # 0-based -> pages 26-30
OUT_DIR    = Path("data/pdf_extract_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RENDER_DPI    = 300
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 50

CONF_THRESHOLD    = 0.85   # fix 1: drop items below this confidence
VALID_CHAR_RATIO  = 0.60   # fix 2: min fraction of valid chars to keep an item
FIGURE_MIN_AREA   = 40_000 # fix 4: min pixel area to treat a region as a figure
DET_DB_THRESH     = 0.20   # fix 5: lower detection threshold to recover faint/missed text (default 0.3)

VLM_BASE_URL = "http://192.168.1.208:8000/v1"
VLM_API_KEY  = "EMPTY"
VLM_MODEL    = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_page_as_array(pdf_path: Path, page_idx: int) -> np.ndarray:
    """Render PDF page to uint8 RGB numpy array at RENDER_DPI."""
    doc  = fitz.open(pdf_path)
    page = doc[page_idx]
    mat  = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def render_page_as_jpeg_b64(pdf_path: Path, page_idx: int, quality: int = 90) -> str:
    """Render PDF page to base64-encoded JPEG string for VLM input."""
    arr = render_page_as_array(pdf_path, page_idx)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Fix 3: Image preprocessing ────────────────────────────────────────────────

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Grayscale → CLAHE (local contrast enhancement) → adaptive threshold → unsharp sharpen.

    Why not global Otsu: a single global threshold is skewed by figure/diagram content
    on the page, which erases faint text in those regions (e.g. text adjacent to diagrams).
    CLAHE + adaptive threshold each operate on small local tiles, preserving text even
    where the background intensity varies.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # CLAHE: amplify local contrast (tileGridSize balances global/local)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Adaptive threshold: each 51×51 tile gets its own threshold
    binarized = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=51, C=10,
    )
    # Unsharp mask: emphasise text edges
    blurred   = cv2.GaussianBlur(binarized, (0, 0), 1.5)
    sharpened = cv2.addWeighted(binarized, 1.8, blurred, -0.8, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


# ── Fix 4: Layout analysis — detect figure regions ────────────────────────────

def detect_figure_regions(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Find large non-text (figure/diagram) regions in the page image.
    Returns list of (x1, y1, x2, y2) bounding boxes in pixel coords.
    Strategy: dark content on white page → dilate text → find large blobs that
    span both horizontal and vertical directions (text lines are thin & wide,
    figures are more block-shaped with significant height).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # invert so content = white, background = black
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # dilate to merge nearby strokes into blobs
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 3))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 60))
    h_dilated = cv2.dilate(thresh, h_kern)
    v_dilated = cv2.dilate(thresh, v_kern)
    # regions that have BOTH horizontal and vertical extent = figures
    combined = cv2.bitwise_and(h_dilated, v_dilated)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    figure_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / max(h, 1)
        # figures: large area, not extremely wide (pure text lines have aspect >> 5)
        if area >= FIGURE_MIN_AREA and aspect < 8:
            figure_rects.append((x, y, x + w, y + h))
    return figure_rects


def bbox_overlaps_figure(bbox: list, figure_rects: list[tuple], thresh: float = 0.50) -> bool:
    """Return True if the OCR bbox overlaps a figure region by more than thresh fraction."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
    b_area = max((bx2 - bx1) * (by2 - by1), 1)
    for fx1, fy1, fx2, fy2 in figure_rects:
        ix1 = max(bx1, fx1); iy1 = max(by1, fy1)
        ix2 = min(bx2, fx2); iy2 = min(by2, fy2)
        if ix2 > ix1 and iy2 > iy1:
            if (ix2 - ix1) * (iy2 - iy1) / b_area >= thresh:
                return True
    return False


# ── Fix 2: Character validity filter ─────────────────────────────────────────

def _is_valid_char(ch: str) -> bool:
    """True if ch is Korean, Latin, digit, or common punctuation."""
    cp = ord(ch)
    if 0xAC00 <= cp <= 0xD7A3: return True   # Hangul syllables
    if 0x1100 <= cp <= 0x11FF: return True    # Hangul Jamo
    if 0x3130 <= cp <= 0x318F: return True    # Hangul compat jamo
    if 0x0020 <= cp <= 0x007E: return True    # Basic ASCII (includes Latin, digits, punctuation)
    if 0x00C0 <= cp <= 0x024F: return True    # Extended Latin
    return False


def is_valid_text(text: str) -> bool:
    """Return True if the item looks like real text (not a figure label/callout)."""
    if not text:
        return False
    # Reject items containing bracket chars — these are consistently figure callout
    # labels (e.g. '[6 lo lo', '[0 [o [Z') that happen to have high OCR confidence
    if '[' in text or ']' in text:
        return False
    # Reject items that are only punctuation / single non-alphanumeric characters
    if len(text.strip()) <= 2 and not any(ch.isalnum() for ch in text):
        return False
    valid = sum(1 for ch in text if _is_valid_char(ch))
    return valid / len(text) >= VALID_CHAR_RATIO


# ── Path A: PaddleOCR + all fixes + kiwipiepy ────────────────────────────────

def _bbox_overlaps_any(bbox: list, existing: list[dict], iou_thresh: float = 0.3) -> bool:
    """True if bbox overlaps any existing item bbox by > iou_thresh (IoU)."""
    xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
    ax1, ay1, ax2, ay2 = min(xs), min(ys), max(xs), max(ys)
    a_area = max((ax2 - ax1) * (ay2 - ay1), 1)
    for it in existing:
        bxs = [p[0] for p in it["bbox"]]; bys = [p[1] for p in it["bbox"]]
        bx1, by1, bx2, by2 = min(bxs), min(bys), max(bxs), max(bys)
        b_area = max((bx2 - bx1) * (by2 - by1), 1)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / min(a_area, b_area) >= iou_thresh:
                return True
    return False


def ocr_paddle(paddle: PaddleOCR, raw_img: np.ndarray) -> dict:
    """
    Run PaddleOCR with all fixes applied — two-pass merge strategy:
      Pass 1  preprocessed image (CLAHE + adaptive threshold): good for most text
      Pass 2  raw image: recovers text that preprocessing locally destroys
              (common near figures where adaptive threshold fails in a horizontal strip)
    Results are deduplicated by bbox IoU before filtering.
    """
    # fix 3: preprocess
    proc_img = preprocess_for_ocr(raw_img)

    # fix 4: detect figure regions on the original (color) image
    figure_rects = detect_figure_regions(raw_img)

    def _parse_results(results, source: str) -> list[dict]:
        items = []
        for line in (results[0] or []):
            bbox, (text, conf) = line
            text = text.strip()
            if not text:
                continue
            items.append({
                "bbox": [[float(pt[0]), float(pt[1])] for pt in bbox],
                "text": text,
                "confidence": round(float(conf), 4),
                "ocr_pass": source,
            })
        return items

    # ── Pass 1: preprocessed ─────────────────────────────────────────────────
    pass1 = _parse_results(paddle.ocr(proc_img, cls=True), "preprocessed")

    # ── Pass 2: raw image — add items not already covered by pass 1 ──────────
    pass2_candidates = _parse_results(paddle.ocr(raw_img, cls=True), "raw")
    extra = [it for it in pass2_candidates if not _bbox_overlaps_any(it["bbox"], pass1)]

    all_items = pass1 + extra

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered_items = []
    drop_reasons: dict[str, int] = {"figure": 0, "low_conf": 0, "invalid_chars": 0}

    for item in all_items:
        text = item["text"]
        # fix 4: layout filter
        if bbox_overlaps_figure(item["bbox"], figure_rects):
            drop_reasons["figure"] += 1
            continue
        # fix 1: confidence filter
        if item["confidence"] < CONF_THRESHOLD:
            drop_reasons["low_conf"] += 1
            continue
        # fix 2: character validity filter
        if not is_valid_text(text):
            drop_reasons["invalid_chars"] += 1
            continue

        filtered_items.append(item)

    raw_text      = " ".join(it["text"] for it in all_items)
    filtered_text = sort_and_join_items(filtered_items)

    return {
        "raw_text":      raw_text,
        "filtered_text": filtered_text,
        "all_items":     all_items,
        "filtered_items": filtered_items,
        "item_count":    len(all_items),
        "filtered_count": len(filtered_items),
        "dropped":       drop_reasons,
        "figure_regions": figure_rects,
    }


def sort_and_join_items(items: list[dict]) -> str:
    """
    Sort OCR items into reading order (top→bottom, left→right) and reconstruct
    text with line breaks (\n) and paragraph breaks (\n\n) based on Y gaps.
    Items whose Y-centers are within 0.7× avg line height are grouped as one line.
    A gap > 1.5× avg line height between consecutive lines becomes a paragraph break.
    """
    if not items:
        return ""

    def y_center(it):
        ys = [p[1] for p in it["bbox"]]
        return (min(ys) + max(ys)) / 2

    def y_min(it):  return min(p[1] for p in it["bbox"])
    def y_max(it):  return max(p[1] for p in it["bbox"])
    def x_left(it): return min(p[0] for p in it["bbox"])
    def item_h(it): return max(y_max(it) - y_min(it), 1)

    sorted_items = sorted(items, key=y_center)
    avg_h = sum(item_h(it) for it in sorted_items) / len(sorted_items)

    # Group into visual lines
    lines: list[list[dict]] = []
    current: list[dict] = [sorted_items[0]]
    for item in sorted_items[1:]:
        line_yc_max = max(y_center(it) for it in current)
        if y_center(item) - line_yc_max <= avg_h * 0.7:
            current.append(item)
        else:
            lines.append(current)
            current = [item]
    lines.append(current)

    # Sort each line left→right
    for line in lines:
        line.sort(key=x_left)

    # Group lines into paragraphs by Y gap
    # Small gap (≤1.5×avg_h) → same paragraph; large gap → new paragraph.
    # Within a paragraph items are concatenated with "" so kiwi sees whole words
    # (e.g. "기" + "하" = "기하", not "기\n하").  Paragraphs are separated by \n\n.
    paragraphs: list[list[list[dict]]] = [[lines[0]]]
    for i in range(1, len(lines)):
        prev_y_max = max(y_max(it) for it in lines[i - 1])
        curr_y_min = min(y_min(it) for it in lines[i])
        gap = curr_y_min - prev_y_max
        if gap > avg_h * 1.5:
            paragraphs.append([lines[i]])
        else:
            paragraphs[-1].append(lines[i])

    para_texts: list[str] = []
    for para in paragraphs:
        # Flatten to reading-order items and join without separator.
        # All word-boundary spacing is left to kiwi.
        all_items = [it for line in para for it in line]
        para_texts.append("".join(it["text"] for it in all_items))

    return "\n\n".join(para_texts)


def fix_mixed_lang_misreads(text: str) -> str:
    """
    Fix 5b: Replace Korean syllables that are flanked by ASCII letters on BOTH sides.
    These are visual misreads by the Korean OCR model, e.g. L→니 in 'SO니IDwORKS'.
    Korean particles legitimately attached at the END of an ASCII token (e.g.
    'SOLIDWORKS에', 'RAM으로') are NOT touched because they have no ASCII after them.

    Known misread map (PP-OCRv4 Korean model, visually similar shapes):
      니 → L   (ㄴ+ㅣ looks like uppercase L)
      이 → I   (이 and I are very similar in some fonts)
      ㅣ → I   (vertical bar)
    """
    # Map: Korean syllable → Latin equivalent when surrounded by ASCII letters
    _MISREAD = {'니': 'L', '이': 'I', 'ㅣ': 'I'}

    def _fix_token(tok: str) -> str:
        has_ascii = bool(re.search(r'[A-Za-z]', tok))
        has_kr    = bool(re.search(r'[\uAC00-\uD7A3\u3130-\u318F]', tok))
        if not (has_ascii and has_kr):
            return tok
        chars = list(tok)
        for i, ch in enumerate(chars):
            if ch in _MISREAD:
                prev_ascii = i > 0 and chars[i - 1].isascii() and chars[i - 1].isalpha()
                next_ascii = i < len(chars) - 1 and chars[i + 1].isascii() and chars[i + 1].isalpha()
                if prev_ascii and next_ascii:           # flanked on both sides → misread
                    chars[i] = _MISREAD[ch]
        return "".join(chars)

    # Tokenise preserving whitespace so we only inspect non-space runs
    return "".join(_fix_token(part) for part in re.split(r'(\s+)', text))


def apply_kiwi_spacing(kiwi: Kiwi, text: str) -> str:
    """Apply kiwipiepy spacing correction paragraph-by-paragraph.
    Splitting by \\n\\n preserves paragraph structure while letting kiwi
    see each paragraph as a whole (fixes cross-line word splits)."""
    if not text.strip():
        return text
    paragraphs = text.split("\n\n")
    return "\n\n".join(kiwi.space(p) if p.strip() else p for p in paragraphs)


# ── Path B: Vision LLM ────────────────────────────────────────────────────────

VLM_PROMPT = (
    "이 이미지에서 모든 텍스트를 정확하게 추출해 주세요. "
    "한국어와 영어를 구분하여 원문 그대로 추출하고, "
    "줄바꿈과 단락 구조를 최대한 유지해 주세요. "
    "텍스트 외의 설명은 추가하지 마세요."
)


def ocr_vlm(client: OpenAI, b64_jpeg: str) -> dict:
    """Send page image to Qwen2.5-VL and return extracted text + avg confidence."""
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
                },
                {"type": "text", "text": VLM_PROMPT},
            ],
        }],
        max_tokens=4096,
        temperature=0.1,
        timeout=120,
        logprobs=True,
    )
    text = response.choices[0].message.content or ""
    lp_content = (response.choices[0].logprobs or None) and response.choices[0].logprobs.content
    if lp_content:
        avg_conf = round(sum(math.exp(t.logprob) for t in lp_content) / len(lp_content), 4)
    else:
        avg_conf = None
    return {"full_text": text.strip(), "model": VLM_MODEL, "avg_confidence": avg_conf}


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[dict]:
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [{"index": i, "text": t} for i, t in enumerate(splitter.split_text(text))]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing PaddleOCR (korean, PP-OCRv4) ...")
    paddle_ocr = PaddleOCR(lang="korean", use_angle_cls=True, show_log=False,
                           det_db_thresh=DET_DB_THRESH)

    print("Initializing kiwipiepy ...")
    kiwi = Kiwi()

    print(f"Connecting to VLM  {VLM_MODEL} @ {VLM_BASE_URL} ...")
    vlm_client = OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY)

    header = (f"{'Page':<6} | "
              f"{'PaddleOCR + fixes + Kiwi':^60} | "
              f"{'VLM (Qwen2.5-VL)':^38}")
    sub    = (f"{'':6} | "
              f"{'all':>5} {'filt':>5} {'fig':>4} {'lc':>4} {'gc':>4} "
              f"{'spaced':>7} {'chunks':>6} {'conf':>6} {'time(s)':>7} | "
              f"{'chars':>7} {'chunks':>6} {'conf':>6} {'time(s)':>7}")
    print(f"\n{header}")
    print(sub)
    print("-" * 115)

    all_pages = []
    for page_idx in PAGE_RANGE:
        page_num = page_idx + 1

        # ── Path A ──
        t0         = time.perf_counter()
        img        = render_page_as_array(PDF_PATH, page_idx)
        paddle_res  = ocr_paddle(paddle_ocr, img)
        fixed_text  = fix_mixed_lang_misreads(paddle_res["filtered_text"])
        spaced      = apply_kiwi_spacing(kiwi, fixed_text)
        a_chunks   = chunk_text(spaced)
        a_time     = round(time.perf_counter() - t0, 2)

        filt_items = paddle_res["filtered_items"]
        avg_conf   = (
            sum(i["confidence"] for i in filt_items) / len(filt_items)
            if filt_items else 0.0
        )
        dropped = paddle_res["dropped"]

        # ── Path B ──
        t0       = time.perf_counter()
        b64_jpeg = render_page_as_jpeg_b64(PDF_PATH, page_idx)
        vlm_res  = ocr_vlm(vlm_client, b64_jpeg)
        b_chunks = chunk_text(vlm_res["full_text"])
        b_time   = round(time.perf_counter() - t0, 2)

        page_result = {
            "source":      str(PDF_PATH),
            "page_number": page_num,
            "render_dpi":  RENDER_DPI,
            "paddle_kiwi": {
                "paddle_raw":      paddle_res["raw_text"],
                "filtered_text":   paddle_res["filtered_text"],
                "fixed_text":      fixed_text,
                "kiwi_spaced":     spaced,
                "all_items":       paddle_res["all_items"],
                "filtered_items":  filt_items,
                "item_count":      paddle_res["item_count"],
                "filtered_count":  paddle_res["filtered_count"],
                "dropped":         dropped,
                "figure_regions":  paddle_res["figure_regions"],
                "avg_confidence":  round(avg_conf, 4),
                "chunks":          a_chunks,
                "chunk_count":     len(a_chunks),
                "processing_time_s": a_time,
            },
            "vlm": {
                **vlm_res,
                "chunks":          b_chunks,
                "chunk_count":     len(b_chunks),
                "processing_time_s": b_time,
            },
        }
        all_pages.append(page_result)

        out_path = OUT_DIR / f"page{page_num}_parse_compare.json"
        out_path.write_text(
            json.dumps(page_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        vlm_conf_str = f"{vlm_res['avg_confidence']:>6.3f}" if vlm_res["avg_confidence"] is not None else f"{'N/A':>6}"
        print(f"p{page_num:<5} | "
              f"{paddle_res['item_count']:>5} {paddle_res['filtered_count']:>5} "
              f"{dropped['figure']:>4} {dropped['low_conf']:>4} {dropped['invalid_chars']:>4} "
              f"{len(spaced):>7} {len(a_chunks):>6} {avg_conf:>6.3f} {a_time:>7.2f} | "
              f"{len(vlm_res['full_text']):>7} {len(b_chunks):>6} {vlm_conf_str} {b_time:>7.2f}")

    summary_path = OUT_DIR / "pages26_30_parse_compare.json"
    summary_path.write_text(
        json.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nPer-page JSONs   : {OUT_DIR}/page{{26..30}}_parse_compare.json")
    print(f"Combined summary : {summary_path}")

# -*- coding: utf-8 -*-
"""
import_datasets.py
==================
Import 2 HuggingFace datasets vao thu muc plain_texts/ cua VietLawBot:

  1. tmquan/anle-toaan-gov-vn   -> An le TAND Toi cao (moi an le = 1 file)
  2. tmquan/phapdien-moj-gov-vn -> Phap dien Bo Tu phap (gom theo linh vuc)

Cach dung:
  python import_datasets.py                # Import toan bo
  python import_datasets.py --dry-run      # Chi dem, khong ghi file
  python import_datasets.py --skip-anle    # Bo qua dataset an le
  python import_datasets.py --skip-phapdien  # Bo qua dataset phap dien
  python import_datasets.py --force        # Ghi de file da ton tai
"""

import os
import re
import sys
import json
import argparse
import unicodedata

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -------------------------------------------------------
# CAU HINH
# -------------------------------------------------------
PLAIN_TEXTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plain_texts")
FIELDS_META_PATH = os.path.join(PLAIN_TEXTS_DIR, "fields.meta")

ANLE_DATASET = "tmquan/anle-toaan-gov-vn"
PHAPDIEN_DATASET = "tmquan/phapdien-moj-gov-vn"

# Map case_type / doc_type -> linh vuc tieng Viet
CASE_TYPE_TO_FIELD = {
    "hinh su": "Hinh su",
    "hình sự": "Hình sự",
    "criminal": "Hình sự",
    "dan su": "Dan su",
    "dân sự": "Dân sự",
    "civil": "Dân sự",
    "hanh chinh": "Hanh chinh",
    "hành chính": "Hành chính",
    "administrative": "Hành chính",
    "lao dong": "Lao dong",
    "lao động": "Lao động",
    "labor": "Lao động",
    "kinh doanh": "Kinh doanh - Thuong mai",
    "thương mại": "Kinh doanh - Thương mại",
    "commercial": "Kinh doanh - Thương mại",
    "hon nhan": "Hon nhan va gia dinh",
    "hôn nhân": "Hôn nhân và gia đình",
    "gia dinh": "Hon nhan va gia dinh",
    "gia đình": "Hôn nhân và gia đình",
    "family": "Hôn nhân và gia đình",
    "pha san": "Kinh doanh - Thuong mai",
    "phá sản": "Kinh doanh - Thương mại",
}


# -------------------------------------------------------
# TIEN ICH
# -------------------------------------------------------
def log(msg):
    """Print message, replacing unencodable chars."""
    print(msg)


def slugify(text: str, max_len: int = 80) -> str:
    """Chuyen chuoi thanh ten file an toan."""
    if not text:
        return "unknown"
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len]


def detect_field_from_case_type(case_type: str) -> str:
    """Nhan dien linh vuc tu case_type / doc_type cua an le."""
    if not case_type:
        return "An le"
    ct_lower = case_type.lower()
    for key, field in CASE_TYPE_TO_FIELD.items():
        if key in ct_lower:
            return field
    return "An le"


def load_fields_meta() -> dict:
    """Load file fields.meta hien tai (neu co)."""
    if os.path.exists(FIELDS_META_PATH):
        with open(FIELDS_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_fields_meta(fields_dict: dict):
    """Luu fields.meta."""
    with open(FIELDS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(fields_dict, f, ensure_ascii=False, indent=2)


def write_txt(path: str, content: str, dry_run: bool, force: bool) -> bool:
    """
    Ghi file .txt.
    Tra ve True neu file duoc ghi (hoac se duoc ghi trong dry_run).
    """
    if os.path.exists(path) and not force:
        return False  # Da ton tai, bo qua
    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return True


# -------------------------------------------------------
# IMPORT AN LE (anle-toaan-gov-vn)
# -------------------------------------------------------
def import_anle(dry_run: bool = False, force: bool = False) -> tuple:
    """
    Import dataset an le.
    Returns: (written_count, skipped_count, error_count)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log("[ERROR] Thieu thu vien 'datasets'. Chay: pip install datasets pyarrow")
        return 0, 0, 0

    log("\n" + "=" * 60)
    log(f"[AN LE] IMPORT: {ANLE_DATASET}")
    log("=" * 60)

    written = skipped = errors = 0
    fields_dict = load_fields_meta()

    try:
        log("[...] Dang load dataset (co the mat vai phut lan dau)...")
        ds = load_dataset(ANLE_DATASET, name="documents", split="train")
        total = len(ds)
        log(f"[OK] Da load {total:,} an le")
    except Exception as e:
        log(f"[ERROR] Loi load dataset an le: {e}")
        return 0, 0, 1

    for i, record in enumerate(ds):
        try:
            # Lay noi dung chinh
            content = record.get("markdown", "") or ""
            if not content.strip():
                content = record.get("principle_text", "") or ""
            if not content.strip():
                skipped += 1
                continue

            # Tao ten file
            doc_name = record.get("doc_name", "") or ""
            title = record.get("title", "") or ""
            precedent_num = record.get("precedent_number", "") or ""

            if precedent_num:
                slug = f"anle-so-{slugify(precedent_num.replace('/', '-'))}"
            elif doc_name:
                slug = f"anle-{slugify(doc_name)}"
            else:
                slug = f"anle-{slugify(title)}-{i}"

            filename = f"{slug}.txt"
            filepath = os.path.join(PLAIN_TEXTS_DIR, filename)

            # Tao header metadata cho file
            case_type = record.get("case_type", "") or ""
            doc_type = record.get("doc_type", "") or ""
            issuing_authority = record.get("issuing_authority", "") or ""
            issue_date = record.get("issue_date", "") or ""
            adopted_date = record.get("adopted_date", "") or ""

            header_lines = [f"# {title or doc_name}"]
            if precedent_num:
                header_lines.append(f"So an le: {precedent_num}")
            if adopted_date or issue_date:
                header_lines.append(f"Ngay ban hanh: {adopted_date or issue_date}")
            if issuing_authority:
                header_lines.append(f"Co quan: {issuing_authority}")
            if case_type:
                header_lines.append(f"Loai vu an: {case_type}")
            header_lines.append("")

            full_content = "\n".join(header_lines) + "\n" + content

            # Ghi file
            did_write = write_txt(filepath, full_content, dry_run, force)
            if did_write:
                field = detect_field_from_case_type(case_type or doc_type)
                fields_dict[filename] = field
                written += 1
                if written % 100 == 0:
                    log(f"  -> Da xu ly {written}/{total} an le...")
            else:
                skipped += 1

        except Exception as e:
            log(f"  [WARN] Loi record #{i}: {e}")
            errors += 1

    if not dry_run:
        save_fields_meta(fields_dict)

    return written, skipped, errors


# -------------------------------------------------------
# IMPORT PHAP DIEN (phapdien-moj-gov-vn)
# -------------------------------------------------------
def import_phapdien(dry_run: bool = False, force: bool = False) -> tuple:
    """
    Import dataset phap dien, gom cac dieu luat theo subject_title (linh vuc).
    Returns: (written_count, skipped_count, error_count)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log("[ERROR] Thieu thu vien 'datasets'. Chay: pip install datasets pyarrow")
        return 0, 0, 0

    log("\n" + "=" * 60)
    log(f"[PHAP DIEN] IMPORT: {PHAPDIEN_DATASET}")
    log("=" * 60)

    fields_dict = load_fields_meta()

    try:
        log("[...] Dang load dataset (co the mat vai phut lan dau)...")
        ds = load_dataset(PHAPDIEN_DATASET, name="articles", split="train")
        total = len(ds)
        log(f"[OK] Da load {total:,} dieu luat")
    except Exception as e:
        log(f"[ERROR] Loi load dataset phap dien: {e}")
        return 0, 0, 1

    # Gom articles theo subject_id
    log("[...] Dang gom dieu luat theo linh vuc...")
    subjects: dict = {}

    for i, record in enumerate(ds):
        try:
            content_text = record.get("content_text", "") or ""
            if not content_text.strip():
                continue

            subject_id = record.get("subject_id", "") or f"unknown_{i}"
            subject_title = record.get("subject_title", "") or "Khac"
            topic_title = record.get("topic_title", "") or ""
            article_title = record.get("article_title", "") or ""
            chapter_title = record.get("chapter_title", "") or ""
            source_note = record.get("source_note_text", "") or ""

            if subject_id not in subjects:
                subjects[subject_id] = {
                    "subject_title": subject_title,
                    "topic_title": topic_title,
                    "articles": []
                }

            # Tao entry cho moi dieu luat
            article_header = []
            if chapter_title:
                article_header.append(f"## {chapter_title}")
            if article_title:
                article_header.append(f"### {article_title}")
            if source_note:
                article_header.append(f"Nguon: {source_note}")

            article_entry = "\n".join(article_header) + "\n\n" + content_text.strip()
            subjects[subject_id]["articles"].append(article_entry)

            if (i + 1) % 5000 == 0:
                log(f"  -> Da gom {i+1:,}/{total:,} dieu luat ({len(subjects):,} linh vuc)...")

        except Exception as e:
            log(f"  [WARN] Loi record #{i}: {e}")

    log(f"[OK] Tong cong: {len(subjects):,} linh vuc phap ly")

    # Ghi file cho tung subject
    written = skipped = errors = 0
    total_subjects = len(subjects)

    for j, (subject_id, data) in enumerate(subjects.items()):
        try:
            subject_title = data["subject_title"]
            topic_title = data["topic_title"]
            articles = data["articles"]

            slug = slugify(subject_id or subject_title)
            filename = f"phapdien-{slug}.txt"
            filepath = os.path.join(PLAIN_TEXTS_DIR, filename)

            # Tao noi dung file
            header_parts = [
                f"# {subject_title}",
                f"Chu de: {topic_title}" if topic_title else "",
                f"Linh vuc: {subject_title}",
                f"So dieu luat: {len(articles)}",
                "",
                "---",
                ""
            ]
            full_content = (
                "\n".join(h for h in header_parts if h is not None)
                + "\n\n"
                + "\n\n---\n\n".join(articles)
            )

            did_write = write_txt(filepath, full_content, dry_run, force)
            if did_write:
                field = topic_title or subject_title or "Phap dien"
                fields_dict[filename] = field
                written += 1
            else:
                skipped += 1

            if (j + 1) % 50 == 0:
                log(f"  -> Da ghi {written}/{total_subjects} linh vuc...")

        except Exception as e:
            log(f"  [WARN] Loi subject '{subject_id}': {e}")
            errors += 1

    if not dry_run:
        save_fields_meta(fields_dict)

    return written, skipped, errors


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Import HuggingFace datasets vao plain_texts/ cua VietLawBot"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Chi dem, khong ghi file thuc su")
    parser.add_argument("--skip-anle", action="store_true",
                        help="Bo qua dataset an le")
    parser.add_argument("--skip-phapdien", action="store_true",
                        help="Bo qua dataset phap dien")
    parser.add_argument("--force", action="store_true",
                        help="Ghi de file da ton tai (mac dinh bo qua)")
    args = parser.parse_args()

    # Tao thu muc output
    if not args.dry_run:
        os.makedirs(PLAIN_TEXTS_DIR, exist_ok=True)

    mode = "[DRY RUN] " if args.dry_run else ""
    log(f"\n[START] {mode}VietLawBot Dataset Importer")
    log(f"[DIR]   Output : {PLAIN_TEXTS_DIR}")
    log(f"[OPT]   Force  : {'Yes' if args.force else 'No'}")

    total_written = total_skipped = total_errors = 0

    # --- An le ---
    if not args.skip_anle:
        w, s, e = import_anle(dry_run=args.dry_run, force=args.force)
        total_written += w
        total_skipped += s
        total_errors += e
        log(f"\n[RESULT] An le: ghi {w} file | bo qua {s} | loi {e}")
    else:
        log("\n[SKIP] Bo qua dataset an le")

    # --- Phap dien ---
    if not args.skip_phapdien:
        w, s, e = import_phapdien(dry_run=args.dry_run, force=args.force)
        total_written += w
        total_skipped += s
        total_errors += e
        log(f"\n[RESULT] Phap dien: ghi {w} file | bo qua {s} | loi {e}")
    else:
        log("\n[SKIP] Bo qua dataset phap dien")

    # Tong ket
    log("\n" + "=" * 60)
    log(f"[DONE] HOAN TAT {('(DRY RUN)' if args.dry_run else '')}")
    log(f"   Tong file ghi : {total_written}")
    log(f"   Tong bo qua   : {total_skipped}")
    log(f"   Tong loi      : {total_errors}")
    log("=" * 60)

    if not args.dry_run and total_written > 0:
        log("\n[NEXT] Buoc tiep theo: Chay chunk_and_index.py de re-index vao Redis")
        log("   python chunk_and_index.py")


if __name__ == "__main__":
    main()

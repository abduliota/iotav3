"""
ocr_pdf.py — Convert scanned PDF to searchable text, then ingest to Supabase
Usage: python ocr_pdf.py --file "PDFs/Newfile/SAMA_EN_4066_VER1.pdf" --name "SAMA EN 4066 VER1" --source SAMA
"""
import argparse
import sys
from pathlib import Path

def ocr_and_ingest(file_path: str, doc_name: str, source: str):
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        print("ERROR: Run: pip install pytesseract pdf2image pillow")
        sys.exit(1)

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    print(f"[ocr] Converting PDF pages to images...")
    images = convert_from_path(str(pdf_path), dpi=300)
    print(f"[ocr] {len(images)} pages found. Running OCR...")

    # Import scraper helpers
    from scraper import chunk_page, upsert_document, insert_chunks, print_summary

    all_pages = []
    for i, img in enumerate(images, 1):
        print(f"[ocr] Page {i}/{len(images)}...", end="\r")
        # Try English + Arabic
        text = pytesseract.image_to_string(img, lang="eng+ara", config="--psm 3")
        text = text.strip()
        if len(text) >= 30:
            all_pages.append({"page": i, "text": text, "total_pages": len(images)})

    print(f"\n[ocr] Extracted text from {len(all_pages)}/{len(images)} pages.")

    if not all_pages:
        print("ERROR: OCR produced no text. Check Tesseract installation.")
        sys.exit(1)

    all_chunks = []
    for p in all_pages:
        all_chunks.extend(chunk_page(p["text"], p["page"], doc_name))

    print(f"[ocr] {len(all_chunks)} chunks produced. Uploading to Supabase...")

    doc_id   = upsert_document(doc_name, source, len(images))
    inserted = insert_chunks(doc_id, all_chunks, "ocr")

    print_summary([{
        "status": "ok",
        "document_name": doc_name,
        "total_pages": len(images),
        "chunks_inserted": inserted,
    }])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",   required=True)
    parser.add_argument("--name",   required=True)
    parser.add_argument("--source", default="SAMA")
    args = parser.parse_args()
    ocr_and_ingest(args.file, args.name, args.source)
import os

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")

    doc = fitz.open(pdf_path)

    pages_data = []

    for page_num, page in enumerate(doc):

        text = page.get_text()

        if text.strip():

            pages_data.append({
                "source": os.path.basename(pdf_path),
                "page": page_num + 1,
                "text": text.strip()
            })

    return pages_data
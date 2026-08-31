from pathlib import Path

def load_document(file_path: Path) -> str:
    """
    Extracts raw text content from TXT, PDF, or DOCX documents.
    """
    suffix = file_path.suffix.lower()
    content = ""
    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    elif suffix == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        content = "\n".join(pages)
    elif suffix == ".docx":
        import docx
        doc = docx.Document(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text:
                paragraphs.append(para.text)
        content = "\n".join(paragraphs)
    else:
        print(f"Warning: Unsupported file format {suffix} for {file_path}")
    return content

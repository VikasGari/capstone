import re
from pathlib import Path
from src.helpers.document_loader import load_document

class DocumentParser:
    """
    Parses diverse corpus documents (txt, pdf, docx).
    Extracts global metadata headers and segments content into clause-level blocks.
    Falls back gracefully to paragraph-level chunking for unstructured prose.
    """
    def _extract_document_metadata(self, content: str, file_path: Path) -> tuple[str, str, str]:
        """Extracts document ID, title, and type from text headers or path fallbacks."""
        doc_id_match = re.search(r"DOCUMENT ID:\s*(.+)", content, re.IGNORECASE)
        doc_title_match = re.search(r"DOCUMENT TITLE:\s*(.+)", content, re.IGNORECASE)
        category_match = re.search(r"CATEGORY:\s*(.+)", content, re.IGNORECASE)
        
        doc_id = doc_id_match.group(1).strip() if doc_id_match else file_path.stem.upper()
        doc_title = doc_title_match.group(1).strip() if doc_title_match else file_path.stem.replace("_", " ").title()
        
        doc_type = "General"
        if category_match:
            doc_type = category_match.group(1).strip()
        elif file_path.parent and file_path.parent.name:
            parent_name = file_path.parent.name
            doc_type = parent_name.replace("_", " ").title()
            
        return doc_id, doc_title, doc_type

    def _create_segment(
        self,
        doc_id: str,
        doc_title: str,
        doc_type: str,
        clause_id: str,
        clause_title: str,
        text: str,
        prefix_context: bool = True
    ) -> dict:
        """Constructs a normalized segment dictionary."""
        formatted_text = f"[{doc_title} - {clause_id} {clause_title}] {text}" if prefix_context else text
        return {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "doc_type": doc_type,
            "clause_id": clause_id,
            "clause_title": clause_title or "General",
            "text": formatted_text
        }

    def parse(self, file_path: Path) -> list[dict]:
        """
        Parses a file and returns normalized clause-level segment dictionaries.
        """
        content = load_document(file_path)
        if not content:
            return []
            
        doc_id, doc_title, doc_type = self._extract_document_metadata(content, file_path)
        
        # Split body content from header section using repeating symbol dividers
        divider_pattern = r"\n[=\-_*]{10,}\n"
        parts = re.split(divider_pattern, content)
        body = parts[1].strip() if len(parts) > 1 else content
        
        segments = []
        
        # Detect headings at the beginning of a line (Clause/Section/Rule X.Y or numeric headings X.Y)
        heading_pattern = r"(?:^|\n)(Clause\s+\d+(?:\.\d+)*[:\.]?|Section\s+\d+(?:\.\d+)*[:\.]?|Rule\s+\d+(?:\.\d+)*[:\.]?|\d+\.\d+[:\.-]?)(?=\s|\n|$)"
        splits = re.split(heading_pattern, body)
        
        # If no clause headers were detected, segment by paragraphs as unstructured fallback
        if len(splits) <= 1:
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            for p_idx, para in enumerate(paragraphs):
                block_id = f"Block {p_idx+1}"
                block_title = para[:50].strip() if len(para) > 50 else para.strip()
                segments.append(
                    self._create_segment(doc_id, doc_title, doc_type, block_id, block_title, para)
                )
            if not segments:
                segments.append(
                    self._create_segment(doc_id, doc_title, doc_type, "General", "General", body.strip(), prefix_context=False)
                )
            return segments
            
        # Capture any text before the first clause header (like introductory paragraphs)
        intro_text = splits[0].strip()
        if intro_text:
            segments.append(
                self._create_segment(doc_id, doc_title, doc_type, "Introduction", "Introduction", intro_text, prefix_context=False)
            )
            
        # Match each clause header to its following block of text
        for i in range(1, len(splits), 2):
            clause_header = splits[i].strip().rstrip(":.")
            clause_body = splits[i+1].strip() if i+1 < len(splits) else ""
            
            lines = [l.strip() for l in clause_body.split("\n") if l.strip()]
            clause_title = "General"
            clause_text = clause_body
            
            if lines:
                clause_title = lines[0]
                clause_text = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
                
            segments.append(
                self._create_segment(doc_id, doc_title, doc_type, clause_header, clause_title, clause_text)
            )
            
        return segments

import os
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.config_manager import ConfigManager

class IngestionPipeline:
    """
    Modular ingestion pipeline.
    Parses diverse policy documents, segments them logically into clauses/sections,
    pre-chunks them with titles, generates local embeddings, and stores them in FAISS.
    General-purpose: adapts to unstructured text or different structural schemas.
    """
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Fetch configuration sections directly from global config
        embedding_cfg = self.config_manager.get_section("embedding")
        vstore_cfg = self.config_manager.get_section("vector_store")
        chunking_cfg = self.config_manager.get_section("chunking")
        paths_cfg = self.config_manager.get_section("paths")
        
        # Extract configurations directly into distinct properties
        self.model_name = embedding_cfg.get("model_name")
        self.persist_directory = vstore_cfg.get("persist_directory")
        self.chunk_size = chunking_cfg.get("chunk_size")
        self.chunk_overlap = chunking_cfg.get("chunk_overlap")
        self.corpus_directory = paths_cfg.get("corpus_directory")

    def parse_file(self, file_path: Path) -> list[dict]:
        """
        Parses a corpus document. Supports txt, pdf, and docx.
        Extracts global document headers (with directory/filename fallbacks) and
        breaks the document content into clause-level segments using multi-heading rules.
        """
        suffix = file_path.suffix.lower()
        content = ""
        try:
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
                print(f"Warning: Unsupported file type {suffix} for {file_path}")
                return []
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
            
        # Extract headers using case-insensitive regex search
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
                # Derive a meaningful title from first 50 chars of paragraph
                block_title = para[:50].strip() if len(para) > 50 else para.strip()
                segments.append({
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "doc_type": doc_type,
                    "clause_id": block_id,
                    "clause_title": block_title or "General",
                    "text": f"[{doc_title} - {block_id} {block_title}] {para}"
                })
            # Fallback if text has no double newlines
            if not segments:
                segments.append({
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "doc_type": doc_type,
                    "clause_id": "General",
                    "clause_title": "General",
                    "text": body.strip()
                })
            return segments
            
        # Capture any text before the first clause header (like introductory paragraphs)
        intro_text = splits[0].strip()
        if intro_text:
            segments.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_type": doc_type,
                "clause_id": "Introduction",
                "clause_title": "Introduction",
                "text": intro_text
            })
            
        # Match each clause header to its following block of text
        for i in range(1, len(splits), 2):
            clause_header = splits[i].strip().rstrip(":.")  # e.g., "Clause 1.1" or "1.1"
            clause_body = splits[i+1].strip() if i+1 < len(splits) else ""
            
            lines = [l.strip() for l in clause_body.split("\n") if l.strip()]
            clause_title = "General"
            clause_text = clause_body
            
            if lines:
                clause_title = lines[0]
                clause_text = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
                
            segments.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_type": doc_type,
                "clause_id": clause_header,
                "clause_title": clause_title,
                "text": f"[{doc_title} - {clause_header} {clause_title}] {clause_text}"
            })
            
        return segments

    def run(self, corpus_dir: str = None) -> int:
        """
        Runs the ingestion pipeline. Loads all files, chunks them, computes 
        embeddings, and populates the FAISS database.
        Idempotent: Clears any existing FAISS files.
        """
        if corpus_dir is None:
            corpus_dir = self.corpus_directory
            
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus directory not found at: {corpus_path}")
            
        all_segments = []
        extensions = ["*.txt", "*.pdf", "*.docx"]
        file_paths = []
        for ext in extensions:
            file_paths.extend(corpus_path.glob(ext))
            
        for file_path in file_paths:
            segments = self.parse_file(file_path)
            for seg in segments:
                seg["source"] = file_path.name
            all_segments.extend(segments)
            
        if not all_segments:
            print("Warning: No documents parsed. Ingestion aborted.")
            return 0
            
        # Recursive splitter to split clause texts if they exceed chunk_size
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.chunk_size),
            chunk_overlap=int(self.chunk_overlap)
        )
        
        final_documents = []
        final_metadatas = []
        final_ids = []
        
        for seg in all_segments:
            chunks = splitter.split_text(seg["text"])
            for sub_idx, chunk in enumerate(chunks):
                final_documents.append(chunk)
                # Keep matching metadata schema
                metadata = {
                    "source": seg["source"],
                    "doc_id": seg["doc_id"],
                    "doc_title": seg["doc_title"],
                    "doc_type": seg["doc_type"],
                    "clause_id": seg["clause_id"],
                    "clause_title": seg["clause_title"],
                    "id": f"{seg['doc_id']}_{seg['clause_id']}_{sub_idx}"  # Store ID in metadata for FAISS mapping
                }
                final_metadatas.append(metadata)
                final_ids.append(metadata["id"])
                
        # Initialize persistent folder path
        persist_dir = self.persist_directory
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Recreate directory files to ensure idempotence
        for filename in ["index.faiss", "index.pkl"]:
            file_path = persist_path / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not clear existing index file {file_path}: {e}")
        
        # Load local embedding model via LangChain Wrapper
        model_name = self.model_name
        print(f"Loading embedding model: {model_name}...")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize and populate FAISS index
        print(f"Building FAISS vector index on {len(final_documents)} chunks...")
        db = FAISS.from_texts(
            texts=final_documents,
            embedding=embeddings,
            metadatas=final_metadatas,
            ids=final_ids
        )
        
        # Persist index to disk
        db.save_local(persist_dir)
        print(f"Ingested {len(final_documents)} chunks from {len(all_segments)} clauses into FAISS (Persisted at: {persist_dir})")
        
        return len(final_documents)

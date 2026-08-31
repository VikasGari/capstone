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
    Parses synthetic policy documents, segments them logically into clauses,
    pre-chunks them with titles, generates local embeddings, and stores them in FAISS.
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
        Parses a corpus document. Extracts global document headers and 
        breaks the document content into clause-level segments.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
            
        # Extract headers using regex
        doc_id_match = re.search(r"DOCUMENT ID:\s*(.+)", content)
        doc_title_match = re.search(r"DOCUMENT TITLE:\s*(.+)", content)
        category_match = re.search(r"CATEGORY:\s*(.+)", content)
        
        doc_id = doc_id_match.group(1).strip() if doc_id_match else file_path.stem
        doc_title = doc_title_match.group(1).strip() if doc_title_match else file_path.stem
        doc_type = category_match.group(1).strip() if category_match else "General"
        
        # Split body content from header section using the divider
        parts = content.split("="*40)
        body = parts[1].strip() if len(parts) > 1 else content
        
        segments = []
        
        # Split body content by 'Clause X.Y:' or 'Section X.Y:' patterns
        pattern = r"(Clause\s+\d+\.\d+:|Section\s+\d+\.\d+:)"
        splits = re.split(pattern, body)
        
        # If no clause headers were detected, treat the entire body as a single segment
        if len(splits) <= 1:
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
            clause_header = splits[i].strip()  # e.g., "Clause 1.1:"
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
                "clause_id": clause_header.rstrip(":"),
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
        for file_path in corpus_path.glob("*.txt"):
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

import os
import re
from pathlib import Path
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config.config_manager import ConfigManager

class IngestionPipeline:
    """
    Modular ingestion pipeline.
    Parses synthetic policy documents, segments them logically into clauses,
    pre-chunks them with titles, generates local embeddings, and stores them in Chroma.
    """
    def __init__(self, config_manager: ConfigManager = None, local_overrides: dict = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Fetch configuration sections directly from global config
        embedding_cfg = self.config_manager.get_section("embedding")
        vstore_cfg = self.config_manager.get_section("vector_store")
        chunking_cfg = self.config_manager.get_section("chunking")
        paths_cfg = self.config_manager.get_section("paths")
        
        # Extract configurations directly into distinct properties (no self.config dict lookup)
        self.model_name = local_overrides.get("model_name") if local_overrides and "model_name" in local_overrides else embedding_cfg.get("model_name")
        self.persist_directory = local_overrides.get("persist_directory") if local_overrides and "persist_directory" in local_overrides else vstore_cfg.get("persist_directory")
        self.collection_name = local_overrides.get("collection_name") if local_overrides and "collection_name" in local_overrides else vstore_cfg.get("collection_name")
        self.chunk_size = local_overrides.get("chunk_size") if local_overrides and "chunk_size" in local_overrides else chunking_cfg.get("chunk_size")
        self.chunk_overlap = local_overrides.get("chunk_overlap") if local_overrides and "chunk_overlap" in local_overrides else chunking_cfg.get("chunk_overlap")
        self.corpus_directory = local_overrides.get("corpus_directory") if local_overrides and "corpus_directory" in local_overrides else paths_cfg.get("corpus_directory")

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
        embeddings, and populates the Chroma database.
        Idempotent: Clears any existing collection data.
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
                metadata = {
                    "source": seg["source"],
                    "doc_id": seg["doc_id"],
                    "doc_title": seg["doc_title"],
                    "doc_type": seg["doc_type"],
                    "clause_id": seg["clause_id"],
                    "clause_title": seg["clause_title"]
                }
                final_metadatas.append(metadata)
                final_ids.append(f"{seg['doc_id']}_{seg['clause_id']}_{sub_idx}")
                
        # Initialize persistent Chroma client
        persist_dir = self.persist_directory
        client = chromadb.PersistentClient(path=persist_dir)
        
        # Load local embedding model
        model_name = self.model_name
        print(f"Loading embedding model: {model_name}...")
        embedding_model = SentenceTransformer(model_name)
        
        # Wrapper for Chroma DB embedding API
        class LocalEmbedder:
            def __init__(self, model):
                self.model = model
            def __call__(self, input):
                return self.model.encode(input, convert_to_numpy=True).tolist()
                
        embed_fn = LocalEmbedder(embedding_model)
        
        # Recreate collection to ensure idempotence
        collection_name = self.collection_name
        try:
            client.delete_collection(collection_name)
            print(f"Cleared existing collection '{collection_name}' for clean re-ingestion.")
        except Exception:
            pass
            
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embed_fn
        )
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(final_documents), batch_size):
            end = i + batch_size
            collection.add(
                documents=final_documents[i:end],
                metadatas=final_metadatas[i:end],
                ids=final_ids[i:end]
            )
            
        print(f"Ingested {len(final_documents)} chunks from {len(all_segments)} clauses into '{collection_name}' (Persisted at: {persist_dir})")
        
        return len(final_documents)

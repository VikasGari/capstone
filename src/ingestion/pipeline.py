from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.config_manager import ConfigManager
from src.ingestion.parser import DocumentParser

class IngestionPipeline:
    """
    Modular ingestion pipeline.
    Coordinates document parsing, recursive text chunking, local embedding computation,
    and persistent FAISS vector indexing.
    """
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.parser = DocumentParser()
        
        # Load configuration sections from global config
        embedding_cfg = self.config_manager.get_section("embedding")
        vstore_cfg = self.config_manager.get_section("vector_store")
        chunking_cfg = self.config_manager.get_section("chunking")
        paths_cfg = self.config_manager.get_section("paths")
        
        # Extract properties
        self.model_name = embedding_cfg.get("model_name")
        self.persist_directory = vstore_cfg.get("persist_directory")
        self.chunk_size = chunking_cfg.get("chunk_size")
        self.chunk_overlap = chunking_cfg.get("chunk_overlap")
        self.corpus_directory = paths_cfg.get("corpus_directory")

    def parse_file(self, file_path: Path) -> list[dict]:
        """Delegates file parsing to the DocumentParser subcomponent."""
        return self.parser.parse(file_path)

    def _clear_existing_index(self, persist_path: Path):
        """Cleans up previous FAISS index files to ensure idempotence."""
        persist_path.mkdir(parents=True, exist_ok=True)
        for filename in ["index.faiss", "index.pkl"]:
            file_path = persist_path / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not clear existing index file {file_path}: {e}")

    def run(self, corpus_dir: str = None) -> int:
        """
        Runs the ingestion pipeline. Loads all files, chunks them, computes 
        embeddings, and populates the FAISS database.
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
                metadata = {
                    "source": seg["source"],
                    "doc_id": seg["doc_id"],
                    "doc_title": seg["doc_title"],
                    "doc_type": seg["doc_type"],
                    "clause_id": seg["clause_id"],
                    "clause_title": seg["clause_title"],
                    "id": f"{seg['doc_id']}_{seg['clause_id']}_{sub_idx}"
                }
                final_metadatas.append(metadata)
                final_ids.append(metadata["id"])
                
        persist_path = Path(self.persist_directory)
        self._clear_existing_index(persist_path)
        
        # Load local embedding model via LangChain Wrapper
        print(f"Loading embedding model: {self.model_name}...")
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        
        # Initialize and populate FAISS index
        print(f"Building FAISS vector index on {len(final_documents)} chunks...")
        db = FAISS.from_texts(
            texts=final_documents,
            embedding=embeddings,
            metadatas=final_metadatas,
            ids=final_ids
        )
        
        db.save_local(self.persist_directory)
        print(f"Ingested {len(final_documents)} chunks from {len(all_segments)} clauses into FAISS (Persisted at: {self.persist_directory})")
        
        return len(final_documents)

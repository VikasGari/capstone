import tempfile
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline

def test_parse_file():
    # Create a temporary file mimicking a corpus document
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write("DOCUMENT ID: EXCH_TEST_01\n")
        tmp.write("DOCUMENT TITLE: Test Exchange Title\n")
        tmp.write("CATEGORY: Test Category\n")
        tmp.write("="*40 + "\n")
        tmp.write("Clause 1.1: Normal Hours\n")
        tmp.write("This is the text for normal hours clause.\n\n")
        tmp.write("Clause 1.2: Post Hours\n")
        tmp.write("This is the text for post closing hours.\n")
        tmp_path = Path(tmp.name)

    try:
        pipeline = IngestionPipeline()
        segments = pipeline.parse_file(tmp_path)
        
        # Verify structure
        assert len(segments) == 2
        
        assert segments[0]["doc_id"] == "EXCH_TEST_01"
        assert segments[0]["doc_title"] == "Test Exchange Title"
        assert segments[0]["doc_type"] == "Test Category"
        assert segments[0]["clause_id"] == "Clause 1.1"
        assert segments[0]["clause_title"] == "Normal Hours"
        assert "This is the text for normal hours clause." in segments[0]["text"]
        
        assert segments[1]["clause_id"] == "Clause 1.2"
        assert segments[1]["clause_title"] == "Post Hours"
        assert "This is the text for post closing hours." in segments[1]["text"]
    finally:
        # Clean up temporary file
        tmp_path.unlink()

def test_ingestion_run():
    # Test that the pipeline can run over a temporary corpus
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a sample text file
        doc_file = temp_path / "test_doc.txt"
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write("DOCUMENT ID: EXCH_TEST_02\n")
            f.write("DOCUMENT TITLE: Another Test Title\n")
            f.write("CATEGORY: Test Category\n")
            f.write("="*40 + "\n")
            f.write("Clause 2.1: Simple Rule\n")
            f.write("This is a simple rule text to index.\n")
            
        # Run ingestion with temporary paths
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as persist_dir:
            local_overrides = {
                "persist_directory": persist_dir,
                "collection_name": "test_collection",
                "corpus_directory": temp_dir
            }
            pipeline = IngestionPipeline(local_overrides=local_overrides)
            num_chunks = pipeline.run()
            
            assert num_chunks == 1

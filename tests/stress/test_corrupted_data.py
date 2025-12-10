import pytest
from cubo.ingestion.document_loader import DocumentLoader
from cubo.utils.exceptions import FileAccessError

def test_corrupted_pdf_handling(tmp_path):
    """
    Stress Test: Uploading a file with PDF extension but garbage content.
    Expectation: Should raise a clean error or return empty, NOT crash the process.
    """
    loader = DocumentLoader(skip_model=True) # isolate from embedding model
    
    # Create corrupted file
    bad_pdf = tmp_path / "corrupted.pdf"
    with open(bad_pdf, "wb") as f:
        f.write(b"%PDF-1.4\n" + b"\x00" * 1024 + b"GARBAGE_DATA")
        
    print(f"\nUpdating corrupted PDF: {bad_pdf}")
    
    # This should handle the error gracefully
    try:
        docs = loader.load_documents_from_folder(str(tmp_path))
        # If it returns docs, they should be empty or handled
        print(f"Result: {docs}")
    except Exception as e:
        pytest.fail(f"Loader crashed on corrupted PDF: {e}")

def test_huge_file_limit(tmp_path):
    """
    Stress Test: Uploading a file exceeding size limits.
    """
    # config normally limits to 10MB
    # We create a 15MB dummy file
    huge_file = tmp_path / "huge_doc.txt"
    with open(huge_file, "wb") as f:
        f.seek(15 * 1024 * 1024)
        f.write(b"0")
        
    loader = DocumentLoader(skip_model=True)
    
    # Logic might log warning and skip, or raise exception.
    # We want to verify it doesn't try to read 15MB into RAM if we intended to block it.
    
    docs = loader.load_documents_from_folder(str(tmp_path))
    # Depending on implementation, this might return [] or fail validation
    # We just ensure no crash.

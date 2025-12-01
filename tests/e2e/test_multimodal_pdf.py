"""
Multimodal PDF Processing End-to-End Test

Tests PDF processing with tables and images:
1. Ingest PDF with tables (pdfplumber extraction)
2. Ingest PDF with images (OCR with Tesseract) - if available
3. Query for table/image content
4. Verify structured data in results
"""

import tempfile
from pathlib import Path

import pytest
import pdfplumber

from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.retrieval.retriever import DocumentRetriever
from cubo.embeddings.lazy_model_manager import get_lazy_model_manager


@pytest.fixture
def pdf_with_table(tmp_path):
    """
    Create a simple PDF with a table using reportlab.
    
    Note: In production, this would be a real scanned contract/invoice.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except ImportError:
        pytest.skip("reportlab not installed")
    
    pdf_path = tmp_path / "invoice_with_table.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Add title
    title = Paragraph("INVOICE #12345", styles['Title'])
    elements.append(title)
    
    # Add table
    data = [
        ['Item', 'Quantity', 'Unit Price', 'Total'],
        ['Software License', '10', '$100', '$1,000'],
        ['Support Contract', '1', '$500', '$500'],
        ['Training Session', '2', '$300', '$600'],
        ['', '', 'TOTAL', '$2,100']
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    doc.build(elements)
    
    return pdf_path


class TestMultimodalPDFProcessing:
    """Test PDF processing with tables and images."""
    
    def test_table_extraction_with_pdfplumber(self, pdf_with_table):
        """
        Test that tables in PDFs are extracted and queryable.
        
        This is critical for invoices, contracts, financial docs.
        """
        # Extract tables using pdfplumber
        with pdfplumber.open(pdf_with_table) as pdf:
            assert len(pdf.pages) > 0, "PDF has no pages"
            
            page = pdf.pages[0]
            tables = page.extract_tables()
            
            assert len(tables) > 0, "No tables extracted from PDF"
            
            # Verify table content
            table = tables[0]
            assert len(table) > 0, "Table is empty"
            
            # Check for expected content
            table_text = str(table).lower()
            assert "software license" in table_text, "Expected item not in table"
            assert "2,100" in table_text or "2100" in table_text, \
                "Total amount not found in table"
            
            print(f"✓ Extracted {len(tables)} tables from PDF")
            print(f"  Table dimensions: {len(table)} rows × {len(table[0])} cols")
    
    def test_pdf_table_ingestion_and_query(self, pdf_with_table, tmp_path):
        """
        E2E: Ingest PDF with table → Query for table data → Verify results
        """
        # 1. Ingest PDF with table extraction
        ingestor = DeepIngestor()
        
        # Process PDF (deep_ingestor should extract tables)
        chunks = ingestor.process_single_file(str(pdf_with_table))
        
        assert len(chunks) > 0, "No chunks created from PDF"
        
        # Verify table data is in chunks
        all_text = " ".join([chunk["text"] for chunk in chunks])
        assert "Software License" in all_text, "Table content not in chunks"
        assert "2,100" in all_text or "2100" in all_text, "Total not in chunks"
        
        print(f"✓ Ingested {len(chunks)} chunks from PDF with table")
        
        # 2. Build retriever
        model = get_lazy_model_manager().get_model()
        retriever = DocumentRetriever(model=model, top_k=3)
        
        for chunk in chunks:
            retriever.add_document(
                document=chunk["text"],
                metadata=chunk.get("metadata", {})
            )
        
        # 3. Query for table-specific information
        query = "What is the total invoice amount?"
        results = retriever.retrieve(query, top_k=3)
        
        assert len(results) > 0, "No results for table query"
        
        # Verify results contain table data
        results_text = " ".join([r["document"] for r in results])
        assert "2,100" in results_text or "2100" in results_text, \
            f"Total amount not found in query results: {results_text}"
        
        print(f"✓ Query '{query}' successfully retrieved table data")
        print(f"  Top result: {results[0]['document'][:100]}...")
    
    def test_structured_data_preservation(self, pdf_with_table):
        """
        Test that structured table data maintains relationships.
        
        E.g., "Software License" should be associated with "$1,000"
        """
        with pdfplumber.open(pdf_with_table) as pdf:
            tables = pdf.pages[0].extract_tables()
            table = tables[0]
            
            # Find row index for "Software License"
            license_row = None
            for i, row in enumerate(table):
                if "Software License" in str(row):
                    license_row = row
                    break
            
            assert license_row is not None, "Software License row not found"
            
            # Verify associated data in same row
            row_text = " ".join([str(cell) for cell in license_row if cell])
            assert "1,000" in row_text or "1000" in row_text, \
                "Price not associated with Software License"
            
            print(f"✓ Structured relationships preserved: {license_row}")
    
    @pytest.mark.skipif(
        not __import__('importlib.util').find_spec('pytesseract'),
        reason="Tesseract OCR not available"
    )
    def test_ocr_processing_placeholder(self):
        """
        Placeholder for OCR testing with Tesseract.
        
        When Tesseract is installed, this would test:
        - Extract text from scanned PDF images
        - Query for text from images
        - Verify OCR accuracy
        """
        # This would require:
        # 1. Create PDF with embedded image containing text
        # 2. Run OCR via pytesseract
        # 3. Query for image text
        # 4. Verify results
        
        pytest.skip("OCR testing requires Tesseract + sample scanned PDFs")
    
    def test_mixed_content_pdf_ingestion(self, pdf_with_table):
        """
        Test PDF with both text paragraphs AND tables.
        
        Verifies both content types are processed correctly.
        """
        # In a real scenario, PDF would have:
        # - Title/heading (text)
        # - Body paragraphs (text)
        # - Table (structured data)
        # - Footer (text)
        
        with pdfplumber.open(pdf_with_table) as pdf:
            page = pdf.pages[0]
            
            # Extract ALL text (including table)
            full_text = page.extract_text()
            assert "INVOICE" in full_text, "Title text not extracted"
            assert "Software License" in full_text, "Table text not extracted"
            
            # Extract tables separately
            tables = page.extract_tables()
            assert len(tables) > 0, "Tables not extracted"
            
            print("✓ Mixed content (text + tables) extracted successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

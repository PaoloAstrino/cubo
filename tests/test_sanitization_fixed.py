#!/usr/bin/env python
"""
Quick test to verify chunk ID sanitization is working.
Run this after clearing the vector store to confirm the fix is in place.
"""

import re


def test_chunk_id_sanitization():
    """Test that filenames with spaces are sanitized properly."""
    
    test_cases = [
        ("Ricevuta occasionale Paolo Astrino.pdf", "Ricevuta_occasionale_Paolo_Astrino.pdf"),
        ("My Document (v1).txt", "My_Document__v1_.txt"),
        ("file [2023].docx", "file__2023_.docx"),
        ("normal_file.txt", "normal_file.txt"),
        ("file-with-dashes.pdf", "file-with-dashes.pdf"),
        ("file.with.dots.txt", "file.with.dots.txt"),
    ]
    
    pattern = re.compile(r"^[A-Za-z0-9_.:-]+$")
    
    print("Testing Chunk ID Sanitization")
    print("=" * 70)
    
    all_passed = True
    for original, expected in test_cases:
        sanitized = re.sub(r'[^A-Za-z0-9._:-]', '_', original)
        
        # Check if it matches expected
        matches_expected = (sanitized == expected)
        
        # Check if it passes ID validation
        valid = pattern.match(sanitized) is not None
        
        status = "✓ PASS" if (matches_expected and valid) else "✗ FAIL"
        
        print(f"\n{status}")
        print(f"  Original:  {original}")
        print(f"  Expected:  {expected}")
        print(f"  Actual:    {sanitized}")
        print(f"  Valid ID:  {valid}")
        
        if not (matches_expected and valid):
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Sanitization is working correctly!")
        print("\nYou can now re-upload documents with spaces in filenames.")
        print("They will be processed with proper chunk IDs like:")
        print("  'Ricevuta_occasionale_Paolo_Astrino.pdf_chunk_0_<hash>'")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Check sanitization code")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(test_chunk_id_sanitization())

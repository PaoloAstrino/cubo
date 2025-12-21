"""Tests for hierarchical chunker logic to reproduce 'Greedy Merging' and 'Windowing' critiques."""
import pytest
from cubo.ingestion.hierarchical_chunker import HierarchicalChunker
from cubo.utils.utils import Utils

class TestHierarchicalChunkerCritique:
    """Reproduce failures in chunking logic."""

    def test_greedy_splitting_no_semantic_lookahead(self):
        """Test that _split_large_node splits greedily without looking for semantic breaks."""
        # Use max_chunk_size=40 to force split between paragraphs
        # Use min_chunk_size=0 to avoid dropping small chunks during test
        chunker = HierarchicalChunker(max_chunk_size=40, min_chunk_size=0, overlap_sentences=0)
        
        # Text with a clear semantic break (double newline) that should be respected
        # but won't be because of greedy filling.
        # Sentence 1: 20 chars
        # Sentence 2: 20 chars
        # Sentence 3: 20 chars
        # Max size 50.
        # Greedy: [S1, S2] (40 chars), [S3] (20 chars).
        # If S2 was "Topic B starts here...", maybe we wanted to split before S2?
        # Actually, the critique says: "It doesn't try to find the 'best' semantic breakpoint (like a double newline or a semicolon)."
        
        text = "Sentence A1. Sentence A2.\n\nSentence B1. Sentence B2."
        # Lengths:
        # "Sentence A1." = 12
        # "Sentence A2." = 12
        # "\n\n" = 2
        # "Sentence B1." = 12
        # "Sentence B2." = 12
        # Total ~50 chars.
        
        # If max_chunk_size is 25.
        # S1 (12) + S2 (12) = 24. Fits.
        # Next is \n\nS3...
        
        # Let's try to force a split that breaks a paragraph.
        # S1 (20), S2 (20), S3 (20). Max 30.
        # Chunk 1: S1.
        # Chunk 2: S2.
        # Chunk 3: S3.
        # This is fine.
        
        # What if S1(15), S2(15), S3(15). Max 35.
        # Chunk 1: S1 + S2 (30).
        # Chunk 2: S3.
        # If S2 was the start of a new paragraph, we merged it with S1.
        
        text = "Intro sentence.\n\nNew Topic sentence. Detail sentence."
        # "Intro sentence." (15)
        # "\n\n" (2)
        # "New Topic sentence." (19)
        # "Detail sentence." (16)
        
        # Max size 40.
        # S1 + \n\n + S2 = 15 + 2 + 19 = 36. Fits.
        # So it chunks: ["Intro sentence.\n\nNew Topic sentence."]
        # Then ["Detail sentence."]
        
        # Ideally, we should respect the \n\n and split there if possible, 
        # keeping "New Topic" with "Detail".
        # i.e. Chunk 1: "Intro sentence."
        # Chunk 2: "New Topic sentence. Detail sentence." (35 chars).
        
        # The greedy algorithm will merge S1 and S2 because they fit.
        
        chunks = chunker.chunk(text)
        # We expect the greedy failure:
        assert len(chunks) >= 2
        # With new logic, it should split at paragraph
        assert "Intro sentence." in chunks[0]['text']
        assert "New Topic sentence." not in chunks[0]['text']
        assert "New Topic sentence." in chunks[1]['text']

    def test_sentence_window_pointer_mode(self):
        """Test that sentence windowing can run in pointer mode (no duplication)."""
        text = "S1. S2. S3."
        chunks = Utils.create_sentence_window_chunks(text, window_size=3, add_window_text=False)
        
        assert len(chunks) == 3
        assert "window" not in chunks[0]
        assert chunks[0]["window_start"] == 0
        # Window for S1 (index 0) with size 3 (half 1): [0, 1] -> S1, S2.
        assert chunks[0]["window_end"] == 1 
        
        # Verify we can reconstruct if we had the sentences
        sentences = Utils._split_into_sentences(text)
        start = chunks[0]["window_start"]
        end = chunks[0]["window_end"] + 1
        window_text = " ".join(sentences[start:end])
        assert window_text == "S1. S2."


"""Tests for token counting and chunk sizing to reproduce 'Character-Based Limits' critique."""
import pytest
from cubo.utils.utils import Utils
from cubo.ingestion.hierarchical_chunker import HierarchicalChunker

class TestTokenChunkingCritique:
    """Reproduce failures in token counting and chunk sizing."""

    def test_token_count_discrepancy(self):
        """Test that character/word count is a poor proxy for token count."""
        # A string with long words has few words but many characters.
        # A string with special characters or unicode might have many tokens.
        
        # Example: "reproducibility" (1 word, ~15 chars, maybe 3-4 tokens)
        text = "reproducibility " * 100
        # 100 words. 1600 characters.
        # Tokens: "reproducibility" is often multiple tokens in BPE.
        # Let's assume we want to verify that Utils._token_count is inaccurate compared to a real tokenizer.
        # But we can't easily assert "inaccuracy" without a real tokenizer to compare to.
        # The critique says: "Utils._token_count defaults to len(text.split())"
        
        count = Utils._token_count(text)
        assert count == 100 # It just counts spaces/words
        
        # If we had a real tokenizer, it would be much higher.
        # We will assert that we CANNOT pass a tokenizer currently or it's not used correctly in defaults.
        # Actually, the critique says "Modern LLM tokenizers ... do NOT map 1:1 to words".
        
        # We want to assert that the system allows us to use a real tokenizer and respects it.
        # If I pass a mock tokenizer, does it work?
        
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Pretend everything is 2 tokens per word
                return [1, 2] * len(text.split())
        
        mock_tokenizer = MockTokenizer()
        count_with_tokenizer = Utils._token_count(text, tokenizer=mock_tokenizer)
        assert count_with_tokenizer == 200
        
        # This part actually works in the current code (I saw it in read_file).
        # The problem is that HierarchicalChunker DOES NOT USE IT.
        
    def test_hierarchical_chunker_respects_tokens(self):
        """Test that HierarchicalChunker respects token limits."""
        # Mock tokenizer that counts 2 tokens per word
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2] * len(text.split())
        
        # We can't easily inject the mock tokenizer via name string.
        # But we can patch Utils._token_count or inject it if we modify HierarchicalChunker to accept an instance.
        # I modified HierarchicalChunker to accept `tokenizer_name`.
        # It loads it using AutoTokenizer.
        
        # To test without real HF tokenizer, I can patch Utils._token_count.
        
        # Create text with sentences so it can be split
        # 6 sentences, 10 words each.
        # Each sentence = 20 tokens (mock).
        # Total 120 tokens.
        text = ("word " * 10 + ". ") * 6
        
        chunker = HierarchicalChunker(max_chunk_size=1000, max_chunk_tokens=50, overlap_sentences=0)
        # We want it to split because 120 > 50.
        
        # Patch Utils._token_count to return 2 * words
        original_token_count = Utils._token_count
        # Need to handle the case where text is just "." or empty
        Utils._token_count = lambda t, tokenizer=None: len(t.split()) * 2
        
        try:
            chunks = chunker.chunk(text)
            # Should be split.
            # S1+S2 = 40 tokens. S3 adds 20 -> 60 > 50. Split.
            # Chunk 1: S1, S2.
            # Chunk 2: S3, S4.
            # Chunk 3: S5, S6.
            # Total 3 chunks.
            assert len(chunks) == 3
            assert all(c['token_count'] <= 50 for c in chunks)
        finally:
            Utils._token_count = original_token_count

    def test_chunk_size_character_limit_breach(self):
        """Test that we can limit by tokens."""
        # This is covered by test_hierarchical_chunker_respects_tokens
        pass

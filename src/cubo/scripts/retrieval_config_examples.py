#!/usr/bin/env python3
"""
Examples of different retrieval configurations in CUBO
"""

# Example 1: Dual retrieval (automatic switching)
retriever_dual = DocumentRetriever(
    model=model,
    use_sentence_window=True,  # Enable sentence window
    use_auto_merging=True,  # Enable auto-merging
    auto_merge_for_complex=True,  # Auto-switch based on query complexity
)

# Example 2: Sentence window only
retriever_sentence_only = DocumentRetriever(
    model=model,
    use_sentence_window=True,  # Enable sentence window
    use_auto_merging=False,  # Disable auto-merging
    auto_merge_for_complex=False,  # No auto-switching
)

# Example 3: Auto-merging only
retriever_auto_only = DocumentRetriever(
    model=model,
    use_sentence_window=False,  # Disable sentence window
    use_auto_merging=True,  # Enable auto-merging
    auto_merge_for_complex=False,  # Always use auto-merging
)

# Example 4: Manual control (you decide when to use each)
# You can call the methods directly:
results_sentence = retriever._retrieve_sentence_window(query, top_k=3)
results_auto = retriever._retrieve_auto_merging(query, top_k=3)

from typing import List

def ngram_diversity_score(
    data: List[str],
    num_n: int = 4,
    tokenizer_pattern: str = r"\s+"
) -> float:
    """
    Calculates the averaged corpus-level n-gram diversity (Distinct-N).
    
    This calculates the ratio of unique n-grams to total n-grams for n=1 to num_n,
    and returns the average.
    
    Args:
        data (List[str]): List of document strings. 
        num_n (int): Max n-gram size to test up to. Defaults to 4.
        
    Returns:
        float: Normalized diversity score (0.0 to 1.0).
               1.0 means every n-gram is unique (highly diverse).
               0.0 means high repetition.
    """
    if not data:
        return 0.0

    # 1. Pre-tokenize documents to avoid repeated splitting
    # We filter empty strings to handle multiple spaces safely
    tokenized_docs = [doc.split() for doc in data]
    
    total_ratios = 0.0
    
    for n in range(1, num_n + 1):
        unique_ngrams = set()
        total_ngram_count = 0
        
        for tokens in tokenized_docs:
            if len(tokens) < n:
                continue
                
            # Efficient sliding window without external libraries
            # We calculate total count mathematically: len - n + 1
            count_in_doc = len(tokens) - n + 1
            total_ngram_count += count_in_doc
            
            # Add to set (only stores unique values)
            # Using a generator expression inside set.update is memory efficient
            if n == 1:
                # Speed optimization for unigrams (no tuple creation needed)
                unique_ngrams.update(tokens)
            else:
                unique_ngrams.update(
                    tuple(tokens[i : i + n]) 
                    for i in range(count_in_doc)
                )

        if total_ngram_count == 0:
            # Avoid division by zero if documents are shorter than n
            ratio = 0.0
        else:
            ratio = len(unique_ngrams) / total_ngram_count
            
        total_ratios += ratio

    # Return the average across all N levels to normalize between 0 and 1
    return total_ratios / num_n
import re
from typing import List

def ngram_diversity_score(
    data: List[str],
    num_n: int = 4,
    tokenizer_pattern: str = r"\s+"
) -> float:
    """
    Calculates the averaged corpus-level n-gram diversity (Distinct-N).
    
    Args:
        data (List[str]): List of document strings. 
        num_n (int): Max n-gram size to test up to. Defaults to 4.
        tokenizer_pattern (str): Regex pattern to split text into tokens. 
                                 Defaults to whitespace (r"\\s+").
        
    Returns:
        float: Normalized diversity score (0.0 to 1.0).
    """
    if not data:
        return 0.0

    # 1. Tokenize based on the provided pattern
    if tokenizer_pattern == r"\s+":
        # Fast path: str.split() is highly optimized in C
        tokenized_docs = [doc.split() for doc in data]
    else:
        # Custom path: Use regex
        regex = re.compile(tokenizer_pattern)
        # re.split can produce empty strings (e.g., if the string starts/ends with 
        # the delimiter), so we filter those out 'if token'
        tokenized_docs = [
            [token for token in regex.split(doc) if token] 
            for doc in data
        ]
    
    total_ratios = 0.0
    
    for n in range(1, num_n + 1):
        unique_ngrams = set()
        total_ngram_count = 0
        
        for tokens in tokenized_docs:
            if len(tokens) < n:
                continue
                
            # Total n-grams of size n in this document
            count_in_doc = len(tokens) - n + 1
            total_ngram_count += count_in_doc
            
            # Add to set (only stores unique values)
            if n == 1:
                unique_ngrams.update(tokens)
            else:
                unique_ngrams.update(
                    tuple(tokens[i : i + n]) 
                    for i in range(count_in_doc)
                )

        if total_ngram_count == 0:
            ratio = 0.0
        else:
            ratio = len(unique_ngrams) / total_ngram_count
            
        total_ratios += ratio

    return total_ratios / num_n
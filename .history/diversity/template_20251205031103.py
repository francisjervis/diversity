import re
from typing import List, Dict, Iterable, Optional, Set
from .functions import extract_patterns

def template_rate(
    data: List[str], 
    templates: Optional[Dict[str, Iterable[str]]] = None,
    shard_size: int = 500,
) -> float:
    """ 
    Calculates the template rate (fraction of documents containing at least 1 template).

    Args:
        data (List[str]): Documents to score.
        templates (dict, optional): Dictionary of template_id -> list of strings.
        shard_size (int, optional): Size of regex shards.

    Returns:
        float: Fraction (0.0 to 1.0).
    """
    if not data: 
        return 0.0
    
    if templates is None:
        templates = extract_patterns(data)
    
    matched_text = _gather_substrings(templates)
    
    if not matched_text: 
        return 0.0
        
    regexes = _compile_regex_shards(matched_text, shard_size=shard_size)
    
    # Generator expression is memory efficient
    match_count = sum(1 for doc in data if _has_any(doc, regexes))
    
    return match_count / len(data)


def templates_per_token(
    data: List[str],
    templates: Optional[Dict[str, Iterable[str]]] = None,
    shard_size: int = 500,
) -> List[float]:
    """ 
    Calculates the density of templates per token for each document.
    
    Returns:
        List[float]: List of rates.
    """
    if not data:
        return []

    if templates is None:
        templates = extract_patterns(data)

    substrings = _gather_substrings(templates)
    if not substrings:
        return [0.0] * len(data)

    # Use lookahead shards to count overlapping occurrences
    shards = _compile_regex_shards(substrings, shard_size, overlap=True)

    tpt: List[float] = []
    
    for doc in data:
        # Fast whitespace tokenization
        # Note: If doc is empty/whitespace, count is 0
        tokens = doc.split()
        word_count = len(tokens)
        
        if word_count == 0:
            tpt.append(0.0)
            continue

        occ = 0
        for rx in shards:
            # sum(1 for ...) is standard idiom for counting iterator items
            occ += sum(1 for _ in rx.finditer(doc))
            
        tpt.append(occ / word_count)

    return tpt


def _compile_regex_shards(
    substrings: List[str],
    shard_size: int = 500,
    *,
    overlap: bool = False,
) -> List[re.Pattern]:
    """
    Compiles list of substrings into batched regex patterns.
    """
    # CRITICAL: Sort by length descending. 
    # This ensures "catastrophe" is matched before "cat" in the alternation,
    # preventing shorter prefixes from stealing matches.
    sorted_substrings = sorted(substrings, key=len, reverse=True)
    
    regs: List[re.Pattern] = []
    
    for i in range(0, len(sorted_substrings), shard_size):
        chunk = sorted_substrings[i:i + shard_size]
        
        if not chunk:
            continue
        
        # re.escape allows us to match literal strings containing special regex chars
        alt = "|".join(map(re.escape, chunk))
        
        if overlap:
            # Lookahead (?=...) allows matching overlapping instances
            # e.g., "AA" in "AAAA" matches 3 times (indices 0, 1, 2)
            pat = f"(?=(?:{alt}))"
        else:
            # Non-capturing group for standard matching
            pat = f"(?:{alt})"
        
        regs.append(re.compile(pat))
        
    return regs


def _has_any(text: str, regexes: List[re.Pattern]) -> bool:
    """Helper to check if text matches any of the compiled regex shards."""
    for rx in regexes:
        if rx.search(text):
            return True
    return False


def _gather_substrings(templates: Dict[str, Iterable[str]]) -> List[str]:
    """
    Flattens the template dictionary values into a unique list of strings.
    """
    matched_text: Set[str] = set()
    
    for v in templates.values():
        matched_text.update(v)
    
    return list(matched_text)
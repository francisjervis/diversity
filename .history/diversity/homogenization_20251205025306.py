from typing import List, Optional
from tqdm import tqdm
from rouge_score import rouge_scorer
from evaluate import load

def homogenization_score(
        data: List[str],
        measure: str = 'rougel',
        use_stemmer: bool = False,
        model: str = "microsoft/deberta-base-mnli",
        verbose: bool = True,
        batch_size: int = 64
) -> float:
    """ 
    Calculates the homogenization score (average pairwise similarity).
    
    Warning: Using 'bertscore' on datasets > 500 documents will be extremely slow 
    due to O(N^2) complexity. For large data, use embedding-based cosine similarity.
    
    Args:
         data (List[str]): Strings to score.
         measure (str, optional): 'rougel', 'bertscore', or 'bleu'.
         use_stemmer (bool, optional): For ROUGE-L.
         model (str, optional): For BERTScore.
         verbose (bool, optional): Show progress bar.
         batch_size (int, optional): For BERTScore inference.

     Returns:
         float: Homogenization score (0.0 to 1.0).
    """
    n = len(data)
    if n < 2:
        # Cannot calculate pairwise similarity with less than 2 docs.
        # If there is only 1 doc, is it homogenized? Undefined, but 1.0 implies consistency.
        return 1.0

    if measure == 'rougel':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    elif measure == 'bertscore': 
        scorer = load("bertscore")
    elif measure == 'bleu':
        scorer = load("bleu")
    else: 
        raise ValueError("Scoring measure must be one of `rougel`, `bleu`, or `bertscore`.")

    total_similarity = 0.0
    
    if verbose:
        print(f'==> Scoring {n} documents using {measure}...')
     
    # Iterate through each document (Reference)
    for i, ref in tqdm(enumerate(data), total=n, disable=not verbose):
        
        # Create list of all OTHER documents (Predictions/Hypotheses)
        preds = data[:i] + data[i+1:]
        
        # Create matching list of references (Current doc repeated)
        # Note: BERTScore needs a list of strings for refs.
        # BLEU needs a list of lists of strings (references per prediction).
        
        doc_score = 0.0
        
        if measure == 'rougel':
            # Generator expression is faster than list comprehension
            doc_score = sum(
                scorer.score(pred, ref)['rougeL'].fmeasure 
                for pred in preds
            )
            # Average over the N-1 comparisons
            doc_score = doc_score / len(preds)

        elif measure == 'bertscore':
            refs = [ref] * len(preds)
            results = scorer.compute(
                predictions=preds, 
                references=refs, 
                model_type=model, 
                batch_size=batch_size,
                verbose=False # avoid nested progress bars
            )
            # Average the F1 scores
            doc_score = sum(results['f1']) / len(preds)

        elif measure == 'bleu':
            # BLEU expects references as [[ref1_a, ref1_b], [ref2_a], ...]
            # We treat the current 'ref' as the single reference for every prediction
            refs_formatted = [[ref] for _ in range(len(preds))]
            
            # HF BLEU computes the score for the whole set at once
            result = scorer.compute(predictions=preds, references=refs_formatted)
            doc_score = result['bleu']

        # Add this document's average similarity to the total
        total_similarity += doc_score
    
    # Calculate global average
    final_score = total_similarity / n
    
    return float(final_score)
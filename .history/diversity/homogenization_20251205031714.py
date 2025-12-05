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
        
        Args:
             data (List[str]): Strings to score.
             measure (str, optional): 'rougel', 'bertscore', or 'bleu'.
             use_stemmer (bool, optional): For ROUGE-L.
             model (str, optional): Model checkpoint for BERTScore.
             verbose (bool, optional): Show progress bar.
             batch_size (int, optional): For BERTScore inference.

         Returns:
             float: Homogenization score (0.0 to 1.0).
        """
        n = len(data)
        if n < 2:
            return 1.0

        # Create scorer based on measure type
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
        
        for i, ref in tqdm(enumerate(data), total=n, disable=not verbose):
            
            # Create list of all OTHER documents
            preds = data[:i] + data[i+1:]
            
            doc_score = 0.0
            num_comparisons = len(preds)
            
            if measure == 'rougel':
                # Average ROUGE-L fmeasure against all other docs
                doc_score = sum(
                    scorer.score(pred, ref)['rougeL'].fmeasure 
                    for pred in preds
                ) / num_comparisons

            elif measure == 'bertscore':
                # BERTScore handles batching internally
                refs = [ref] * num_comparisons
                results = scorer.compute(
                    predictions=preds, 
                    references=refs, 
                    model_type=model, 
                    batch_size=batch_size,
                    verbose=False
                )
                doc_score = sum(results['f1']) / num_comparisons

            elif measure == 'bleu':
                # To ensure consistency with ROUGE/BERTScore, we must calculate 
                # Mean Pairwise Sentence BLEU, not Corpus BLEU.
                # We iterate pairwise to normalize correctly by number of comparisons.
                bleu_sum = 0.0
                for pred in preds:
                    # predictions=list, references=list of lists
                    res = scorer.compute(predictions=[pred], references=[[ref]])
                    bleu_sum += res['bleu']
                
                doc_score = bleu_sum / num_comparisons

            total_similarity += doc_score
        
        return float(total_similarity / n)
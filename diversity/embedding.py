'''
Computes remote clique and chamfer distance on embeddings for a set of documents
to understand their semantic (in embedding space) diversity.

Following Samuel Rhys Cox et al. 2021. "Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation". In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (CHI '21). Association for Computing Machinery, New York, NY, USA, Article 393, 1–35. https://doi.org/10.1145/3411764.3445782
'''

from typing import List, Optional, Any, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from evaluate import load as load_metric
from tqdm import tqdm

def remote_clique(
        data: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64,
        use_pairwise_matrix: bool = False
) -> float:
    """
    Calculates the remote clique score for a set of documents (corpus-level).
    This measures the average mean pairwise distance between all items.
    
    Args:
        data (List[str]): List of text documents/strings to score.
        model (str, optional): The HuggingFace SentenceTransformer model name to use for embedding. 
                               Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose (bool, optional): If True, displays a progress bar during embedding. Defaults to True.
        batch_size (int, optional): Batch size for the embedding model inference. Defaults to 64.
        use_pairwise_matrix (bool, optional): Controls the distance metric and computation method.
            - If False (default): Computes **Cosine Distance** (1 - cosine_similarity). Uses an O(N) 
              mathematical optimization (vector sum) which is extremely fast but approximates the 
              paper's strict definition.
            - If True: Computes **Angular Distance** (arccos(cosine_similarity)). This forces an O(N^2) 
              calculation of the full pairwise matrix, which is significantly slower but strictly 
              adheres to Section 3.2 of the paper.
    
    Returns:
        float: The remote clique diversity score. Higher values indicate greater diversity.
    """
    client = DiversityClient(model=model)
    return client.remote_clique(
        data, 
        verbose=verbose, 
        batch_size=batch_size, 
        use_pairwise_matrix=use_pairwise_matrix
    )


def chamfer_dist(
        data: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64,
        use_pairwise_matrix: bool = False
) -> float:
    """
    Calculates the chamfer distance for a set of documents (corpus-level).
    This measures the average distance to the nearest neighbor for each item.
    
    Args:
        data (List[str]): List of text documents/strings to score.
        model (str, optional): The HuggingFace SentenceTransformer model name to use for embedding. 
                               Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose (bool, optional): If True, displays a progress bar during embedding. Defaults to True.
        batch_size (int, optional): Batch size for the embedding model inference. Defaults to 64.
        use_pairwise_matrix (bool, optional): Controls the distance metric used.
            - If False (default): Computes **Cosine Distance** (1 - cosine_similarity).
            - If True: Computes **Angular Distance** (arccos(cosine_similarity)). This strictly 
              adheres to Section 3.2 of the paper. 
            Note: Both settings require O(N^2) comparisons to find nearest neighbors, but the 
            metric values will differ.
    
    Returns:
        float: The chamfer distance score. Higher values indicate lower redundancy (higher diversity).
    """
    client = DiversityClient(model=model)
    return client.chamfer_dist(
        data, 
        verbose=verbose, 
        batch_size=batch_size, 
        use_pairwise_matrix=use_pairwise_matrix
    )
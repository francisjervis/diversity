'''
Computes remote clique and chamfer distance on embeddings for a set of documents
to understand their semantic (in embedding space) diversity.

Following Samuel Rhys Cox et al. 2021. "Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation". In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (CHI '21). Association for Computing Machinery, New York, NY, USA, Article 393, 1–35. https://doi.org/10.1145/3411764.3445782
'''

from typing import List, Optional

# Import here to avoid circular imports
from .client import DiversityClient


def remote_clique(
        data: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
) -> float:
    """
    Calculates the remote clique score for a set of documents (corpus-level).
    This is the average mean pairwise distance of a data instance to other instances.
    
    Note: For better performance when computing multiple metrics, use DiversityClient instead:
        >>> from diversity import DiversityClient
        >>> client = DiversityClient(model="Qwen/Qwen3-Embedding-0.6B")
        >>> rc = client.remote_clique(data)
    
    Args:
        data (List[str]): Strings to score.
        model (str, optional): Model to use for embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose (bool, optional): Whether to display progress bar. Defaults to True.
        batch_size (int, optional): Batch size for embedding. Defaults to 64.
    
    Returns:
        float: Remote clique score.
    """
    client = DiversityClient(model=model)
    return client.remote_clique(data, verbose=verbose, batch_size=batch_size)


def chamfer_dist(
        data: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
) -> float:
    """
    Calculates the chamfer distance for a set of documents (corpus-level).
    This is the average minimum pairwise distance of a data instance to other instances.
    
    Note: For better performance when computing multiple metrics, use DiversityClient instead:
        >>> from diversity import DiversityClient
        >>> client = DiversityClient(model="Qwen/Qwen3-Embedding-0.6B")
        >>> cd = client.chamfer_dist(data)
    
    Args:
        data (List[str]): Strings to score.
        model (str, optional): Model to use for embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose (bool, optional): Whether to display progress bar. Defaults to True.
        batch_size (int, optional): Batch size for embedding. Defaults to 64.
    
    Returns:
        float: Chamfer distance.
    """
    client = DiversityClient(model=model)
    return client.chamfer_dist(data, verbose=verbose, batch_size=batch_size)

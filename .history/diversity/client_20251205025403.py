"""
Client class for managing cached SentenceTransformer models.

This client enables efficient reuse of loaded models across multiple
embedding calculations, avoiding redundant model loading.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Optional, List, Union

class DiversityClient:
    """
    Client class that caches SentenceTransformer models for efficient reuse.
    
    This class maintains an internal cache of loaded models, allowing
    multiple embedding calculations to share the same model instance
    without reloading it each time.
    
    Example:
        >>> from diversity import DiversityClient
        >>> client = DiversityClient(model="Qwen/Qwen3-Embedding-0.6B")
        >>> rc = client.remote_clique(texts)
    """
    
    def __init__(self, model: str = 'Qwen/Qwen3-Embedding-0.6B', device: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            model (str): The default SentenceTransformer model.
            device (str, optional): Device to load model on (e.g., 'cpu', 'cuda', 'mps'). 
                                    If None, automatically detects best device.
        """
        self._default_model = model
        self._device = device
        self._model_cache: Dict[str, SentenceTransformer] = {}
    
    def _get_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """
        Get a cached model or load it if not already cached.
        """
        model_to_use = model_name or self._default_model
        if model_to_use not in self._model_cache:
            self._model_cache[model_to_use] = SentenceTransformer(model_to_use, device=self._device)
        return self._model_cache[model_to_use]
    
    def remote_clique(
        self,
        data: List[str],
        model: Optional[str] = None,
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
    ) -> float:
        """
        Calculates the remote clique score (average mean pairwise cosine distance).
        
        Optimization:
        Instead of computing an N*N distance matrix (which causes OOM for N > 20k),
        we use the algebraic identity: Sum(Sim_Matrix) = Dot(Sum(Vectors), Sum(Vectors)).
        
        Formula:
            Mean_Similarity = |Sum(Vectors)|^2 / N^2
            Mean_Distance = 1 - Mean_Similarity
        
        Args:
            data (List[str]): Strings to score.
            model (str, optional): Model to use.
            verbose (bool, optional): Show progress bar.
            batch_size (int, optional): Batch size for embedding.
        
        Returns:
            float: Remote clique score (0.0 to 1.0). Higher is more diverse.
        """
        n = len(data)
        if n == 0:
            return 0.0
            
        transformer_model = self._get_model(model)
        
        # Critical: Embeddings must be normalized for dot product to equal cosine similarity
        embeddings = transformer_model.encode(
            data, 
            batch_size=batch_size, 
            show_progress_bar=verbose, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # 1. Sum all embedding vectors along the feature axis
        # Shape becomes (embedding_dim,)
        vector_sum = np.sum(embeddings, axis=0)
        
        # 2. Compute the squared magnitude of the sum vector
        # This equals the sum of all elements in the N*N similarity matrix
        sum_of_similarities = np.dot(vector_sum, vector_sum)
        
        # 3. Calculate mean similarity
        # We divide by N^2 because the full matrix has N*N elements
        mean_similarity = sum_of_similarities / (n * n)
        
        # 4. Convert to distance (Distance = 1 - Similarity)
        # Note: This naturally includes the diagonal (self-distance = 0)
        mean_distance = 1.0 - mean_similarity
        
        return float(mean_distance)
    
    def chamfer_dist(
        self,
        data: List[str],
        model: Optional[str] = None,
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64,
        eval_batch_size: int = 1000  # Size of matrix chunks to process
    ) -> float:
        """
        Calculates the chamfer distance (average minimum pairwise distance).
        
        This implementation uses chunking to avoid OOM errors on large datasets.
        
        Args:
            data (List[str]): Strings to score.
            model (str, optional): Model to use.
            verbose (bool, optional): Show progress bar.
            batch_size (int, optional): Batch size for embedding generation.
            eval_batch_size (int, optional): Batch size for distance calculation matrix chunks.
                                             Lower this if running OOM.
        
        Returns:
            float: Chamfer distance (0.0 to 1.0).
        """
        n = len(data)
        if n <= 1:
            return 0.0

        transformer_model = self._get_model(model)
        
        # 1. Encode with normalization (Crucial: Dot Product == Cosine Similarity)
        # We keep embeddings in memory (O(N * D)), which is much smaller than O(N^2)
        embeddings = transformer_model.encode(
            data, 
            batch_size=batch_size, 
            show_progress_bar=verbose, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        min_distances = []
        
        # 2. Process in chunks to avoid creating the full N*N matrix
        # This reduces memory usage from ~20GB (for N=50k) to ~400MB
        for i in range(0, n, eval_batch_size):
            end = min(i + eval_batch_size, n)
            
            # Select a chunk of queries
            query_chunk = embeddings[i:end]
            
            # Compute similarity: (Chunk_Size, D) @ (D, N) -> (Chunk_Size, N)
            # This creates a rectangular slice of the similarity matrix
            sim_chunk = np.dot(query_chunk, embeddings.T)
            
            # Mask self-similarity (the diagonal)
            # The row 'k' in this chunk corresponds to global index 'i + k'
            for k in range(len(query_chunk)):
                # Set self-similarity to -1.0 (lowest possible cosine sim)
                # so it is not selected as the max similarity
                sim_chunk[k, i + k] = -1.0
            
            # Find max similarity (nearest neighbor) for each item in chunk
            max_sims = np.max(sim_chunk, axis=1)
            
            # Convert to distance: Dist = 1 - Sim
            min_dists = 1.0 - max_sims
            min_distances.extend(min_dists)

        return float(np.mean(min_distances))

        
    def clear_cache(self) -> None:
        """Clear all cached models from memory."""
        self._model_cache.clear()
    
    def remove_model(self, model_name: str) -> None:
        """
        Remove a specific model from the cache.
        
        Args:
            model_name (str): The name of the model to remove from cache.
        """
        self._model_cache.pop(model_name, None)
    
    def has_model(self, model_name: str) -> bool:
        """
        Check if a model is currently cached.
        
        Args:
            model_name (str): The name of the model to check.
        
        Returns:
            bool: True if the model is cached, False otherwise.
        """
        return model_name in self._model_cache

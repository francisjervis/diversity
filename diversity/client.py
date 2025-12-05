"""
Client class for managing cached SentenceTransformer models.

This client enables efficient reuse of loaded models across multiple
embedding calculations, avoiding redundant model loading.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from typing import Dict, Optional, List


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
        >>> cd = client.chamfer_dist(texts)
        >>> # Both calls use the same cached model instance
    """
    
    def __init__(self, model: str = 'Qwen/Qwen3-Embedding-0.6B'):
        """
        Initialize the client with a default model.
        
        Args:
            model (str): The default SentenceTransformer model to use
                        (e.g., "Qwen/Qwen3-Embedding-0.6B").
        """
        self._default_model = model
        self._model_cache: Dict[str, SentenceTransformer] = {}
    
    def _get_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """
        Get a cached model or load it if not already cached.
        
        Args:
            model_name (str, optional): The name/identifier of the SentenceTransformer model
                                       to load. If None, uses the default model.
        
        Returns:
            SentenceTransformer: The loaded model instance.
        """
        model_to_use = model_name or self._default_model
        if model_to_use not in self._model_cache:
            self._model_cache[model_to_use] = SentenceTransformer(model_to_use)
        return self._model_cache[model_to_use]
    
    def remote_clique(
        self,
        data: List[str],
        model: Optional[str] = None,
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
    ) -> float:
        """
        Calculates the remote clique score for a set of documents (corpus-level).
        This is the average mean pairwise distance of a data instance to other instances.
        
        Args:
            data (List[str]): Strings to score.
            model (str, optional): Model to use for embedding. If None, uses the default model.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.
            batch_size (int, optional): Batch size for embedding. Defaults to 64.
        
        Returns:
            float: Remote clique score.
        """
        transformer_model = self._get_model(model)
        embeddings = transformer_model.encode(data, batch_size=batch_size, show_progress_bar=verbose)
        distances = cosine_distances(embeddings)
        mean_distances = np.mean(distances, axis=1)
        return np.mean(mean_distances).round(3)
    
    def chamfer_dist(
        self,
        data: List[str],
        model: Optional[str] = None,
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
    ) -> float:
        """
        Calculates the chamfer distance for a set of documents (corpus-level).
        This is the average minimum pairwise distance of a data instance to other instances.
        
        Args:
            data (List[str]): Strings to score.
            model (str, optional): Model to use for embedding. If None, uses the default model.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.
            batch_size (int, optional): Batch size for embedding. Defaults to 64.
        
        Returns:
            float: Chamfer distance.
        """
        transformer_model = self._get_model(model)
        embeddings = transformer_model.encode(data, batch_size=batch_size, show_progress_bar=verbose)
        distances = cosine_distances(embeddings)
        min_distances = np.min(distances + np.eye(len(distances)) * 1e9, axis=1)
        return np.mean(min_distances).round(3)
    
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

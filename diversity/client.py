class DiversityClient:
    """
    Client class that caches SentenceTransformer models and Evaluation metrics
    for efficient reuse across multiple calculations.
    """
    
    def __init__(self, model: str = 'Qwen/Qwen3-Embedding-0.6B', device: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            model (str): The default SentenceTransformer model to load.
            device (str, optional): Device to load models on (e.g., 'cpu', 'cuda', 'mps'). 
                                    If None, automatically detects best device.
        """
        self._default_model = model
        self._device = device
        self._model_cache: Dict[str, SentenceTransformer] = {}
        self._scorer_cache: Dict[str, Any] = {}
    
    def _get_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """Get a cached embedding model or load it if not already cached."""
        model_to_use = model_name or self._default_model
        if model_to_use not in self._model_cache:
            self._model_cache[model_to_use] = SentenceTransformer(model_to_use, device=self._device)
        return self._model_cache[model_to_use]

    def _get_scorer(self, measure: str, use_stemmer: bool = False) -> Any:
        """Get a cached scorer object (ROUGE, BERTScore, BLEU) or load it."""
        if measure == 'rougel':
            cache_key = f"rougel_{use_stemmer}"
        else:
            cache_key = measure

        if cache_key not in self._scorer_cache:
            if measure == 'rougel':
                self._scorer_cache[cache_key] = rouge_scorer.RougeScorer(
                    ['rougeL'], use_stemmer=use_stemmer
                )
            elif measure in ['bertscore', 'bleu']:
                self._scorer_cache[cache_key] = load_metric(measure)
            else:
                raise ValueError("Scoring measure must be one of `rougel`, `bleu`, or `bertscore`.")
        
        return self._scorer_cache[cache_key]
    
    def remote_clique(
        self,
        data: List[str],
        model: Optional[str] = None,
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64,
        eval_batch_size: int = 1000,
        use_pairwise_matrix: bool = False
    ) -> float:
        """
        Calculates the remote clique score.

        Args:
            data (List[str]): Strings to score.
            model (Optional[str]): Override default model.
            verbose (Optional[bool]): Show progress bar.
            batch_size (Optional[int]): Inference batch size.
            eval_batch_size (int): Size of chunks to process when computing pairwise matrix to 
                                   avoid OOM errors. Defaults to 1000.
            use_pairwise_matrix (bool): 
                If False (default): Uses O(N) vector sum optimization for Cosine Distance.
                If True: Uses O(N^2) full matrix calculation for Angular Distance.
        
        Returns:
            float: Remote clique score.
        """
        n = len(data)
        if n == 0:
            return 0.0
            
        transformer_model = self._get_model(model)
        
        embeddings = transformer_model.encode(
            data, 
            batch_size=batch_size, 
            show_progress_bar=verbose, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        if not use_pairwise_matrix:
            # O(N) Optimization: Cosine Distance
            vector_sum = np.sum(embeddings, axis=0)
            sum_of_similarities = np.dot(vector_sum, vector_sum)
            mean_similarity = sum_of_similarities / (n * n)
            mean_distance = 1.0 - mean_similarity
            return float(mean_distance)
        
        else:
            # O(N^2) Calculation: Angular Distance (Strict Paper Adherence)
            total_angular_dist = 0.0
            
            # Iterate in batches to maintain memory stability
            for i in range(0, n, eval_batch_size):
                end = min(i + eval_batch_size, n)
                query_chunk = embeddings[i:end]
                
                sim_chunk = np.dot(query_chunk, embeddings.T)
                
                # Clip to avoid numerical instability outside [-1, 1] for arccos
                sim_chunk = np.clip(sim_chunk, -1.0, 1.0)
                
                # Convert to Angular Distance
                dist_chunk = np.arccos(sim_chunk)
                
                total_angular_dist += np.sum(dist_chunk)
                
            return float(total_angular_dist / (n * n))
    
    def chamfer_dist(
        self,
        data: List[str],
        model: Optional[str] = None,
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64,
        eval_batch_size: int = 1000,
        use_pairwise_matrix: bool = False
    ) -> float:
        """
        Calculates the chamfer distance.

        Args:
            data (List[str]): Strings to score.
            model (Optional[str]): Override default model.
            verbose (Optional[bool]): Show progress bar.
            batch_size (Optional[int]): Inference batch size.
            eval_batch_size (int): Size of chunks to process when computing pairwise matrix.
            use_pairwise_matrix (bool):
                If False (default): Uses Cosine Distance (1 - cos).
                If True: Uses Angular Distance (arccos(cos)).

        Returns:
            float: Chamfer distance score.
        """
        n = len(data)
        if n <= 1:
            return 0.0

        transformer_model = self._get_model(model)
        embeddings = transformer_model.encode(
            data, 
            batch_size=batch_size, 
            show_progress_bar=verbose, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        min_distances = []
        
        for i in range(0, n, eval_batch_size):
            end = min(i + eval_batch_size, n)
            query_chunk = embeddings[i:end]
            sim_chunk = np.dot(query_chunk, embeddings.T)
            
            # Mask self-similarity
            for k in range(len(query_chunk)):
                sim_chunk[k, i + k] = -1.0
            
            max_sims = np.max(sim_chunk, axis=1)
            
            if use_pairwise_matrix:
                # Angular Distance
                max_sims = np.clip(max_sims, -1.0, 1.0)
                dists = np.arccos(max_sims)
            else:
                # Cosine Distance
                dists = 1.0 - max_sims
                
            min_distances.extend(dists)

        return float(np.mean(min_distances))

    def homogenization_score(
        self,
        data: List[str],
        measure: str = 'rougel',
        use_stemmer: bool = False,
        model: str = "microsoft/deberta-base-mnli",
        verbose: bool = True,
        batch_size: int = 64
    ) -> float:
        """
        Calculates the homogenization score (average pairwise similarity) using text-overlap metrics.
        
        Args:
             data (List[str]): Strings to score.
             measure (str, optional): The metric to use: 'rougel', 'bertscore', or 'bleu'.
             use_stemmer (bool, optional): For ROUGE-L only. Applies stemming before scoring.
             model (str, optional): Model checkpoint to use if measure is 'bertscore'.
             verbose (bool, optional): Show progress bar.
             batch_size (int, optional): Inference batch size for 'bertscore'.

         Returns:
             float: Homogenization score (0.0 to 1.0). Higher values indicate higher similarity.
        """
        n = len(data)
        if n < 2:
            return 1.0

        scorer = self._get_scorer(measure, use_stemmer)
        total_similarity = 0.0
        
        if verbose:
            print(f'==> Scoring {n} documents using {measure}...')
        
        for i, ref in tqdm(enumerate(data), total=n, disable=not verbose):
            preds = data[:i] + data[i+1:]
            num_comparisons = len(preds)
            doc_score = 0.0
            
            if measure == 'rougel':
                doc_score = sum(
                    scorer.score(pred, ref)['rougeL'].fmeasure 
                    for pred in preds
                ) / num_comparisons

            elif measure == 'bertscore':
                refs = [ref] * num_comparisons
                results = scorer.compute(
                    predictions=preds, 
                    references=refs, 
                    model_type=model, 
                    batch_size=batch_size,
                    verbose=False,
                    device=self._device
                )
                doc_score = sum(results['f1']) / num_comparisons

            elif measure == 'bleu':
                bleu_sum = 0.0
                for pred in preds:
                    res = scorer.compute(predictions=[pred], references=[[ref]])
                    bleu_sum += res['bleu']
                doc_score = bleu_sum / num_comparisons

            total_similarity += doc_score
        
        return float(total_similarity / n)

    def clear_cache(self) -> None:
        """Clear all cached models and metrics from memory."""
        self._model_cache.clear()
        self._scorer_cache.clear()
    
    def remove_model(self, model_name: str) -> None:
        """Remove a specific embedding model from the cache."""
        self._model_cache.pop(model_name, None)
    
    def has_model(self, model_name: str) -> bool:
        """Check if an embedding model is currently cached."""
        return model_name in self._model_cache
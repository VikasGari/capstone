def l2_to_similarity(dist: float) -> float:
    """Converts Euclidean L2 distance score back to relative similarity score."""
    return float(1.0 / (1.0 + dist))

def reciprocal_rank_fusion(results_lists: list[list[dict]], rrf_k: int) -> list[dict]:
    """
    Fuses multiple lists of candidate document rankings using Reciprocal Rank Fusion (RRF).
    Supports fusing any arbitrary number of candidate rankings.
    """
    rrf_scores = {}
    doc_details = {}
    
    for results in results_lists:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            if doc_id not in doc_details:
                doc_details[doc_id] = {
                    "id": doc_id,
                    "document": item["document"],
                    "metadata": item["metadata"]
                }
            # Calculate and add RRF rank score
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))
            
    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    fused_results = []
    for doc_id in sorted_ids:
        doc = doc_details[doc_id]
        fused_results.append({
            "id": doc["id"],
            "document": doc["document"],
            "metadata": doc["metadata"],
            "score": rrf_scores[doc_id]
        })
        
    return fused_results

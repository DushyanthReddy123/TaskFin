"""
FAISS-based retriever for searching bills and transactions.

This module loads the FAISS index and metadata created by seed_data.py
and provides a search function to query the embeddings.
"""

import os
import pickle
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Module-level variables for lazy loading
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_metadata: Optional[List[Dict[str, Any]]] = None
_model_info: Optional[Dict[str, Any]] = None


def _get_embeddings_dir() -> str:
    """Get the embeddings directory path relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "embeddings")


def _load_retriever_components():
    """Load the model, index, and metadata from disk."""
    global _model, _index, _metadata, _model_info
    
    if _model is not None and _index is not None:
        # Already loaded
        return
    
    embeddings_dir = _get_embeddings_dir()
    
    # Check if embeddings directory exists
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(
            f"Embeddings directory not found at {embeddings_dir}. "
            "Please run seed_data.py first to generate embeddings."
        )
    
    # Load model info
    model_info_path = os.path.join(embeddings_dir, "model_info.pkl")
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(
            f"Model info not found at {model_info_path}. "
            "Please run seed_data.py first to generate embeddings."
        )
    
    with open(model_info_path, 'rb') as f:
        _model_info = pickle.load(f)
    
    # Load and initialize the model
    model_name = _model_info['model_name']
    _model = SentenceTransformer(model_name)
    
    # Load FAISS index
    index_path = os.path.join(embeddings_dir, "faiss_index.bin")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Please run seed_data.py first to generate embeddings."
        )
    
    _index = faiss.read_index(index_path)
    
    # Load metadata
    metadata_path = os.path.join(embeddings_dir, "metadata.pkl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            "Please run seed_data.py first to generate embeddings."
        )
    
    with open(metadata_path, 'rb') as f:
        _metadata = pickle.load(f)
    
    # Validate that metadata length matches index size
    if len(_metadata) != _index.ntotal:
        raise ValueError(
            f"Metadata length ({len(_metadata)}) does not match "
            f"index size ({_index.ntotal})"
        )


def init_retriever():
    """Initialize the retriever by loading all components."""
    _load_retriever_components()


def search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for the top-k most similar items to the query.
    
    Args:
        query: The search query string
        k: Number of results to return (default: 5)
    
    Returns:
        List of dictionaries containing:
        - 'metadata': The original metadata dict for the item
        - 'distance': The L2 distance from the query (lower is better)
        - 'text': Reconstructed human-readable text (if available)
    
    Raises:
        FileNotFoundError: If embeddings directory or files are missing
        ValueError: If metadata and index sizes don't match
    """
    # Load components if not already loaded
    _load_retriever_components()
    
    if _model is None or _index is None or _metadata is None:
        raise RuntimeError("Retriever components not initialized")
    
    # Encode the query
    query_embedding = _model.encode([query], show_progress_bar=False)
    query_embedding = query_embedding.astype('float32')
    
    # Ensure correct shape: (1, dimension)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Search the index
    distances, indices = _index.search(query_embedding, min(k, _index.ntotal))
    
    # Build results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(_metadata):
            # Invalid index, skip
            continue
        
        metadata = _metadata[idx].copy()
        
        # Reconstruct text from metadata
        text = _reconstruct_text(metadata)
        
        result = {
            'metadata': metadata,
            'distance': float(distance),
            'text': text,
            'rank': i + 1
        }
        results.append(result)
    
    return results


def _reconstruct_text(metadata: Dict[str, Any]) -> str:
    """
    Reconstruct the human-readable text from metadata.
    
    This matches the format used in seed_data.py.
    """
    if metadata.get('type') == 'bill':
        status_text = metadata.get('status', 'unknown')
        due_date = metadata.get('due_date')
        due_date_str = due_date if due_date else "unknown"
        return (
            f"Bill: {metadata.get('name', 'Unknown')}. "
            f"Amount: ${metadata.get('amount', 0):.2f}. "
            f"Due date: {due_date_str}. "
            f"Status: {status_text}."
        )
    elif metadata.get('type') == 'transaction':
        date_str = metadata.get('date', 'unknown')
        return (
            f"Transaction: {metadata.get('description', 'Unknown')}. "
            f"Amount: ${metadata.get('amount', 0):.2f}. "
            f"Date: {date_str}."
        )
    else:
        return "Unknown item"


def get_index_stats() -> Dict[str, Any]:
    """
    Get statistics about the loaded index.
    
    Returns:
        Dictionary with index statistics
    """
    _load_retriever_components()
    
    if _model_info is None or _index is None or _metadata is None:
        raise RuntimeError("Retriever components not initialized")
    
    return {
        'model_name': _model_info.get('model_name'),
        'dimension': _model_info.get('dimension'),
        'total_vectors': _index.ntotal,
        'metadata_count': len(_metadata),
        'is_loaded': True
    }

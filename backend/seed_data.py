"""
Seed data script to generate embeddings and store in FAISS.

This script:
1. Reads all bills and transactions from the database
2. Converts them to searchable text representations
3. Generates embeddings using sentence-transformers
4. Stores embeddings in FAISS index
5. Saves the FAISS index and metadata to disk
"""

import os
import sys
import pickle
from datetime import date
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import models, session


def format_bill_text(bill: models.Bill) -> str:
    """Convert a Bill object to a searchable text string."""
    status_text = "paid" if bill.status == "paid" else "unpaid"
    due_date_str = bill.due_date.strftime("%Y-%m-%d") if bill.due_date else "unknown"
    return (
        f"Bill: {bill.name}. "
        f"Amount: ${bill.amount:.2f}. "
        f"Due date: {due_date_str}. "
        f"Status: {status_text}."
    )


def format_transaction_text(transaction: models.Transaction) -> str:
    """Convert a Transaction object to a searchable text string."""
    date_str = transaction.date.strftime("%Y-%m-%d") if transaction.date else "unknown"
    return (
        f"Transaction: {transaction.description}. "
        f"Amount: ${transaction.amount:.2f}. "
        f"Date: {date_str}."
    )


def load_data_from_db():
    """Load all bills and transactions from the database."""
    db = session.SessionLocal()
    try:
        bills = db.query(models.Bill).all()
        transactions = db.query(models.Transaction).all()
        return bills, transactions
    finally:
        db.close()


def generate_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for a list of text strings."""
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.astype('float32')


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    # Using L2 distance (Euclidean) - can switch to cosine similarity if needed
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def main():
    """Main function to seed embeddings."""
    print("Starting embedding generation...")
    
    # Initialize embedding model
    # Using a lightweight, general-purpose model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load data from database
    print("Loading data from database...")
    bills, transactions = load_data_from_db()
    
    if not bills and not transactions:
        print("No data found in database. Please seed the database first.")
        return
    
    # Convert to text representations
    texts = []
    metadata = []  # Store metadata for each text (type, id, original object)
    
    for bill in bills:
        text = format_bill_text(bill)
        texts.append(text)
        metadata.append({
            'type': 'bill',
            'id': bill.id,
            'user_id': bill.user_id,
            'name': bill.name,
            'amount': float(bill.amount),
            'due_date': bill.due_date.isoformat() if bill.due_date else None,
            'status': bill.status
        })
    
    for transaction in transactions:
        text = format_transaction_text(transaction)
        texts.append(text)
        metadata.append({
            'type': 'transaction',
            'id': transaction.id,
            'user_id': transaction.user_id,
            'description': transaction.description,
            'amount': float(transaction.amount),
            'date': transaction.date.isoformat() if transaction.date else None
        })
    
    print(f"Total texts to embed: {len(texts)}")
    print(f"  - Bills: {len(bills)}")
    print(f"  - Transactions: {len(transactions)}")
    
    # Generate embeddings
    embeddings = generate_embeddings(texts, model)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Create FAISS index
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings)
    print(f"FAISS index created with {index.ntotal} vectors")
    
    # Save index and metadata
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "faiss_index.bin")
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save the model info for later use
    model_info = {
        'model_name': 'all-MiniLM-L6-v2',
        'dimension': embeddings.shape[1],
        'total_vectors': len(texts)
    }
    model_info_path = os.path.join(output_dir, "model_info.pkl")
    with open(model_info_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print("\n✅ Embedding generation complete!")
    print(f"   - Index saved to: {index_path}")
    print(f"   - Metadata saved to: {metadata_path}")
    print(f"   - Model info saved to: {model_info_path}")


if __name__ == "__main__":
    main()

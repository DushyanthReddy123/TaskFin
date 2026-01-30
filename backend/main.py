from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import date, timedelta
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .db import models, session
from . import auth, retriever
import time
import os

# Create tables
# Wait a bit for DB to start if running docker-compose up concurrently
# In production, use migration tools like Alembic.
# For now, we just create if not exists on startup.
models.Base.metadata.create_all(bind=session.engine)

app = FastAPI()


# Pydantic models for search endpoint
class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    metadata: Dict[str, Any]
    distance: float
    text: str
    rank: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    k: int


def get_db():
    db = session.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup_event():
    # Insert fake data if not exists
    db = session.SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == "test@example.com").first()
        if not user:
            # Create user with hashed password (default password: "password")
            hashed_password = auth.get_password_hash("password")
            user = models.User(name="Test User", email="test@example.com", password=hashed_password)
            db.add(user)
            db.commit()
            db.refresh(user)

            # Fake bills
            bill1 = models.Bill(name="Electric Bill", amount=120.50, due_date=date(2023, 10, 15), status="unpaid", owner=user)
            bill2 = models.Bill(name="Internet", amount=60.00, due_date=date(2023, 10, 20), status="paid", owner=user)
            db.add_all([bill1, bill2])
            
            # Fake transactions
            txn1 = models.Transaction(amount=120.50, date=date(2023, 9, 15), description="Last Month Electric", owner=user)
            db.add(txn1)
            
            db.commit()
        elif not user.password:
            # If user exists but has no password, add one
            hashed_password = auth.get_password_hash("password")
            user.password = hashed_password
            db.commit()
    except Exception as e:
        print(f"Error seeding data: {e}")
    finally:
        db.close()

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login endpoint that returns JWT token."""
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/chat")
def chat(current_user: models.User = Depends(auth.get_current_user)):
    """Protected chat endpoint."""
    return {"message": "ok"}


@app.get("/bills")
def get_bills(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Get bills for the authenticated user."""
    bills = db.query(models.Bill).filter(models.Bill.user_id == current_user.id).all()
    if not bills:
        return []
    return bills


@app.post("/pay-bill")
def pay_bill(bill_id: int, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Mark a bill as paid. Protected endpoint."""
    bill = db.query(models.Bill).filter(
        models.Bill.id == bill_id,
        models.Bill.user_id == current_user.id
    ).first()
    
    if not bill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bill not found"
        )
    
    bill.status = "paid"
    db.commit()
    db.refresh(bill)
    return bill


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Search endpoint for querying bills and transactions using FAISS.
    
    This endpoint performs semantic search on the embedded bills and transactions
    and returns the top-k most similar items.
    
    Args:
        request: SearchRequest with query string and optional k (default: 5)
    
    Returns:
        SearchResponse with list of results containing metadata, distance, and text
    
    Raises:
        HTTPException: If embeddings are not found or retriever fails to initialize
    """
    try:
        results = retriever.search(request.query, request.k)
        
        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                metadata=result['metadata'],
                distance=result['distance'],
                text=result['text'],
                rank=result['rank']
            )
            for result in results
        ]
        
        return SearchResponse(
            results=search_results,
            query=request.query,
            k=len(search_results)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search service not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

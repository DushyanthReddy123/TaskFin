from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import date, timedelta
from .db import models, session
from . import auth
import time
import os

# Create tables
# Wait a bit for DB to start if running docker-compose up concurrently
# In production, use migration tools like Alembic.
# For now, we just create if not exists on startup.
models.Base.metadata.create_all(bind=session.engine)

app = FastAPI()

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

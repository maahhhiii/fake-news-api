from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import engine, Base, get_db
from app import models, schemas, crud
from app.ml_model import predict_news

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Fake News Detection API",
    description="Detect whether news is REAL or FAKE using NLP and Machine Learning",
    version="1.0"
)


@app.get("/")
def home():
    return {"message": "Welcome to Fake News Detection API"}


# Predict news
@app.post("/predict", response_model=schemas.PredictionResponse)
def predict(
    request: schemas.NewsRequest,
    db: Session = Depends(get_db)
):
    result = predict_news(request.text)

    # Save prediction to database
    crud.create_prediction(
        db=db,
        news_text=request.text,
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

    return result


# Get all prediction history
@app.get("/history", response_model=List[schemas.PredictionHistory])
def get_history(db: Session = Depends(get_db)):
    return crud.get_predictions(db)


# Get prediction by ID
@app.get("/history/{prediction_id}",
         response_model=schemas.PredictionHistory)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    prediction = crud.get_prediction_by_id(db, prediction_id)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    return prediction


# Delete prediction by ID
@app.delete("/history/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    prediction = crud.delete_prediction(db, prediction_id)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    return {"message": "Prediction deleted successfully"}
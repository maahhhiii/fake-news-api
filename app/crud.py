from sqlalchemy.orm import Session
from app import models

# Save prediction to database
def create_prediction(
    db: Session,
    news_text: str,
    prediction: str,
    confidence: float
):
    db_prediction = models.Prediction(
        news_text=news_text,
        prediction=prediction,
        confidence=confidence
    )

    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    return db_prediction


# Get all prediction history
def get_predictions(db: Session):
    return db.query(models.Prediction).all()


# Get prediction by ID
def get_prediction_by_id(db: Session, prediction_id: int):
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.id == prediction_id)
        .first()
    )


# Delete prediction by ID
def delete_prediction(db: Session, prediction_id: int):
    prediction = (
        db.query(models.Prediction)
        .filter(models.Prediction.id == prediction_id)
        .first()
    )

    if prediction:
        db.delete(prediction)
        db.commit()

    return prediction
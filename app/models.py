from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.database import Base   

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    news_text = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
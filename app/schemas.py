from pydantic import BaseModel
from datetime import datetime

class NewsRequest(BaseModel):
    text:str
    
class PredictionResponse(BaseModel):
    prediction:str
    confidence:float
    
class PredictionHistory(BaseModel):
    id:int
    news_text:str
    prediction:str
    confidence:str
    created_at:datetime
    
    class config:
        from_attributes=True
from sqlalchemy import Column, Integer, String, Text
from database import Base,engine

# Define the SentimentPrediction model for storing predictions
class SentimentPrediction(Base):
    __tablename__ = "sentiment_predictions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String, index=True)

# Create the tables in the database
Base.metadata.create_all(bind=engine)

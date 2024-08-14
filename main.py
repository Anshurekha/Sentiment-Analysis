import logging
import pickle
import string
import re
import aiosqlite
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from database import SessionLocal, database
from models import SentimentPrediction

# Set up logging configuration
logging.basicConfig(
    filename='app.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Load the model and vectorizer with error handling
try:
    with open('LRmodel.pkl', 'rb') as model_file:
        LRmodel = pickle.load(model_file)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Error loading model.")

try:
    with open('CountVectorizer.pkl', 'rb') as vectorizer_file:
        cv = pickle.load(vectorizer_file)
    logger.info("Vectorizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading vectorizer: {e}")
    raise HTTPException(status_code=500, detail="Error loading vectorizer.")

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to predict sentiment
def predict_sentiment(text):
    try:
        porter_stemmer = PorterStemmer()

        def clean_punctuations(tweet):
            translator = str.maketrans('', '', string.punctuation)
            return tweet.translate(translator)

        def remove_stop_word(text):
            words = remove_stopwords(text)
            return words

        text_cleaned = clean_punctuations(text)
        text_cleaned = remove_stop_word(text_cleaned)
        text_tokens = simple_preprocess(text_cleaned, deacc=True)
        text_stemmed_tokens = [porter_stemmer.stem(word) for word in text_tokens]
        text_stemmed_tokens = ' '.join(text_stemmed_tokens)

        text_transformed = cv.transform([text_stemmed_tokens])
        predicted_sentiment = LRmodel.predict(text_transformed)

        if predicted_sentiment == -1:
            return "Negative"
        elif predicted_sentiment == 0:
            return "Neutral"
        else:
            return "Positive"

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction.")

# FastAPI endpoint for predicting sentiment
@app.post("/predict/")
def predict(text: str, db: Session = Depends(get_db)):
    try:
        logger.info(f"Prediction requested for text: {text}")
        sentiment = predict_sentiment(text)
        logger.info(f"Prediction result: {sentiment}")

        # Save the prediction to the database
        prediction = SentimentPrediction(text=text, sentiment=sentiment)
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        return {"text": text, "sentiment": sentiment}
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error in /predict endpoint.")

# Startup and shutdown events with error handling
@app.on_event("startup")
async def startup():
    try:
        await database.connect()
        logger.info("Application startup and connected to the database")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise HTTPException(status_code=500, detail="Error during startup.")

@app.on_event("shutdown")
async def shutdown():
    try:
        await database.disconnect()
        logger.info("Application shutdown and disconnected from the database")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        raise HTTPException(status_code=500, detail="Error during shutdown.")

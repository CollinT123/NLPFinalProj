from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from model import EntityMLP, token_overlap, entity_overlap, cosine_tfidf, extract_entities
from contextlib import asynccontextmanager
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for models
distilbert_model = None
distilbert_tokenizer = None
entity_overlap_model = None
fake_news_pipeline = None

def square_keep_negative(n):
    """Square a number but keep negative sign if original was negative."""
    if n < 0:
        return -(n * n)
    else:
        return n * n

def get_sentiment_score(text: str, max_length: int = 512):
    """Get sentiment score from DistilBERT model."""
    inputs = distilbert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    p_neg = probs[0].item()
    p_neu = probs[1].item()
    p_pos = probs[2].item()
    
    # Square p_neg and p_pos using square_keep_negative before computing score
    p_neg_squared = square_keep_negative(p_neg)
    p_pos_squared = square_keep_negative(p_pos)
    score = p_pos_squared - p_neg_squared  # in [-1, 1]

    return {
        "p_neg": p_neg,
        "p_neu": p_neu,
        "p_pos": p_pos,
        "score": score,
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    global distilbert_model, distilbert_tokenizer, entity_overlap_model, fake_news_pipeline
    
    # Load DistilBERT model
    model_path = "Models/distilbert-imdb-financial-3class"
    print(f"Loading DistilBERT model from {model_path}...")
    distilbert_tokenizer = AutoTokenizer.from_pretrained(model_path)
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    distilbert_model.eval()
    print("DistilBERT model loaded successfully.")
    
    # Load Entity Overlap model
    entity_model_path = "Models/entity_overlap_model.pt"
    print(f"Loading Entity Overlap model from {entity_model_path}...")
    entity_overlap_model = EntityMLP()
    entity_overlap_model.load_state_dict(torch.load(entity_model_path, map_location=device))
    entity_overlap_model.to(device)
    entity_overlap_model.eval()
    print("Entity Overlap model loaded successfully.")
    
    # Load Fake News Detection model
    print("Loading Fake News Detection model from HuggingFace...")
    fake_news_pipeline = pipeline("text-classification", model="willphan1712/fake_news_detection")
    print("Fake News Detection model loaded successfully.")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")

app = FastAPI(title="News Analysis API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class NewsRequest(BaseModel):
    headline: str
    article: str

class NewsResponse(BaseModel):
    distilbert_sentiment: dict
    entity_overlap: dict
    fake_news_detection: dict

@app.get("/")
def root():
    return {"message": "News Analysis API - Use POST /predict to analyze news"}

@app.post("/predict", response_model=NewsResponse)
def predict(request: NewsRequest):
    """
    Analyze news headline and article using three models:
    1. DistilBERT sentiment analysis (with squared scores)
    2. Entity overlap model
    3. Fake news detection model
    """
    try:
        # 1. DistilBERT Sentiment Analysis
        headline_sentiment = get_sentiment_score(request.headline)
        article_sentiment = get_sentiment_score(request.article)
        
        # Square the scores (only for DistilBERT)
       
        difference = headline_sentiment["score"] - article_sentiment["score"]
        
        
        distilbert_result = {
            "headline": {
                **headline_sentiment,
            },
            "article": {
                **article_sentiment,
            },
            "difference": difference,
        }
        
        # 2. Entity Overlap Model
        features = np.array([[
            token_overlap(request.headline, request.article),
            entity_overlap(request.headline, request.article),
            cosine_tfidf(request.headline, request.article),
            len(extract_entities(request.headline)),
            len(extract_entities(request.article)),
        ]], dtype=np.float32)
        
        x = torch.tensor(features).to(device)
        with torch.no_grad():
            pred_score = entity_overlap_model(x).item()
        
        entity_result = {
            "score": pred_score,
            "prediction": "Not Misleading" if pred_score >= 0.5 else "Misleading",
            "features": {
                "token_overlap": float(features[0][0]),
                "entity_overlap": float(features[0][1]),
                "cosine_tfidf": float(features[0][2]),
                "num_entities_headline": int(features[0][3]),
                "num_entities_body": int(features[0][4])
            }
        }
        
        # 3. Fake News Detection Model
        # Combine headline and article for fake news detection
        combined_text = f"{request.headline} {request.article}"
        fake_news_result_raw = fake_news_pipeline(combined_text, truncation=True, max_length=512)
        
        # Format the result
        label = fake_news_result_raw[0]['label']
        score = fake_news_result_raw[0]['score']
        
        fake_news_result = {
            "label": label,
            "score": score,
            "result": "This is a good heading" if label == "LABEL_1" else "This is a misleading heading"
        }
        
        return NewsResponse(
            distilbert_sentiment=distilbert_result,
            entity_overlap=entity_result,
            fake_news_detection=fake_news_result
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
FastAPI server for Question Predictor

Usage:
    python api.py
    # or
    uvicorn api:app --reload --port 5002

Endpoints:
    GET  /chapters                      - List all chapters
    GET  /predictions/top               - Get top predictions
    GET  /predictions/chapter/{chapter} - Get chapter predictions with questions
    GET  /predictions/topic/{topic}     - Check topic probability with questions
    POST /chat                          - Conversational chat (OpenAI agent)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from src.prediction_service import PredictionService

app = FastAPI(
    title="Question Predictor API",
    description="Physics Board Exam Question Prediction Service with Sample Questions",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service (only once)
prediction_service = PredictionService(subject='physics')

# Optional: OpenAI agent (only if API key is set) - reuses same prediction_service
openai_agent = None
if os.getenv('OPENAI_API_KEY'):
    from src.conversational_agent import QuestionPredictorAgent
    openai_agent = QuestionPredictorAgent(subject='physics', prediction_service=prediction_service)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    reset_history: bool = False


class ChatResponse(BaseModel):
    response: str


class PredictionResponse(BaseModel):
    status: str
    predictions: List[Dict]


# Get unique chapters from data
def get_available_chapters() -> List[str]:
    """Get list of unique chapters"""
    return sorted(prediction_service.df['Chapter_Name'].unique().tolist())


# Endpoints
@app.get("/")
def root():
    return {
        "service": "Question Predictor API",
        "version": "1.0.0",
        "subject": "Physics",
        "endpoints": {
            "chapters": "GET /chapters - List all available chapters",
            "top_predictions": "GET /predictions/top?n=10 - Top predicted topics",
            "chapter_topics": "GET /predictions/chapter/{chapter}?n=5 - Chapter predictions with questions",
            "topic_check": "GET /predictions/topic/{topic} - Check specific topic with questions",
            "chat": "POST /chat - OpenAI conversational agent"
        }
    }


@app.get("/chapters")
def list_chapters():
    """
    Get list of all available chapters for dropdown
    """
    chapters = get_available_chapters()
    return {
        "status": "success",
        "count": len(chapters),
        "chapters": chapters
    }


@app.get("/predictions/top")
def get_top_predictions(n: int = Query(10, ge=1, le=50, description="Number of predictions")):
    """
    Get top N predicted topics with sample questions

    Returns topics ranked by probability with:
    - Probability score and confidence level
    - Historical frequency
    - Question types and sample questions
    """
    predictions = prediction_service.get_top_predictions(top_n=n)
    return {
        "status": "success",
        "count": len(predictions),
        "predictions": predictions
    }


@app.get("/predictions/chapter/{chapter}")
def get_chapter_predictions(
    chapter: str,
    n: int = Query(5, ge=1, le=20, description="Number of topics to return")
):
    """
    Get important topics from a specific chapter with sample questions

    Use /chapters endpoint to get list of valid chapter names.
    Supports partial matching (e.g., "Electric" matches "Electric Charges and Fields")
    """
    predictions = prediction_service.get_chapter_topics(chapter, top_n=n)

    if not predictions:
        available = get_available_chapters()
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No topics found for chapter matching '{chapter}'",
                "available_chapters": available
            }
        )

    return {
        "status": "success",
        "chapter_query": chapter,
        "count": len(predictions),
        "predictions": predictions
    }


@app.get("/predictions/topic/{topic}")
def check_topic(topic: str):
    """
    Check probability for a specific topic with sample questions

    Returns:
    - Probability score and confidence
    - Historical frequency and last appearance
    - Question types with sample questions
    """
    result = prediction_service.check_topic_probability(topic)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Topic '{topic}' not found. Try partial matches like 'Ohm', 'Lens', 'Induction'"
        )

    return {"status": "success", "prediction": result}


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Chat with the OpenAI-powered agent (requires OPENAI_API_KEY env var)

    Example queries:
    - "Electric Charges ke important questions batao"
    - "Ohm's Law aayega kya?"
    - "Most important topics for physics"
    """
    if not openai_agent:
        raise HTTPException(
            status_code=503,
            detail="OpenAI agent not available. Set OPENAI_API_KEY environment variable."
        )

    if request.reset_history:
        openai_agent.reset_conversation()

    response = openai_agent.chat(request.message)
    return ChatResponse(response=response)


@app.get("/stats")
def get_stats():
    """
    Get service statistics
    """
    stats = prediction_service.get_summary_stats()
    stats['available_chapters'] = len(get_available_chapters())
    return stats


@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "subject": "physics"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)

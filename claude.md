# Question Predictor - Project Context

## Overview
ML-powered service that predicts which Physics questions are likely to appear in Class 12 board exams based on historical patterns (2016-2024 data). Returns actual predicted questions, not just topics.

## Architecture

### Data Flow
```
Raw CSV (716 questions)
    ↓ [One-time processing]
Topic Extraction (keyword matching in src/topic_extraction.py)
    ↓
Saved to: data/processed/physics_with_topics.csv

On Server Startup (fast, ~1-2 seconds):
    ↓
Load pre-processed CSV (topics already extracted)
    ↓
Calculate probability scores (frequency × recency × trend)
    ↓
Build question index (group by topic/type)
    ↓
Ready to serve requests
```

### Key Point
- Server does NOT re-run topic extraction on startup
- It loads pre-processed data and calculates simple probability scores
- To reprocess raw data: `python scripts/process_subject.py --subject physics --force`

## Directory Structure
```
question-predictor/
├── api.py                      # FastAPI server (port 5002)
├── start_server.sh             # Start server
├── kill_server.sh              # Stop server
├── config/
│   ├── subjects.py             # Subject registry (physics, chemistry, etc.)
│   └── topics/
│       └── physics_topics.py   # Physics keyword patterns for topic extraction
├── src/
│   ├── prediction_service.py   # Core ML service - calculates probabilities
│   ├── conversational_agent.py # OpenAI-powered chat agent
│   ├── topic_extraction.py     # Extracts topics from question text
│   └── agent_demo.py           # Demo agent (no API key needed)
├── scripts/
│   ├── process_subject.py      # Process raw data for a subject
│   └── validate_subject.py     # Validate predictions against holdout
├── data/
│   ├── raw/                    # Original CSVs
│   │   ├── physics_train_2016_2024.csv
│   │   └── physics_holdout_2025.csv
│   ├── processed/              # Pre-processed with topics
│   │   └── physics_with_topics.csv
│   └── output/                 # RAG exports
│       └── predictions_for_rag.json
├── logs/
│   └── server.log
└── docs/
    └── ADDING_NEW_SUBJECTS.md
```

## API Endpoints (port 5002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chapters` | GET | List all 15 chapters for dropdown |
| `/predictions/top?n=10` | GET | Top N predicted topics with ALL questions |
| `/predictions/chapter/{chapter}?n=5` | GET | Chapter topics with ALL questions |
| `/predictions/topic/{topic}` | GET | Check specific topic with ALL questions |
| `/chat` | POST | OpenAI conversational agent |
| `/health` | GET | Health check |
| `/stats` | GET | Service statistics |

## Response Format
Each prediction includes:
- `topic`: Topic name
- `probability_score`: 0-1 score
- `confidence`: "Very High", "High", "Moderate", "Low"
- `frequency`: "8/8 years"
- `questions_by_type`: ALL historical questions grouped by type (Short Answer, MCQ, etc.)

## ML Model
- **Algorithm**: Optimized Baseline (frequency × recency × trend)
- **Performance**: 77% F1 score, 88% question coverage
- **Validation**: Tested against 2025 holdout data (27 topics, 83 questions)

## Commands

```bash
# Start server
./start_server.sh

# Stop server
./kill_server.sh

# Reprocess data (after adding new questions)
python scripts/process_subject.py --subject physics --force

# Validate predictions
python scripts/validate_subject.py --subject physics

# Test chat endpoint
curl -X POST http://localhost:5002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Ohms Law ke predicted questions batao"}'
```

## Adding New Subjects
See `docs/ADDING_NEW_SUBJECTS.md` for detailed guide:
1. Add raw CSV to `data/raw/`
2. Create topic keywords in `config/topics/{subject}_topics.py`
3. Register in `config/subjects.py`
4. Run `python scripts/process_subject.py --subject {subject}`

## Environment Variables
- `OPENAI_API_KEY`: Required for `/chat` endpoint (set in `.env` file)

## Key Files to Understand
1. `src/prediction_service.py` - Core prediction logic
2. `config/topics/physics_topics.py` - Topic keyword patterns
3. `src/conversational_agent.py` - OpenAI function calling setup

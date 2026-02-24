# ResumeSync — HR Smart Query Engine

## What This Does

HR types natural language → AI extracts intent → MongoDB query → semantic scoring → LLM re-ranking → ranked candidates returned.

```
"I need a Python developer with 2 years experience"
         ↓
  [LLM Intent Parser]
         ↓
  { required: ["Python"], preferred: ["Django","FastAPI"], min_exp: 2 }
         ↓
  [MongoDB hard filter]
         ↓
  [Semantic scorer — skill overlap + experience match]
         ↓
  [LLM re-ranker — top 8 candidates]
         ↓
  Ranked list: id, name, score, reason → Frontend
```

---

## Setup

### 1. Install dependencies
```bash
cd hr_query_backend
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
export MONGO_URI="mongodb+srv://jadhavsushant379_db_user:EjRiiekC4N1iZHg5@cluster0.f4zpb4k.mongodb.net/"
export ANTHROPIC_API_KEY="sk-ant-..."   # get from console.anthropic.com
```

### 3. Run the server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the UI
Open `hr_interface.html` in a browser. Point `API_BASE` in the script to your server URL.

---

## API Endpoints

### POST `/api/hr/query`
Main endpoint for HR queries.

**Request:**
```json
{
  "query": "I need a Python developer with 2 years experience",
  "max_results": 10
}
```

**Response:**
```json
{
  "query": "...",
  "intent_summary": "Looking for a Python backend developer with 2+ years",
  "total_found": 5,
  "candidates": [
    {
      "id": "mongodb_object_id",
      "name": "Sushant Jadhav",
      "email": "...",
      "phone": "...",
      "location": "Pune",
      "current_role": "Full-Stack Developer",
      "years_experience": "Less than 1 year",
      "technical_skills": ["Python", "FastAPI", ...],
      "fit_score": "6",
      "ai_match_score": 7.5,
      "ai_match_reason": "Strong Python skills, FastAPI experience matches..."
    }
  ],
  "search_metadata": { ... }
}
```

### GET `/api/health`
Check if backend, MongoDB, and LLM are connected.

### GET `/api/candidates/all`
Debug — see all candidates in DB.

---

## How Scoring Works

Each candidate gets scored across 3 dimensions:

| Component | Weight | What it checks |
|-----------|--------|---------------|
| Skill overlap | 50% | Required + preferred skill matches |
| Experience match | 30% | Years of experience vs requirement |
| Semantic embedding | 20% | Cosine similarity (if embeddings stored) |

Then LLM adjusts the final score for top 8 candidates with contextual reasoning.

---

## Integration with Your Frontend

The frontend just needs to call:
```js
POST /api/hr/query
{ query: "your natural language query" }
```

Returns array of candidates with `id`, `name`, `email`, `phone`, `ai_match_score`, `ai_match_reason`.

You pass these IDs/names back to your main UI to display candidate cards.

---

## Future Improvements (when more data is available)
- Store actual embeddings in MongoDB alongside candidate data for faster cosine similarity
- Add filters: location, notice_period, salary range as query params
- Batch email outreach to shortlisted candidates
- Analytics dashboard for HR — track query history, shortlist rates
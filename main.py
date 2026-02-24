"""
ResumeSync — Final RAG-based HR Query Engine
=============================================
Architecture:
  1. New resume arrives → auto-embed → insert into Qdrant (real-time, no rebuild)
  2. HR query → Gemini extracts intent → Qdrant semantic retrieval → 
     Gemini RAG reasoning over full profiles → accurate ranked results

Key design decisions:
  - Qdrant runs LOCAL (no server) using ./qdrant_storage folder
  - New candidates auto-indexed via POST /api/candidates/index
  - MongoDB is source of truth, Qdrant is the search index
  - If a candidate is added to MongoDB, call /index to add to Qdrant
  - Full candidate profile is passed to Gemini (RAG) — not just scores
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, json, re, hashlib

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, Range
    )
    QDRANT_OK = True
except ImportError:
    QDRANT_OK = False

try:
    from sentence_transformers import SentenceTransformer
    EMBED_OK = True
except ImportError:
    EMBED_OK = False

try:
    from pymongo import MongoClient
    MONGO_OK = True
except ImportError:
    MONGO_OK = False

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URI       = os.getenv("MONGO_URI", "mongodb+srv://jadhavsushant379_db_user:EjRiiekC4N1iZHg5@cluster0.f4zpb4k.mongodb.net/")
DB_NAME         = "recruitment_db"
MONGO_COLLECTION= "candidates"
GEMINI_KEY      = os.getenv("GEMINI_API_KEY", "")
QDRANT_PATH     = "./qdrant_storage"
COLLECTION      = "candidates"
VECTOR_DIM      = 384

app = FastAPI(title="ResumeSync RAG Engine", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Singletons ────────────────────────────────────────────────────────────────
_qdrant = None
_model  = None
_mongo  = None

def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(path=QDRANT_PATH)
        # Create collection if not exists
        existing = [c.name for c in _qdrant.get_collections().collections]
        if COLLECTION not in existing:
            _qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            print(f"✅ Created Qdrant collection: {COLLECTION}")
    return _qdrant

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_mongo():
    global _mongo
    if _mongo is None and MONGO_OK:
        _mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    return _mongo


# ── Experience normalization ──────────────────────────────────────────────────
EXP_MAP = {
    "less than 1 year": 0.5, "0-1 years": 0.5, "fresher": 0.0,
    "1-2 years": 1.5, "2-3 years": 2.5, "3-5 years": 4.0,
    "5+ years": 6.0, "5-7 years": 6.0, "7+ years": 8.0,
    "7-10 years": 8.5, "10+ years": 11.0, "10-15 years": 12.0, "15+ years": 16.0,
}

def exp_to_float(exp_str: str) -> float:
    s = (exp_str or "").lower().strip()
    for k, v in EXP_MAP.items():
        if k in s:
            return v
    nums = re.findall(r"\d+", s)
    return float(nums[0]) if nums else 0.0

def mongo_id_to_int(mongo_id: str) -> int:
    """Convert MongoDB ObjectId string to stable integer for Qdrant point ID."""
    return int(hashlib.md5(mongo_id.encode()).hexdigest()[:8], 16)


# ── Candidate → Qdrant payload ────────────────────────────────────────────────
def build_candidate_payload(doc: dict) -> tuple[str, dict]:
    """
    Returns (search_text, payload) for a candidate document.
    search_text is what gets embedded.
    payload is stored in Qdrant for retrieval.
    """
    out   = doc.get("output", {})
    cand  = out.get("candidate", {})
    summ  = out.get("summary", {})
    fit   = out.get("fit_score", {})

    skills       = summ.get("technical_skills", [])
    achievements = summ.get("key_achievements", [])
    reasoning    = fit.get("reasoning", [])
    role         = summ.get("current_role", "")
    exp_str      = summ.get("years_experience", "")
    exp_num      = exp_to_float(exp_str)

    # Rich text for embedding — quality here directly impacts search quality
    search_text = (
        f"Role: {role}. "
        f"Experience: {exp_str}. "
        f"Skills: {', '.join(skills)}. "
        f"Work: {'. '.join(achievements[:3])}. "
        f"Profile: {'. '.join(reasoning[:2])}. "
        f"Location: {cand.get('location', '')}."
    )

    mongo_id = str(doc.get("_id", ""))

    payload = {
        "mongo_id":               mongo_id,
        "first_name":             cand.get("first_name", ""),
        "last_name":              cand.get("last_name", ""),
        "email":                  cand.get("email", ""),
        "phone":                  cand.get("phone", ""),
        "location":               cand.get("location", ""),
        "current_role":           role,
        "years_experience":       exp_str,
        "exp_years_num":          exp_num,           # ← Qdrant range filter uses this
        "technical_skills":       [s.lower() for s in skills],
        "technical_skills_display": skills,
        "key_achievements":       achievements,
        "fit_score":              fit.get("overall", "N/A"),
        "fit_reasoning":          reasoning,
        "expected_salary":        cand.get("expected_salary", ""),
        "notice_period":          cand.get("notice_period", ""),
        # Full profile for RAG context — Gemini reads this
        "full_profile":           search_text,
    }

    return search_text, payload


# ── STEP 1: Intent Extraction ─────────────────────────────────────────────────
INTENT_PROMPT = """You convert HR recruiter queries into JSON search parameters.

HR Query: "{query}"

Return ONLY this JSON, no markdown:
{{
  "required_skills": [],
  "preferred_skills": [],
  "min_exp": 0,
  "max_exp": null,
  "is_fresher": false,
  "semantic_query": "",
  "intent_summary": ""
}}

Instructions:
- required_skills: skills explicitly mentioned ["Python", "React"]  
- preferred_skills: ecosystem skills to broaden search. Python→["Django","FastAPI","Flask"]. React→["Next.js","TypeScript","Redux"]
- min_exp: number. "above 5"=5, "5+"=5, "2 years"=2, "fresher"=0, "entry level"=0
- max_exp: number or null. "fresher"=1, "entry level"=1, "junior"=2, else null
- is_fresher: true only for fresher/entry-level/fresh graduate/0 experience
- semantic_query: rewrite query as a rich candidate profile description for vector search.
  Example: "I want Python dev" → "Experienced Python developer skilled in Django FastAPI REST APIs backend development"
- intent_summary: plain English summary of what HR wants
"""

def extract_intent(query: str) -> dict:
    if GEMINI_OK and GEMINI_KEY:
        try:
            genai.configure(api_key=GEMINI_KEY)
            m    = genai.GenerativeModel("gemini-2.5-flash")
            resp = m.generate_content(INTENT_PROMPT.format(query=query))
            text = re.sub(r"```json|```", "", resp.text).strip()
            intent = json.loads(text)
            print(f"\n📋 Intent: {json.dumps(intent, indent=2)}")
            return intent
        except Exception as e:
            print(f"⚠️ Gemini intent failed: {e}")

    return _fallback_intent(query)


def _fallback_intent(query: str) -> dict:
    q = query.lower()

    fresher_words = ["fresher", "fresh graduate", "entry level", "0 experience",
                     "no experience", "just graduated", "beginner", "junior fresher"]
    is_fresher = any(w in q for w in fresher_words)

    min_exp, max_exp = 0, None
    if is_fresher:
        min_exp, max_exp = 0, 1
    else:
        for pat in [
            r"(?:above|more than|over|minimum|atleast|at least)\s*(\d+)",
            r"(\d+)\s*\+\s*(?:year|yr)",
            r"(\d+)\s*(?:year|yr)",
        ]:
            m = re.search(pat, q)
            if m:
                min_exp = int(m.group(1))
                break

    SKILL_MAP = {
        "python":           (["Python"],          ["Django", "FastAPI", "Flask", "SQLAlchemy"]),
        "react":            (["React"],            ["Next.js", "Redux", "TypeScript"]),
        "node":             (["Node.js"],          ["Express.js", "NestJS"]),
        "java":             (["Java"],             ["Spring Boot", "Hibernate"]),
        "angular":          (["Angular"],          ["TypeScript", "RxJS"]),
        "flutter":          (["Flutter"],          ["Dart", "Firebase"]),
        "machine learning": (["Machine Learning"], ["TensorFlow", "PyTorch", "scikit-learn"]),
        "deep learning":    (["Deep Learning"],    ["TensorFlow", "PyTorch", "Keras"]),
        "devops":           (["Docker"],           ["Kubernetes", "AWS", "Terraform"]),
        "typescript":       (["TypeScript"],       ["React", "Node.js"]),
        "golang":           (["Go"],               ["gRPC", "Docker"]),
        "swift":            (["Swift"],            ["iOS", "Xcode"]),
        "kotlin":           (["Kotlin"],           ["Android"]),
        "data science":     (["Python"],           ["Pandas", "NumPy", "SQL", "ML"]),
        "blockchain":       (["Solidity"],         ["Web3.js", "Ethereum"]),
        "aws":              (["AWS"],              ["Docker", "Terraform", "Kubernetes"]),
    }

    required, preferred = [], []
    for key, (req, pref) in SKILL_MAP.items():
        if key in q:
            required.extend(req)
            preferred.extend(pref)

    exp_part   = "fresher entry level 0 experience" if is_fresher else f"{min_exp}+ years experience"
    skill_part = " ".join(required + preferred[:3])
    semantic_q = f"{skill_part} developer {exp_part} {q}".strip()

    return {
        "required_skills": list(set(required)),
        "preferred_skills": list(set(preferred)),
        "min_exp": min_exp,
        "max_exp": max_exp,
        "is_fresher": is_fresher,
        "semantic_query": semantic_q,
        "intent_summary": f"Looking for {'fresher' if is_fresher else f'{min_exp}+ yr'} {', '.join(required) or 'software'} developer",
    }


# ── STEP 2: Qdrant retrieval ──────────────────────────────────────────────────
def retrieve_candidates(intent: dict, limit: int = 20) -> tuple[list, str]:
    """
    Returns (candidates_payload_list, retrieval_mode)
    Tries strict → relaxed → pure semantic
    """
    qdrant = get_qdrant()
    model  = get_model()

    semantic_q = intent.get("semantic_query") or intent.get("intent_summary", "")
    query_vec  = model.encode(semantic_q).tolist()

    is_fresher = intent.get("is_fresher", False)
    min_exp    = float(intent.get("min_exp") or 0)
    max_exp    = intent.get("max_exp")

    # Build experience filter
    exp_filter = None
    if is_fresher:
        exp_filter = FieldCondition(key="exp_years_num", range=Range(lte=1.0))
    elif min_exp > 0:
        rng = {"gte": min_exp * 0.85}
        if max_exp:
            rng["lte"] = float(max_exp) * 1.15
        exp_filter = FieldCondition(key="exp_years_num", range=Range(**rng))

    def do_search(qdrant_filter, fetch_limit):
        return qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vec,
            query_filter=qdrant_filter,
            limit=fetch_limit,
            with_payload=True,
            score_threshold=0.1,
        )

    required = [s.lower() for s in intent.get("required_skills", [])]

    # ── Attempt 1: Strict — experience filter + skill match ───────────────────
    results = do_search(
        Filter(must=[exp_filter]) if exp_filter else None,
        limit
    )

    if required and results:
        skill_matched = [
            r for r in results
            if any(
                any(req in cs or cs in req for cs in r.payload.get("technical_skills", []))
                for req in required
            )
        ]
        if skill_matched:
            print(f"✅ Strict retrieval: {len(skill_matched)} candidates")
            return [r.payload for r in skill_matched], "strict"

    if results and not required:
        print(f"✅ Experience-filtered retrieval: {len(results)} candidates")
        return [r.payload for r in results], "exp_filtered"

    # ── Attempt 2: Relaxed — skills only, no experience filter ───────────────
    if required:
        results = do_search(None, limit * 2)
        skill_matched = [
            r for r in results
            if any(
                any(req in cs or cs in req for cs in r.payload.get("technical_skills", []))
                for req in required
            )
        ]
        if skill_matched:
            print(f"⚠️ Relaxed (skills only): {len(skill_matched)} candidates")
            return [r.payload for r in skill_matched], "relaxed_no_exp"

    # ── Attempt 3: Pure semantic — ONLY for real tech queries ─────────────────
    KNOWN_TERMS = [
        "developer", "engineer", "architect", "analyst", "scientist", "intern",
        "python", "java", "react", "node", "angular", "flutter", "kotlin", "swift",
        "go", "rust", "typescript", "javascript", "sql", "aws", "gcp", "azure",
        "devops", "machine learning", "deep learning", "data", "backend", "frontend",
        "fullstack", "full stack", "mobile", "cloud", "security", "blockchain",
        "ml", "ai", "nlp", "senior", "junior", "lead", "fresher",
        "php", "ruby", "scala", "c++", "c#", "dotnet", ".net", "spring", "django",
        "docker", "kubernetes", "microservices", "api", "database", "mongodb"
    ]
    query_words = intent.get("semantic_query", "").lower() + " " + intent.get("intent_summary", "").lower()
    has_real_terms = any(term in query_words for term in KNOWN_TERMS)

    if not has_real_terms:
        print(f"❌ No recognizable tech terms — returning empty")
        return [], "no_match"

    results = do_search(None, limit)
    qualified = [r for r in results if r.score >= 0.40]

    if not qualified:
        print(f"❌ Semantic scores too low — no match")
        return [], "no_match"

    print(f"⚠️ Pure semantic fallback: {len(qualified)} above threshold")
    return [r.payload for r in qualified], "pure_semantic"


# ── STEP 3: RAG — Gemini reasons over full profiles ───────────────────────────
RAG_PROMPT = """You are a senior technical recruiter. 
The HR has given a requirement and you have retrieved candidate profiles from the database.
Your job is to evaluate each candidate CAREFULLY and return ranked results.

HR REQUIREMENT:
"{query}"

CANDIDATE PROFILES:
{profiles}

Instructions:
- Read each profile carefully against the HR requirement
- Score honestly: 
    9-10 = perfect match
    7-8  = strong match with minor gaps  
    5-6  = partial match, notable gaps
    3-4  = weak match
    1-2  = does not meet requirement at all
- For freshers query: candidates with more than 1 year experience should score MAX 3
- For senior roles: candidates without required experience should score MAX 3
- reason: one specific sentence mentioning what matches and what doesn't

Return ONLY a JSON array, no markdown, no explanation:
[
  {{"id": "mongo_id", "score": 8.5, "reason": "specific one-line assessment"}},
  ...
]
"""

def rag_evaluate(candidates: list, query: str) -> list:
    """
    Core RAG step: pass full candidate profiles to Gemini for reasoning.
    Returns candidates sorted by Gemini score.
    """
    if not GEMINI_OK or not GEMINI_KEY or not candidates:
        # No Gemini — sort by Qdrant semantic score if available
        return sorted(candidates, key=lambda x: x.get("_sem_score", 0), reverse=True)

    # Build profile text for each candidate — this is the RAG context
    profiles_text = ""
    for i, c in enumerate(candidates[:15]):  # max 15 to Gemini
        profiles_text += f"""
--- Candidate {i+1} ---
ID: {c.get('mongo_id', '')}
Name: {c.get('first_name', '')} {c.get('last_name', '')}
Role: {c.get('current_role', '')}
Experience: {c.get('years_experience', '')} ({c.get('exp_years_num', 0)} years)
Skills: {', '.join(c.get('technical_skills_display', []))}
Achievements: {'; '.join(c.get('key_achievements', [])[:3])}
Location: {c.get('location', '')}
Notice Period: {c.get('notice_period', '')}
Expected Salary: {c.get('expected_salary', '')}
"""

    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp  = model.generate_content(
            RAG_PROMPT.format(query=query, profiles=profiles_text)
        )
        text    = re.sub(r"```json|```", "", resp.text).strip()
        results = json.loads(text)
        print(f"✅ Gemini RAG evaluated {len(results)} candidates")

        # Map scores back to candidates
        score_map = {r["id"]: r for r in results}
        for c in candidates:
            mid = c.get("mongo_id", "")
            if mid in score_map:
                c["_rag_score"]  = score_map[mid]["score"]
                c["_rag_reason"] = score_map[mid]["reason"]
            else:
                c["_rag_score"]  = c.get("_sem_score", 5.0)
                c["_rag_reason"] = "Evaluated by semantic similarity."

        return sorted(candidates, key=lambda x: x.get("_rag_score", 0), reverse=True)

    except Exception as e:
        print(f"⚠️ Gemini RAG failed: {e}")
        return sorted(candidates, key=lambda x: x.get("_sem_score", 0), reverse=True)


# ── Pydantic models ───────────────────────────────────────────────────────────
class HRQueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10

class IndexRequest(BaseModel):
    mongo_id: str  # index a single candidate by their MongoDB ID

class CandidateResult(BaseModel):
    id: str
    name: str
    email: str
    phone: str
    location: str
    current_role: str
    years_experience: str
    technical_skills: list[str]
    fit_score: str
    ai_score: float
    ai_reason: str

class HRQueryResponse(BaseModel):
    query: str
    intent_summary: str
    total_found: int
    candidates: list[CandidateResult]
    warning: Optional[str] = None
    debug: dict


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.post("/api/hr/query", response_model=HRQueryResponse)
async def hr_query(req: HRQueryRequest):
    """Main HR query endpoint — RAG pipeline."""
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query cannot be empty")

    print(f"\n{'='*60}\nHR Query: {query}")

    # 1. Extract intent
    intent = extract_intent(query)

    # 2. Retrieve candidates from Qdrant
    try:
        candidates, mode = retrieve_candidates(intent, limit=max(20, req.max_results * 2))
    except Exception as e:
        raise HTTPException(500, f"Retrieval error: {e}. Run: python init_qdrant.py")

    if not candidates or mode == "no_match":
        return HRQueryResponse(
            query=query,
            intent_summary=intent.get("intent_summary", query),
            total_found=0,
            candidates=[],
            warning="No matching candidates found for your requirement. Try different skills or adjust experience level.",
            debug={"intent": intent, "mode": mode}
        )

    # 3. RAG evaluation — Gemini reads full profiles and scores
    ranked = rag_evaluate(candidates, query)

    # 4. Take top N and format
    top_n = ranked[:req.max_results]
    out   = []
    for c in top_n:
        out.append(CandidateResult(
            id               = c.get("mongo_id", ""),
            name             = f"{c.get('first_name','')} {c.get('last_name','')}".strip(),
            email            = c.get("email", ""),
            phone            = c.get("phone", ""),
            location         = c.get("location", ""),
            current_role     = c.get("current_role", ""),
            years_experience = c.get("years_experience", ""),
            technical_skills = c.get("technical_skills_display", []),
            fit_score        = c.get("fit_score", "N/A"),
            ai_score         = round(float(c.get("_rag_score", c.get("_sem_score", 5.0))), 1),
            ai_reason        = c.get("_rag_reason", ""),
        ))

    # Warning if we fell back
    warning = None
    if mode == "relaxed_no_exp":
        warning = f"⚠️ No candidates matched exact experience requirement — showing skill-matched results."
    elif mode == "pure_semantic":
        warning = f"⚠️ No exact matches found — showing semantically similar candidates."

    return HRQueryResponse(
        query=query,
        intent_summary=intent.get("intent_summary", query),
        total_found=len(candidates),
        candidates=out,
        warning=warning,
        debug={"intent": intent, "retrieval_mode": mode, "retrieved": len(candidates), "gemini_rag": GEMINI_OK and bool(GEMINI_KEY)}
    )


@app.post("/api/candidates/index-all")
async def index_all_candidates():
    """
    Index ALL candidates from MongoDB into Qdrant.
    Run this once on first setup, and whenever you want to sync.
    New candidates should use /api/candidates/index instead.
    """
    try:
        mongo  = get_mongo()
        qdrant = get_qdrant()
        model  = get_model()
        col    = mongo[DB_NAME][MONGO_COLLECTION]
        docs   = list(col.find({}))

        if not docs:
            return {"status": "error", "message": "No candidates in MongoDB"}

        points = []
        for doc in docs:
            search_text, payload = build_candidate_payload(doc)
            vector   = model.encode(search_text).tolist()
            point_id = mongo_id_to_int(str(doc["_id"]))
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        qdrant.upsert(collection_name=COLLECTION, points=points)
        return {
            "status":   "success",
            "indexed":  len(points),
            "message":  f"Indexed {len(points)} candidates into Qdrant"
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/candidates/index")
async def index_one_candidate(req: IndexRequest):
    """
    Index a SINGLE new candidate from MongoDB into Qdrant.
    Call this every time a new resume is added to MongoDB.
    This is how the system stays real-time — no rebuild needed.
    """
    try:
        from bson import ObjectId
        mongo  = get_mongo()
        qdrant = get_qdrant()
        model  = get_model()
        col    = mongo[DB_NAME][MONGO_COLLECTION]

        doc = col.find_one({"_id": ObjectId(req.mongo_id)})
        if not doc:
            raise HTTPException(404, f"Candidate {req.mongo_id} not found in MongoDB")

        search_text, payload = build_candidate_payload(doc)
        vector   = model.encode(search_text).tolist()
        point_id = mongo_id_to_int(req.mongo_id)

        qdrant.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )

        name = f"{payload['first_name']} {payload['last_name']}"
        return {
            "status":  "success",
            "message": f"Indexed {name} into Qdrant",
            "exp":     payload["exp_years_num"],
            "skills":  payload["technical_skills_display"][:5],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
async def health():
    qdrant_ok, count = False, 0
    try:
        info      = get_qdrant().get_collection(COLLECTION)
        qdrant_ok = True
        count     = info.points_count
    except: pass

    mongo_ok = False
    try:
        get_mongo().admin.command("ping")
        mongo_ok = True
    except: pass

    ready = qdrant_ok and count > 0 and EMBED_OK

    return {
        "status":         "ready" if ready else "setup_needed",
        "qdrant":         qdrant_ok,
        "vectors_stored": count,
        "mongodb":        mongo_ok,
        "embeddings":     EMBED_OK,
        "gemini":         GEMINI_OK and bool(GEMINI_KEY),
        "next_step":      "Ready for queries!" if ready else "Call POST /api/candidates/index-all first",
    }


@app.get("/api/candidates/count")
async def count():
    try:
        info = get_qdrant().get_collection(COLLECTION)
        return {"total_indexed": info.points_count}
    except Exception as e:
        raise HTTPException(500, str(e))
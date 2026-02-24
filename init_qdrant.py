"""
init_qdrant.py
--------------
Run this ONCE to load all MongoDB candidates into Qdrant.
After this, new candidates should be indexed via:
  POST /api/candidates/index  {"mongo_id": "..."}

Usage:
  python init_qdrant.py
"""

import urllib.request
import json

API_BASE = "http://localhost:8000"

def main():
    print("ResumeSync — Initializing Qdrant from MongoDB")
    print("=" * 50)
    print("Make sure the server is running first!")
    print("  uvicorn main:app --reload --port 8000")
    print("=" * 50)

    # Check health first
    print("\n1. Checking server health...")
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/health", timeout=5) as r:
            health = json.loads(r.read())
            print(f"   MongoDB: {health.get('mongodb')}")
            print(f"   Embeddings: {health.get('embeddings')}")
            print(f"   Gemini: {health.get('gemini')}")
    except Exception as e:
        print(f"   ❌ Server not reachable: {e}")
        print("   Start server first: uvicorn main:app --reload --port 8000")
        return

    # Index all candidates
    print("\n2. Indexing all candidates from MongoDB into Qdrant...")
    print("   (This may take 1-2 minutes for embedding generation)")

    try:
        req = urllib.request.Request(
            f"{API_BASE}/api/candidates/index-all",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b"{}"
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
            print(f"   ✅ {result.get('message')}")
    except Exception as e:
        print(f"   ❌ Indexing failed: {e}")
        return

    # Verify
    print("\n3. Verifying...")
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/candidates/count", timeout=5) as r:
            count = json.loads(r.read())
            print(f"   ✅ Vectors in Qdrant: {count.get('total_indexed')}")
    except:
        pass

    print("\n✅ Done! System is ready.")
    print("\nTest queries:")
    print("  fresher who knows React")
    print("  Python developer with 3+ years experience")
    print("  Senior Java developer above 7 years")
    print("  Machine learning engineer with PyTorch")


if __name__ == "__main__":
    main()
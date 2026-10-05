import os
import hmac
import hashlib
import json
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, status
from google import genai
from google.genai import types
import redis.asyncio as aioredis

app = FastAPI(title="GitHub Automated PR Reviewer API with Redis Caching", version="1.1.0")

# Infrastructure Configurations
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "super_secret_webhook_token")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Global clients initialization
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

redis_client: aioredis.Redis = None

@app.on_event("startup")
async def startup_event():
    """Initialize the asynchronous Redis connection pool on application startup."""
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close the Redis connection pool on application shutdown."""
    global redis_client
    if redis_client:
        await redis_client.close()

async def verify_github_signature(request: Request):
    """Cryptographically verify that the webhook signature matches the computed HMAC-SHA256."""
    signature_header = request.headers.get("X-Hub-Signature-256")
    if not signature_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Hub-Signature-256 header")
    
    if not signature_header.startswith("sha256="):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature format. Must be sha256.")
        
    expected_signature = signature_header.split("sha256=")[-1]
    payload_body = await request.body()
    
    computed_mac = hmac.new(GITHUB_WEBHOOK_SECRET.encode(), msg=payload_body, digestmod=hashlib.sha256)
    computed_signature = computed_mac.hexdigest()
    
    if not hmac.compare_digest(computed_signature, expected_signature):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cryptographic signature verification failed.")

async def process_pr_review_task(commit_sha: str, pr_data: dict):
    """
    Asynchronous background worker function that performs the heavy lifting:
    Checks Redis cache, talks to Gemini, and updates cache atomically.
    """
    cache_key = f"pr_review:{commit_sha}"
    
    try:
        # 1. Double check cache hit inside worker to prevent race conditions during heavy traffic
        cached_review = await redis_client.get(cache_key)
        if cached_review:
            print(f"[Worker Cache Hit] Review already completed for commit {commit_sha}. Skipping API invocation.")
            return

        if not gemini_client:
            print("[Worker Error] Gemini Client not initialized. Skipping review computation.")
            return

        # Mock incoming structural diff payload
        mock_diff = """
        diff --git a/database.py b/database.py
        --- a/database.py
        +++ b/database.py
        @@ -10,4 +10,8 @@ def get_user_by_id(db_conn, user_id):
         def get_user_by_query_unsafe(db_conn, input_string):
        +    cursor = db_conn.cursor()
        +    cursor.execute(f"SELECT * FROM users WHERE name = '{input_string}'")
        +    return cursor.fetchall()
        """

        system_instruction = (
            "You are an expert Senior Staff Security and Performance Backend Engineer.\n"
            "Analyze the provided Git diff block payload carefully.\n"
            "Identify severe bugs, critical vulnerabilities (such as SQL injection), and critical architectural violations.\n"
            "Return your critical response structured in clean markdown format."
        )

        # 2. Invoke Gemini Inference via modern SDK 
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Analyze this code diff payload:\n{mock_diff}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )
        )
        
        generated_review = response.text
        print(f"[Worker Success] Generated Review Output for {commit_sha}")

        # 3. Write back to Redis Cache with a 24-hour TTL (86400 seconds) to save API budget
        await redis_client.setex(cache_key, 86400, generated_review)
        
        # Production extension: Post review back to GitHub Pull Request comments API endpoint here.

    except Exception as e:
        print(f"[Worker Exception] Failed to execute LLM evaluation or caching layer: {e}")

@app.post("/webhook/github", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(verify_github_signature)])
async def github_webhook_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Main ingestion endpoint for GitHub Webhook events.
    Utilizes Redis to check for redundant review tasks before queuing background operations.
    """
    payload = await request.json()
    action = payload.get("action")
    
    if action in ["opened", "synchronize"]:
        # Extract unique commit SHA to ensure idempotency across repetitive webhooks
        commit_sha = payload.get("pull_request", {}).get("head", {}).get("sha", "default_mock_sha")
        cache_key = f"pr_review:{commit_sha}"
        
        # Immediate Cache Lookahead (Fast Path)
        cached_review = await redis_client.get(cache_key)
        if cached_review:
            return {
                "status": "cached",
                "detail": f"Review for commit {commit_sha} fetched from cache. Skipping computational task execution."
            }
        
        # Offload computationally heavy tasks to a background worker loop immediately
        background_tasks.add_task(process_pr_review_task, commit_sha, payload)
        return {"status": "accepted", "detail": "Pull Request event queued for architectural processing"}
        
    return {"status": "ignored", "detail": "Action event bypassed"}


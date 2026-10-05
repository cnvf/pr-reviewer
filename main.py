from dotenv import load_dotenv
load_dotenv()

import os
import hmac
import hashlib
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException, status
from google import genai
from google.genai import types
import redis.asyncio as aioredis
import httpx                 
from github import Github     
from github import Auth       

app = FastAPI(title="Production Async GitHub PR Reviewer API", version="1.3.0")

# Infrastructure Configurations
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "super_secret_webhook_token")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 

# Global clients initialization
redis_client: aioredis.Redis = None
http_client: httpx.AsyncClient = None  
gemini_async_client = None  # ◄ CHANGED: Dedicated explicit async client reference

@app.on_event("startup")
async def startup_event():
    """Initialize asynchronous connection pools on application startup."""
    global redis_client, http_client, gemini_async_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    http_client = httpx.AsyncClient()  
    
    # ◄ FIXED: Properly initialize the dedicated async client using .aio on startup
    if GEMINI_API_KEY:
        gemini_async_client = genai.Client(api_key=GEMINI_API_KEY).aio
    else:
        print("[Startup Warning] GEMINI_API_KEY environment variable missing!")

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close asynchronous pools on application shutdown."""
    global redis_client, http_client
    if redis_client:
        await redis_client.close()
    if http_client:
        await http_client.aclose()  

def verify_github_signature_raw(payload_body: bytes, signature_header: str) -> bool:
    """Cryptographically verify that the webhook signature matches the computed HMAC-SHA256."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected_signature = signature_header.split("sha256=")[-1]
    computed_mac = hmac.new(GITHUB_WEBHOOK_SECRET.encode(), msg=payload_body, digestmod=hashlib.sha256)
    return hmac.compare_digest(computed_mac.hexdigest(), expected_signature)

async def process_pr_review_task(commit_sha: str, pr_data: dict):
    """
    Hardened Asynchronous background worker loop.
    Fully non-blocking execution via httpx, aioredis, and gemini_async_client.
    """
    # This print statement is guaranteed to run the microsecond the task is spawned!
    print(f"\n🚀 [WORKER STARTING] Processing initiated for commit: {commit_sha}", flush=True)
    cache_key = f"pr_review:{commit_sha}"
    
    try:
        # 1. Double check cache hit inside worker
        cached_review = await redis_client.get(cache_key)
        if cached_review:
            print(f"[Worker Cache Hit] Review already completed for commit {commit_sha}. Skipping.", flush=True)
            return

        if not gemini_async_client:
            print("[Worker Error] Gemini Async Client not initialized. Exiting task.", flush=True)
            return

        # 2. Dynamically fetch real diff via HTTPX
        pull_request_obj = pr_data.get("pull_request", {})
        diff_url = pull_request_obj.get("diff_url")
        
        if not diff_url:
            print("[Worker Error] Could not find diff_url in payload.", flush=True)
            return
            
        print(f"[Worker Progress] Downloading raw diff text from GitHub...", flush=True)
        headers = {"Accept": "application/vnd.github.v3.diff"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
            
        diff_response = await http_client.get(diff_url, headers=headers, follow_redirects=True)
        if diff_response.status_code != 200:
            print(f"[Worker Error] Failed to fetch diff from GitHub via HTTPX. Code: {diff_response.status_code}", flush=True)
            return
            
        real_diff = diff_response.text
        if len(real_diff) > 50000: 
            real_diff = real_diff[:50000] + "\n\n[Warning: Diff truncated due to token size constraints]"

        system_instruction = (
            "You are an expert Senior Staff Security and Performance Backend Engineer.\n"
            "Analyze the provided Git diff block payload carefully.\n"
            "Identify severe bugs, critical vulnerabilities (such as SQL injection), and critical architectural violations.\n"
            "Return your critical response structured in clean markdown format."
        )

        # 3. Invoke Gemini Inference via MODERN NON-BLOCKING ASYNC SDK (.aio)
        print(f"[Worker Progress] Forwarding code blocks to Gemini 3.8-Flash engine...", flush=True)
        
        # Using the cleanly instantiated global async client directly
        response = await gemini_async_client.models.generate_content(
            model='gemini-3.8-flash',  
            contents=f"Analyze this code diff payload:\n{real_diff}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        generated_review = response.text
        print(f"[Worker Success] Successfully generated LLM report for {commit_sha}", flush=True)

        # 4. Write back to Redis Cache with a 24-hour TTL
        await redis_client.setex(cache_key, 86400, generated_review)
        
        # 5. Post comment back to GitHub Pull Request
        if GITHUB_TOKEN:
            repo_full_name = pr_data.get("repository", {}).get("full_name")
            pr_number = pr_data.get("number")
            
            print(f"[Worker Progress] Dispatching comment blocks to GitHub PR #{pr_number}...", flush=True)
            auth = Auth.Token(GITHUB_TOKEN)
            g = Github(auth=auth)
            
            repo = g.get_repo(repo_full_name)
            pull_request = repo.get_pull(pr_number)
            
            # Executing synchronous post isolated inside the decoupled async worker task 
            pull_request.create_issue_comment(generated_review)
            print(f"🎉 [GitHub Success] Code Review completely posted to PR #{pr_number}!", flush=True)
        else:
            print("[Warning] GITHUB_TOKEN not found. Skipping GitHub comment submission.", flush=True)

    except Exception as worker_error:
        # Catches any underlying library initialization exceptions and prints the raw error trace
        print(f"\n❌ [FATAL WORKER CRASH] An exception occurred inside the background worker thread!", flush=True)
        print(f"Error Details: {str(worker_error)}", flush=True)
        import traceback
        traceback.print_exc()

@app.post("/webhook/github", status_code=status.HTTP_202_ACCEPTED)
async def github_webhook_endpoint(request: Request):
    """
    Main ingestion endpoint for GitHub Webhook events.
    Verifies signatures and fires background tasks natively via asyncio.create_task.
    """
    signature_header = request.headers.get("X-Hub-Signature-256")
    body_bytes = await request.body()
    
    if not verify_github_signature_raw(body_bytes, signature_header):
        print(f"[Security Alert] Cryptographic check failed. Check secret values.", flush=True)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cryptographic verification failed.")
        
    try:
        payload = json.loads(body_bytes.decode('utf-8'))
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")
        
    action = payload.get("action")
    
    if action in ["opened", "synchronize"]:
        commit_sha = payload.get("pull_request", {}).get("head", {}).get("sha", "default_mock_sha")
        cache_key = f"pr_review:{commit_sha}"
        
        cached_review = await redis_client.get(cache_key)
        if cached_review:
            print(f"[Endpoint Log] Cache hit detected for {commit_sha}. Returning instantly.", flush=True)
            return {
                "status": "cached",
                "detail": f"Review for commit {commit_sha} fetched from cache."
            }
        
        print(f"[Endpoint Log] Enqueueing process task for commit: {commit_sha}...", flush=True)
        
        # Native, zero-overhead background scheduling
        asyncio.create_task(process_pr_review_task(commit_sha, payload))
        
        return {"status": "accepted", "detail": "Pull Request event queued natively via asyncio"}
        
    return {"status": "ignored", "detail": "Action event bypassed"}

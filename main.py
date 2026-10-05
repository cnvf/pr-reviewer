import os
import hmac
import hashlib
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, status
from google import genai
from google.genai import types
import redis.asyncio as aioredis
import httpx
from github import Github     
from github import Auth       

load_dotenv()  # Load environment variables from .env file

app = FastAPI(title="Fully Async GitHub PR Reviewer API", version="1.2.0")

# Infrastructure Configurations
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "super_secret_webhook_token")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 

# Global clients initialization
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

redis_client: aioredis.Redis = None
http_client: httpx.AsyncClient = None

@app.on_event("startup")
async def startup_event():
    """Initialize asynchronous connection pools on application startup."""
    global redis_client, http_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    http_client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close asynchronous pools on application shutdown."""
    global redis_client, http_client
    if redis_client:
        await redis_client.close()
    if http_client:
        await http_client.aclose()

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
    Hardened Asynchronous background worker loop.
    Fully non-blocking execution via httpx, aioredis, and gemini_client.aio.
    """
    cache_key = f"pr_review:{commit_sha}"
    print(f"[Worker Active] Commencing evaluation task for commit: {commit_sha}", flush=True)
    
    try:
        # 1. Double check cache hit inside worker
        cached_review = await redis_client.get(cache_key)
        if cached_review:
            print(f"[Worker Cache Hit] Review already completed for commit {commit_sha}. Skipping API invocation.", flush=True)
            return

        if not gemini_client:
            print("[Worker Error] Gemini Client not initialized. Skipping review computation.", flush=True)
            return

        # 2. Dynamically fetch real diff via HTTPX
        pull_request_obj = pr_data.get("pull_request", {})
        diff_url = pull_request_obj.get("diff_url")
        
        if not diff_url:
            print("[Worker Error] Could not find diff_url in payload.", flush=True)
            return
            
        headers = {"Accept": "application/vnd.github.v3.diff"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
            
        diff_response = await http_client.get(diff_url, headers=headers)
        if diff_response.status_code != 200:
            print(f"[Worker Error] Failed to fetch diff from GitHub via HTTPX: {diff_response.status_code}", flush=True)
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

        # 3. Invoke Gemini Inference via MODERN NON-BLOCKING ASYNC SDK (.aio) ◄ NEW
        print(f"[Worker Progress] Handshaking with Gemini 3.8-Flash engine...", flush=True)
        
        # We leverage the .aio property of the modern Google GenAI Client wrapper
        response = await gemini_client.aio.models.generate_content(
            model='gemini-3.8-flash',  
            contents=f"Analyze this code diff payload:\n{real_diff}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        generated_review = response.text
        print(f"[Worker Success] Generated Review Output for {commit_sha}", flush=True)

        # 4. Write back to Redis Cache with a 24-hour TTL
        await redis_client.setex(cache_key, 86400, generated_review)
        
        # 5. Post comment back to GitHub Pull Request
        if GITHUB_TOKEN:
            repo_full_name = pr_data.get("repository", {}).get("full_name")
            pr_number = pr_data.get("number")
            
            print(f"[Worker Progress] Posting text blocks to GitHub repository...", flush=True)
            auth = Auth.Token(GITHUB_TOKEN)
            g = Github(auth=auth)
            
            repo = g.get_repo(repo_full_name)
            pull_request = repo.get_pull(pr_number)
            
            pull_request.create_issue_comment(generated_review)
            print(f"[GitHub Success] Posted review comment to PR #{pr_number}", flush=True)
        else:
            print("[Warning] GITHUB_TOKEN not found. Skipping comment insertion.", flush=True)

    except Exception as worker_error:
        # ◄ THE CRITICAL BACKEND WRAPPER: Catches any silent failures and prints them explicitly
        print(f"\n❌ [FATAL WORKER CRASH] An exception occurred inside the background worker thread!", flush=True)
        print(f"Details: {str(worker_error)}", flush=True)
        import traceback
        traceback.print_exc()

@app.post("/webhook/github", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(verify_github_signature)])
async def github_webhook_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Main ingestion endpoint for GitHub Webhook events.
    Parses the raw stream safely *after* the signature dependency verification executes.
    """
    # 1. Read the raw body text explicitly to ensure the stream is processed correctly
    body_bytes = await request.body()
    
    # 2. Parse the body bytes into JSON manually to avoid empty stream exceptions
    try:
        payload = json.loads(body_bytes.decode('utf-8'))
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload received")
        
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

import os, json, uuid, time, sqlite3, asyncio, re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_SECRET = os.getenv("APP_SECRET", "")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")
DB = os.getenv("DB_PATH", "control.db")
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]

app = FastAPI(title="Rahul12 AI Control Plane", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

clients: set[WebSocket] = set()
queue: list[dict[str, Any]] = []
state = {
    "online": False, "last_seen": None, "resource": "unknown",
    "gpu": {"count": 0, "devices": []}, "cpu": {}, "ram": {},
    "task": None, "task_id": None, "progress": 0,
    "stage": "idle", "stage_label": "Idle", "logs": [],
    "quota": {}, "version": "2.0.0", "workflow": []
}

STAGES = [
    ("inspect", "Inspect request"),
    ("plan", "Plan workflow"),
    ("route", "Route agent/model"),
    ("compute", "Prepare Kaggle compute"),
    ("execute", "Execute task"),
    ("verify", "Verify result"),
    ("complete", "Complete"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY, role TEXT, content TEXT, created_at TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, type TEXT, payload TEXT, created_at TEXT)")
    con.commit()
    return con


def save_event(kind: str, payload: Any):
    con = db()
    con.execute("INSERT INTO events VALUES (?,?,?,?)", (str(uuid.uuid4()), kind, json.dumps(payload, ensure_ascii=False), now()))
    con.commit(); con.close()


async def broadcast(event: dict[str, Any]):
    dead = []
    for ws in list(clients):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    model: Optional[str] = None
    focus: Optional[str] = Field(default=None, max_length=10000)


class TaskIn(BaseModel):
    type: str
    payload: dict = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)


class WorkerHeartbeat(BaseModel):
    resource: str = "unknown"
    gpu: dict = Field(default_factory=dict)
    cpu: dict = Field(default_factory=dict)
    ram: dict = Field(default_factory=dict)
    task: Optional[str] = None
    task_id: Optional[str] = None
    progress: float = Field(default=0, ge=0, le=100)
    stage: str = "idle"
    stage_label: str = "Idle"
    quota: dict = Field(default_factory=dict)
    workflow: list = Field(default_factory=list)
    version: str = "2.0.0"


class WorkerEvent(BaseModel):
    level: str = "info"
    message: str
    task_id: Optional[str] = None
    meta: dict = Field(default_factory=dict)


class CompleteIn(BaseModel):
    task_id: str
    ok: bool = True
    result: Any = None


def auth_worker(token: str):
    if not WORKER_TOKEN or token != WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid worker token")


def auth_app():
    if not APP_SECRET:
        # Development is allowed; production should always set APP_SECRET.
        return


async def call_openai_compatible(base_url: str, key: str, model: str, messages: list, timeout=75) -> str:
    if not key:
        raise RuntimeError("provider key not configured")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"invalid provider response: {e}")


async def ai_chat(messages: list, preferred: Optional[str] = None):
    providers = {
        "nvidia": (os.getenv("NVIDIA_KEY_1"), "https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")),
        "nvidia2": (os.getenv("NVIDIA_KEY_2"), "https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")),
        "glm": (os.getenv("GLM_KEY"), os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"), os.getenv("GLM_MODEL", "glm-4.5")),
        "openrouter": (os.getenv("OPENROUTER_KEY"), "https://openrouter.ai/api/v1", os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")),
        "groq": (os.getenv("GROQ_KEY"), "https://api.groq.com/openai/v1", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
        "deepseek": (os.getenv("DEEPSEEK_KEY_1"), "https://api.b.ai/v1", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
    }
    order = list(providers)
    if preferred in providers:
        order.remove(preferred); order.insert(0, preferred)
    errors = []
    for name in order:
        key, base, model = providers[name]
        if not key:
            continue
        try:
            return {"provider": name, "model": model, "answer": await call_openai_compatible(base, key, model, messages)}
        except Exception as e:
            errors.append(f"{name}:{type(e).__name__}")
    raise RuntimeError("All configured AI providers failed: " + ", ".join(errors))


async def tavily_search(query: str):
    key = os.getenv("TAVILY_KEY")
    if not key:
        return {"results": [], "warning": "Tavily not configured"}
    async with httpx.AsyncClient(timeout=40) as c:
        r = await c.post("https://api.tavily.com/search", json={
            "api_key": key, "query": query, "search_depth": "advanced",
            "include_answer": True, "include_raw_content": False, "max_results": 8
        })
        r.raise_for_status(); return r.json()


async def youtube_search(query: str):
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        return {"items": [], "warning": "YouTube API not configured"}
    base = os.getenv("YOUTUBE_BASE", "https://www.googleapis.com/youtube/v3")
    async with httpx.AsyncClient(timeout=40) as c:
        r = await c.get(base.rstrip("/") + "/search", params={
            "part": "snippet", "q": query, "type": "video", "maxResults": 8, "key": key
        })
        r.raise_for_status(); return r.json()


def route_intent(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["status", "worker", "gpu", "tpu", "cpu", "quota", "log"]): return "status"
    if any(x in t for x in ["news", "research", "search", "latest", "খবর", "রিসার্চ", "সার্চ"]): return "research"
    if any(x in t for x in ["dataset", "jsonl", "training data", "ডেটাসেট", "ট্রেনিং ডেটা"]): return "dataset"
    if any(x in t for x in ["train", "training", "ট্রেনিং"]): return "train"
    if any(x in t for x in ["github", "repo", "repository"]): return "sync_repo"
    return "chat"


def make_workflow(intent: str, task_id: Optional[str] = None):
    labels = [label for _, label in STAGES]
    return [{"id": sid, "label": label, "status": "pending"} for sid, label in STAGES]


async def queue_task(kind: str, payload: dict):
    task = {
        "id": str(uuid.uuid4()), "type": kind, "payload": payload,
        "priority": 5, "status": "queued", "created_at": now(),
        "workflow": make_workflow(kind)
    }
    queue.append(task)
    queue.sort(key=lambda x: (x["status"] != "queued", -x.get("priority", 5), x["created_at"]))
    save_event("task_queued", task)
    await broadcast({"type": "task", "data": task})
    return task


@app.get("/api/health")
async def health():
    return {"ok": True, "version": app.version, "time": now(), "worker_online": state["online"]}


@app.get("/api/status")
async def status():
    return {"worker": state, "queue": queue[-30:]}


@app.get("/api/logs")
async def logs(limit: int = 200):
    return {"logs": state["logs"][-max(1, min(limit, 1000)): ]}


@app.post("/api/chat")
async def chat(body: ChatIn):
    con = db(); con.execute("INSERT INTO messages VALUES (?,?,?,?)", (str(uuid.uuid4()), "user", body.message, now())); con.commit(); con.close()
    intent = route_intent(body.message)

    if intent == "status":
        answer = {
            "worker": state["online"], "resource": state["resource"], "gpu": state["gpu"],
            "cpu": state["cpu"], "ram": state["ram"], "task": state["task"],
            "progress": state["progress"], "stage": state["stage_label"], "quota": state["quota"]
        }
        result = {"provider": "control-plane", "answer": json.dumps(answer, ensure_ascii=False, indent=2), "intent": intent}
    elif intent == "research":
        tav, yt = await asyncio.gather(tavily_search(body.message), youtube_search(body.message))
        prompt = [
            {"role": "system", "content": "You are a source-grounded research agent. Use the supplied search results only. Separate facts from uncertainty. Return concise source titles and URLs when available. Do not invent links."},
            {"role": "user", "content": json.dumps({"query": body.message, "tavily": tav, "youtube": yt}, ensure_ascii=False)}
        ]
        result = await ai_chat(prompt, body.model); result["intent"] = intent
    elif intent in {"dataset", "train", "sync_repo"}:
        task = await queue_task(intent, {"request": body.message, "focus": body.focus or ""})
        result = {"provider": "orchestrator", "intent": intent, "task_id": task["id"], "answer": f"Workflow queued: {task['id']}\nStage: Inspect request\nWaiting for Kaggle worker."}
    else:
        system = """You are the Rahul12 control-plane assistant. Answer in the user's language. You can explain system state, plan workflows and use research sources supplied by the server. Never expose secrets. Never claim a server-side action succeeded unless the worker/backend returned evidence. When execution is required, create a task through the control plane rather than pretending."""
        msgs = [{"role": "system", "content": system}]
        if body.focus: msgs.append({"role": "system", "content": "Current focus:\n" + body.focus})
        msgs.append({"role": "user", "content": body.message})
        result = await ai_chat(msgs, body.model); result["intent"] = intent

    con = db(); con.execute("INSERT INTO messages VALUES (?,?,?,?)", (str(uuid.uuid4()), "assistant", result["answer"], now())); con.commit(); con.close()
    await broadcast({"type": "chat", "data": result})
    return result


@app.post("/api/tasks")
async def create_task(body: TaskIn):
    allowed = {"research", "dataset", "train", "status", "sync_repo"}
    if body.type not in allowed:
        raise HTTPException(400, "Unsupported task type")
    task = await queue_task(body.type, body.payload)
    task["priority"] = body.priority
    return task


@app.post("/api/worker/heartbeat")
async def heartbeat(body: WorkerHeartbeat, x_worker_token: str = Header(default="")):
    auth_worker(x_worker_token)
    state.update(body.model_dump())
    state["online"] = True; state["last_seen"] = now()
    await broadcast({"type": "status", "data": state})
    return {"ok": True}


@app.post("/api/worker/log")
async def worker_log(body: WorkerEvent, x_worker_token: str = Header(default="")):
    auth_worker(x_worker_token)
    item = {"time": now(), **body.model_dump()}
    state["logs"].append(item); state["logs"] = state["logs"][-1000:]
    save_event("worker_log", item)
    await broadcast({"type": "log", "data": item})
    return {"ok": True}


@app.get("/api/worker/poll")
async def worker_poll(x_worker_token: str = Header(default="")):
    auth_worker(x_worker_token)
    state["online"] = True; state["last_seen"] = now()
    for task in queue:
        if task["status"] == "queued":
            task["status"] = "running"; task["started_at"] = now()
            await broadcast({"type": "task", "data": task})
            return {"task": task}
    return {"task": None}


@app.post("/api/worker/complete")
async def worker_complete(body: CompleteIn, x_worker_token: str = Header(default="")):
    auth_worker(x_worker_token)
    for task in queue:
        if task["id"] == body.task_id:
            task["status"] = "completed" if body.ok else "failed"
            task["result"] = body.result; task["finished_at"] = now()
            state["stage"] = "complete" if body.ok else "error"
            state["stage_label"] = "Complete" if body.ok else "Failed"
            state["progress"] = 100 if body.ok else state["progress"]
            await broadcast({"type": "task", "data": task}); await broadcast({"type": "status", "data": state})
            return task
    raise HTTPException(404, "Task not found")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept(); clients.add(websocket)
    try:
        await websocket.send_json({"type": "status", "data": state})
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        clients.discard(websocket)


@app.on_event("startup")
async def startup():
    db()
    async def monitor():
        while True:
            await asyncio.sleep(20)
            if state["last_seen"]:
                try:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(state["last_seen"])).total_seconds()
                    if age > 60:
                        state["online"] = False; state["stage"] = "offline"; state["stage_label"] = "Worker offline"
                        await broadcast({"type": "status", "data": state})
                except Exception:
                    pass
    asyncio.create_task(monitor())

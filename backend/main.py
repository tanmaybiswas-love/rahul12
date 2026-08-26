import os, json, uuid, time, sqlite3, asyncio, re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import kaggle_manager
from fastapi.staticfiles import StaticFiles

APP_SECRET = os.getenv("APP_SECRET", "")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")
DB = os.getenv("DB_PATH", "control.db")
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
CONTROL_URL = os.getenv("CONTROL_URL", "").rstrip("/")
WEB_DIR = os.getenv("WEB_DIR", os.path.join(os.path.dirname(__file__), "..", "web"))

app = FastAPI(title="Rahul12 AI Control Plane", version="3.0.0")
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
    "quota": {}, "version": "3.0.0", "workflow": [],
    "stop_signal": False,
}

SPEC_WORKFLOW = [
    ("user", "User"),
    ("chat", "Chat"),
    ("planner", "AI Planner"),
    ("router", "Task Router"),
    ("resource", "Resource Router"),
    ("kaggle", "Kaggle"),
    ("compute", "CPU / GPU / TPU"),
    ("tools", "Tools"),
    ("result", "Result"),
]

# Map the worker's internal stages onto the public workflow graph.
STAGE_TO_NODE = {
    "inspect": "planner",
    "plan": "router",
    "route": "resource",
    "compute": "kaggle",
    "execute": "compute",
    "verify": "tools",
    "complete": "result",
    "idle": None,
    "offline": None,
    "error": "result",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


SECRET_PATTERNS = [
    re.compile(r"(ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]+"),
    re.compile(r"KGAT_[A-Za-z0-9]+"),
    re.compile(r"hf_[A-Za-z0-9]+"),
    re.compile(r"(sk|nvapi|glpat|kgl)[-_][A-Za-z0-9_\-]+"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.I),
]


def redact(text: str) -> str:
    out = text
    for pat in SECRET_PATTERNS:
        out = pat.sub("***", out)
    return out


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
    version: str = "3.0.0"


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
    if any(x in t for x in ["status", "worker", "gpu", "tpu", "cpu", "quota", "log", "কাজ", "চলছে"]): return "status"
    if any(x in t for x in ["news", "research", "search", "latest", "খবর", "রিসার্চ", "সার্চ"]): return "research"
    if any(x in t for x in ["dataset", "jsonl", "training data", "ডেটাসেট", "ট্রেনিং ডেটা"]): return "dataset"
    if any(x in t for x in ["train", "training", "ট্রেনিং"]): return "train"
    if any(x in t for x in ["github", "repo", "repository", "প্রজেক্ট"]): return "sync_repo"
    return "chat"


# ---------------------------------------------------------------- resource router
def route_resource(intent: str, payload: dict, request_text: str = "") -> dict:
    text = (request_text or "").lower()
    if any(x in text for x in ["tpu", "tensorflow", "jax", "flax", "tf record", "tpu-"]):
        return {"resource": "TPU", "reason": "TPU-compatible workload (TensorFlow/JAX/Flax)", "stage": "compute"}
    if intent == "train" or any(x in text for x in ["torch", "pytorch", "cuda", "llm", "model", "gpu", "fine-tune", "finetune", "inference"]):
        return {"resource": "GPU", "reason": "Training / CUDA / model inference workload", "stage": "compute"}
    if payload.get("compute") in ("gpu", "tpu"):
        return {"resource": payload["compute"].upper(), "reason": "Explicit compute request", "stage": "compute"}
    return {"resource": "CPU", "reason": "Lightweight API / preprocessing / script task", "stage": "compute"}


# ---------------------------------------------------------------- planner
def rule_plan(intent: str, request_text: str, rr: dict) -> list[dict]:
    steps = [
        {"id": "inspect", "label": "Inspect request", "tool": "planner"},
        {"id": "plan", "label": "Build task plan", "tool": "planner"},
        {"id": "route", "label": "Route intent", "tool": "router"},
        {"id": "resource", "label": f"Select {rr['resource']}", "tool": "resource-router"},
        {"id": "kaggle", "label": "Kaggle compute", "tool": "kaggle"},
        {"id": "execute", "label": f"Run on {rr['resource']}", "tool": "compute"},
        {"id": "verify", "label": "Verify output", "tool": "tools"},
        {"id": "complete", "label": "Return result", "tool": "chat"},
    ]
    return steps


async def plan_task(intent: str, request_text: str, rr: dict, preferred: Optional[str] = None) -> dict:
    steps = rule_plan(intent, request_text, rr)
    plan = {
        "steps": steps,
        "resource": rr["resource"],
        "reason": rr["reason"],
        "mode": "rule-based",
        "summary": f"Plan: {intent} using {rr['resource']}. {rr['reason']}",
    }
    prompt = [
        {"role": "system", "content": "You are an AI task planner for a control plane. Return a short JSON object with keys: summary (one line), tool (which tool), and steps (array of {id,label}). Plan for the given user request. Keep it factual."},
        {"role": "user", "content": json.dumps({"request": request_text, "intent": intent, "selected_resource": rr["resource"], "reason": rr["reason"]}, ensure_ascii=False)},
    ]
    try:
        res = await ai_chat(prompt, preferred)
        plan["mode"] = "ai"
        plan["answer"] = res["answer"]
        try:
            parsed = json.loads(re.sub(r"```json|```", "", res["answer"]).strip())
            if isinstance(parsed, dict):
                plan["summary"] = parsed.get("summary", plan["summary"])
                plan["tool"] = parsed.get("tool")
                if isinstance(parsed.get("steps"), list):
                    plan["steps"] = parsed["steps"]
        except Exception:
            pass
    except Exception as e:
        plan["mode"] = "rule-based"
        plan["warning"] = redact(str(e))
    return plan


def make_workflow(status: str = "pending", active: Optional[str] = None):
    return [{"id": nid, "label": label, "status": "active" if nid == active else status} for nid, label in SPEC_WORKFLOW]


def workflow_from_stage(stage: str) -> list[dict]:
    active = STAGE_TO_NODE.get(stage)
    order = [nid for nid, _ in SPEC_WORKFLOW]
    nodes = []
    for nid, label in SPEC_WORKFLOW:
        idx = order.index(nid)
        if active and nid == active:
            nodes.append({"id": nid, "label": label, "status": "active"})
        elif active and idx < order.index(active):
            nodes.append({"id": nid, "label": label, "status": "done"})
        elif stage == "error":
            nodes.append({"id": nid, "label": label, "status": "error" if nid in ("result", "kaggle") else "done"})
        else:
            nodes.append({"id": nid, "label": label, "status": "pending"})
    return nodes


async def queue_task(kind: str, payload: dict, request_text: str = ""):
    rr = route_resource(kind, payload, request_text)
    plan = await plan_task(kind, request_text, rr)
    task = {
        "id": str(uuid.uuid4()), "type": kind, "payload": payload,
        "priority": 5, "status": "queued", "created_at": now(),
        "workflow": make_workflow("pending"),
        "resource_route": rr,
        "plan": plan,
    }
    queue.append(task)
    queue.sort(key=lambda x: (x["status"] != "queued", -x.get("priority", 5), x["created_at"]))
    queue[:] = queue[-200:]
    save_event("task_queued", task)
    await broadcast({"type": "task", "data": task})
    return task


# ---------------------------------------------------------------- unified status
def unified_status() -> dict:
    kstatus = kaggle_manager.kernel_status()
    kquota = kaggle_manager.quota()
    return {
        "system": "online",
        "backend": "online",
        "kaggle": "connected" if kaggle_manager.is_configured() else "not-configured",
        "kernel": kstatus.get("status", "unknown"),
        "resource": state.get("resource", "unknown"),
        "gpu": state.get("gpu", {}).get("count", 0),
        "task": state.get("task"),
        "progress": state.get("progress", 0),
        "stage": state.get("stage", "idle"),
        "stage_label": state.get("stage_label", "Idle"),
        "logs": [redact(x.get("message", "")) for x in state.get("logs", [])[-50:]],
        "quota": kquota,
        "kernel_status": kstatus,
        "updated_at": now(),
    }


@app.get("/api/health")
async def health():
    return {"ok": True, "version": app.version, "time": now(), "worker_online": state["online"]}


@app.get("/api/status")
async def status():
    return {"worker": state, "queue": queue[-50:], "workflow": workflow_from_stage(state["stage"])}


@app.get("/api/logs")
async def logs(limit: int = 200):
    items = []
    for item in state["logs"][-max(1, min(limit, 1000)): ]:
        copy = dict(item); copy["message"] = redact(copy.get("message", "")); items.append(copy)
    return {"logs": items}


@app.post("/api/chat")
async def chat(body: ChatIn):
    con = db(); con.execute("INSERT INTO messages VALUES (?,?,?,?)", (str(uuid.uuid4()), "user", redact(body.message), now())); con.commit(); con.close()
    intent = route_intent(body.message)
    rr = route_resource(intent, {}, body.message)

    if intent == "status":
        answer = {
            "system": "online", "backend": "online",
            "worker": state["online"], "resource": state["resource"], "gpu": state["gpu"],
            "cpu": state["cpu"], "ram": state["ram"], "task": state["task"],
            "progress": state["progress"], "stage": state["stage_label"],
            "quota": kaggle_manager.quota(),
            "kernel": kaggle_manager.kernel_status(),
        }
        result = {"provider": "control-plane", "answer": json.dumps(answer, ensure_ascii=False, indent=2), "intent": intent, "resource_route": rr}
    elif intent == "research":
        tav, yt = await asyncio.gather(tavily_search(body.message), youtube_search(body.message))
        prompt = [
            {"role": "system", "content": "You are a source-grounded research agent. Use the supplied search results only. Separate facts from uncertainty. Return concise source titles and URLs when available. Do not invent links."},
            {"role": "user", "content": json.dumps({"query": body.message, "tavily": tav, "youtube": yt}, ensure_ascii=False)}
        ]
        try:
            result = await ai_chat(prompt, body.model); result["intent"] = intent; result["resource_route"] = rr
        except Exception as e:
            result = {"provider": "control-plane", "intent": intent,
                      "answer": f"Research sources fetched but no AI provider is configured to summarize. Raw results:\n{json.dumps({'tavily': tav, 'youtube': yt}, ensure_ascii=False, indent=2)[:4000]}",
                      "warning": redact(str(e)), "resource_route": rr}
    elif intent in {"dataset", "train", "sync_repo"}:
        task = await queue_task(intent, {"request": body.message, "focus": body.focus or ""}, body.message)
        result = {"provider": "orchestrator", "intent": intent, "task_id": task["id"],
                  "resource_route": task["resource_route"], "plan": task["plan"],
                  "answer": f"Workflow queued: {task['id'][:8]}\nResource: {task['resource_route']['resource']}\nReason: {task['resource_route']['reason']}\nPlan: {task['plan']['summary']}"}
    else:
        system = """You are the Rahul12 control-plane assistant. Answer in the user's language. You can explain system state, plan workflows and use research sources supplied by the server. Never expose secrets. Never claim a server-side action succeeded unless the worker/backend returned evidence. When execution is required, create a task through the control plane rather than pretending."""
        msgs = [{"role": "system", "content": system}]
        if body.focus: msgs.append({"role": "system", "content": "Current focus:\n" + body.focus})
        msgs.append({"role": "user", "content": body.message})
        try:
            result = await ai_chat(msgs, body.model); result["intent"] = intent; result["resource_route"] = rr
        except Exception as e:
            result = {"provider": "control-plane", "intent": intent,
                      "answer": "AI providers are not configured on this backend. Add server-side provider keys (NVIDIA/GLM/OpenRouter/Groq/DeepSeek) to the backend secret store to enable conversational AI. Status and Kaggle task routing remain available.",
                      "warning": redact(str(e)), "resource_route": rr}

    con = db(); con.execute("INSERT INTO messages VALUES (?,?,?,?)", (str(uuid.uuid4()), "assistant", redact(result["answer"]), now())); con.commit(); con.close()
    await broadcast({"type": "chat", "data": result})
    return result


# ---------------------------------------------------------------- tasks
@app.get("/api/tasks")
async def tasks(limit: int = 50):
    return {"tasks": queue[-max(1, min(limit, 200)):]}


@app.get("/api/tasks/{task_id}")
async def task_detail(task_id: str):
    for task in queue:
        if task["id"] == task_id:
            return task
    raise HTTPException(404, "Task not found")


@app.post("/api/tasks")
async def create_task(body: TaskIn):
    allowed = {"research", "dataset", "train", "status", "sync_repo"}
    if body.type not in allowed:
        raise HTTPException(400, "Unsupported task type")
    task = await queue_task(body.type, body.payload, json.dumps(body.payload, ensure_ascii=False))
    task["priority"] = body.priority
    return task


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    for task in queue:
        if task["id"] == task_id and task["status"] == "queued":
            task["status"] = "cancelled"; task["finished_at"] = now()
            await broadcast({"type": "task", "data": task})
            return task
    raise HTTPException(404, "Task not found or already running")


# ---------------------------------------------------------------- kaggle control
@app.get("/api/kaggle/status")
async def kaggle_status():
    return {
        "configured": kaggle_manager.is_configured(),
        "kernel": kaggle_manager.kernel_status(),
        "quota": kaggle_manager.quota(),
        "kernels": kaggle_manager.list_kernels(),
        "resource": state.get("resource"),
        "gpu": state.get("gpu"),
        "worker_online": state["online"],
    }


@app.get("/api/kaggle/logs")
async def kaggle_logs():
    kout = kaggle_manager.kernel_output()
    return {"kernel_output": redact(kout.get("logs", ""))[-12000:], "worker_logs": [redact(x.get("message", "")) for x in state.get("logs", [])[-100:]]}


@app.post("/api/kaggle/run")
async def kaggle_run():
    if not kaggle_manager.is_configured():
        raise HTTPException(400, "Kaggle API token not configured on the backend")
    if not CONTROL_URL:
        raise HTTPException(400, "CONTROL_URL not configured on the backend")
    state["stop_signal"] = False
    try:
        res = await asyncio.to_thread(kaggle_manager.push_worker, CONTROL_URL, WORKER_TOKEN)
    except kaggle_manager.KaggleError as e:
        raise HTTPException(500, str(e))
    save_event("kaggle_run", res)
    await broadcast({"type": "kaggle", "data": {"action": "run", **res}})
    return {"action": "run", **res}


@app.post("/api/kaggle/stop")
async def kaggle_stop():
    state["stop_signal"] = True
    state["stage"] = "stopping"; state["stage_label"] = "Stop requested"
    for task in queue:
        if task["status"] == "queued":
            task["status"] = "cancelled"; task["finished_at"] = now()
    save_event("kaggle_stop", {"time": now()})
    await broadcast({"type": "kaggle", "data": {"action": "stop"}}); await broadcast({"type": "status", "data": state})
    return {"action": "stop", "ok": True}


# ---------------------------------------------------------------- worker API
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
    body.message = redact(body.message)
    item = {"time": now(), **body.model_dump()}
    state["logs"].append(item); state["logs"] = state["logs"][-1000:]
    save_event("worker_log", item)
    await broadcast({"type": "log", "data": item})
    return {"ok": True}


@app.get("/api/worker/poll")
async def worker_poll(x_worker_token: str = Header(default="")):
    auth_worker(x_worker_token)
    state["online"] = True; state["last_seen"] = now()
    if state.get("stop_signal"):
        state["stop_signal"] = False
        return {"task": None, "stop": True}
    for task in queue:
        if task["status"] == "queued":
            task["status"] = "running"; task["started_at"] = now()
            await broadcast({"type": "task", "data": task})
            return {"task": task, "stop": False}
    return {"task": None, "stop": False}


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


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


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

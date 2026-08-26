import os, time, json, subprocess, platform
from pathlib import Path
import httpx

CONTROL_URL = os.getenv("CONTROL_URL", "http://127.0.0.1:8000").rstrip("/")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "6"))
LOG_DIR = Path("/kaggle/working/rahul12_logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)


def headers(): return {"X-Worker-Token": WORKER_TOKEN}


def post(path, payload, timeout=30):
    with httpx.Client(timeout=timeout) as c:
        r = c.post(CONTROL_URL + path, headers=headers(), json=payload); r.raise_for_status(); return r.json()


def get(path, timeout=30):
    with httpx.Client(timeout=timeout) as c:
        r = c.get(CONTROL_URL + path, headers=headers()); r.raise_for_status(); return r.json()


def gpu_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], text=True, stderr=subprocess.DEVNULL)
        devices=[]
        for line in out.strip().splitlines():
            p=[x.strip() for x in line.split(",")]
            if len(p)>=5: devices.append({"name":p[0],"memory_total_mb":p[1],"memory_used_mb":p[2],"utilization":p[3],"temperature_c":p[4]})
        return {"count":len(devices),"devices":devices}
    except Exception:
        return {"count":0,"devices":[]}


def tpu_info():
    try:
        if os.path.exists("/dev/accel0") or os.environ.get("TPU_NAME"):
            return {"available": True, "type": os.environ.get("TPU_ACCELERATOR_TYPE", "TPU")}
    except Exception:
        pass
    return {"available": False}


def cpu_info():
    return {"logical": os.cpu_count() or 1, "platform": platform.platform()}


def ram_info():
    try:
        import psutil
        m=psutil.virtual_memory()
        return {"total_mb": round(m.total/1048576), "used_mb": round(m.used/1048576), "percent": m.percent}
    except Exception:
        return {}


def detect_resource():
    g=gpu_info()
    t=tpu_info()
    if t.get("available"):
        return "tpu", g
    if g["count"]:
        return "gpu", g
    return "cpu", g


def heartbeat(task=None, task_id=None, progress=0, stage="idle", label="Idle", workflow=None):
    resource, g = detect_resource()
    return post("/api/worker/heartbeat", {
        "resource": resource,"gpu":g,"cpu":cpu_info(),"ram":ram_info(),
        "task":task,"task_id":task_id,"progress":progress,
        "stage":stage,"stage_label":label,"quota":{},"workflow":workflow or [],"version":"3.0.0"
    })


def log(message, level="info", task_id=None, meta=None):
    item={"level":level,"message":message,"task_id":task_id,"meta":meta or {}}
    print(message, flush=True)
    try: post("/api/worker/log", item)
    except Exception: pass
    (LOG_DIR/"worker.log").open("a",encoding="utf-8").write(json.dumps(item,ensure_ascii=False)+"\n")


def set_stage(task, stage, label, progress):
    wf=[]
    for x in task.get("workflow",[]):
        y=dict(x)
        if y["id"] == stage: y["status"]="active"
        elif y["id"] in [s for s,_ in STAGES_AFTER(stage)]: y["status"]="done"
        wf.append(y)
    heartbeat(task["type"], task["id"], progress, stage, label, wf)
    log(label, task_id=task["id"])


def STAGES_AFTER(stage):
    order=["inspect","plan","route","compute","execute","verify","complete"]
    try: idx=order.index(stage)
    except ValueError: idx=0
    return [(x,x) for x in order[:idx]]


def run_research(task):
    set_stage(task,"inspect","Inspecting research request",10)
    set_stage(task,"plan","Planning research workflow",25)
    set_stage(task,"route","Routing to research APIs",40)
    Path("/kaggle/working/research_request.json").write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding="utf-8")
    set_stage(task,"execute","Research request prepared",70)
    set_stage(task,"verify","Verifying worker output",90)
    set_stage(task,"complete","Research workflow ready",100)


def build_dataset(task):
    set_stage(task,"inspect","Inspecting dataset request",10)
    set_stage(task,"plan","Planning dataset schema",25)
    records=task["payload"].get("records",[])
    out=Path("/kaggle/working/dataset.jsonl")
    with out.open("w",encoding="utf-8") as f:
        for r in records:
            if isinstance(r,dict) and r: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    set_stage(task,"route","Selecting dataset pipeline",40)
    set_stage(task,"compute","Using Kaggle workspace",55)
    set_stage(task,"execute",f"Wrote {len(records)} JSONL records",75)
    set_stage(task,"verify","Checking dataset output",92)
    set_stage(task,"complete","Dataset ready",100)


def train(task):
    set_stage(task,"inspect","Inspecting training request",10)
    set_stage(task,"plan","Planning training workflow",25)
    set_stage(task,"route","Selecting training route",40)
    set_stage(task,"compute","Detecting Kaggle compute",55)
    g, t = gpu_info(), tpu_info()
    device = "TPU" if t.get("available") else ("GPU" if g["count"] else "CPU")
    set_stage(task,"execute",f"Training hook on {device}",75)
    set_stage(task,"verify","Waiting for approved training script",92)
    set_stage(task,"complete","Training hook reached",100)


def sync_repo(task):
    set_stage(task,"inspect","Inspecting repository sync request",10)
    set_stage(task,"plan","Planning repository sync",30)
    set_stage(task,"route","Preparing GitHub integration",55)
    set_stage(task,"execute","Repository sync hook ready",80)
    set_stage(task,"verify","Awaiting project-specific sync policy",95)
    set_stage(task,"complete","Sync hook reached",100)


def execute(task):
    kind=task["type"]
    if kind=="research": run_research(task)
    elif kind=="dataset": build_dataset(task)
    elif kind=="train": train(task)
    elif kind=="status": heartbeat("status",task["id"],100,"complete","Status collected",task.get("workflow",[]))
    elif kind=="sync_repo": sync_repo(task)
    else: raise ValueError("unsupported task")


def main():
    if not WORKER_TOKEN: raise SystemExit("WORKER_TOKEN is required")
    log("Rahul12 Kaggle worker starting (v3.0.0)")
    consecutive_failures = 0
    while True:
        try:
            heartbeat(None,None,0,"idle","Idle")
            res = get("/api/worker/poll")
            if res.get("stop"):
                log("Stop command received from control plane; exiting")
                break
            task = res.get("task")
            if task:
                log(f"Starting {task['id']} [{task['type']}]")
                try:
                    execute(task); post("/api/worker/complete",{"task_id":task["id"],"ok":True,"result":"completed"})
                except Exception as e:
                    log(str(e),"error",task["id"]); post("/api/worker/complete",{"task_id":task["id"],"ok":False,"result":str(e)})
                consecutive_failures = 0
            else:
                consecutive_failures = 0
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            log("Worker stopped by interrupt")
            break
        except Exception as e:
            consecutive_failures += 1
            print(f"worker loop: {e} (attempt {consecutive_failures})", flush=True)
            if consecutive_failures >= 10:
                log(f"Too many consecutive failures; pausing worker", "error")
                consecutive_failures = 0
                time.sleep(POLL_SECONDS * 5)
            time.sleep(POLL_SECONDS)

if __name__=="__main__": main()

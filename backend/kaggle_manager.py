import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

KAGGLE_TOKEN = os.getenv("KAGGLE_API_TOKEN", "")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_ACCELERATOR = os.getenv("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")
WORKER_KERNEL = os.getenv("KAGGLE_WORKER_KERNEL", "rahulglmmodel/rahul12-worker-t4")
WORKER_KERNEL_ID = int(os.getenv("KAGGLE_WORKER_KERNEL_ID", "0") or 0)

WORKER_DIR = Path(os.getenv("KAGGLE_WORKER_DIR", os.path.join(os.path.dirname(__file__), "..", "kaggle_worker")))


class KaggleError(RuntimeError):
    pass


def _base_env() -> dict:
    env = dict(os.environ)
    env.pop("KAGGLE_CONFIG_DIR", None)
    if KAGGLE_TOKEN:
        env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    return env


def _run_cli(args: list, timeout: int = 90) -> tuple[int, str]:
    proc = subprocess.run(
        ["kaggle", *args],
        capture_output=True,
        text=True,
        env=_base_env(),
        timeout=timeout,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = "\n".join(x for x in (out, err) if x)
    if proc.returncode != 0 and "Authentication required" in err:
        raise KaggleError("Kaggle API token not configured on the backend")
    return proc.returncode, combined


def is_configured() -> bool:
    return bool(KAGGLE_TOKEN)


def quota() -> dict:
    if not is_configured():
        return {"configured": False, "resources": []}
    code, out = _run_cli(["quota"])
    resources = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] in ("GPU", "CPU", "TPU"):
            resources.append({"resource": parts[0], "used": parts[1], "remaining": parts[2], "total": parts[3], "refreshAt": parts[4]})
    return {"configured": True, "resources": resources, "raw": out[:2000]}


def kernel_status() -> dict:
    if not is_configured():
        return {"slug": WORKER_KERNEL, "status": "unknown", "configured": False}
    code, out = _run_cli(["kernels", "status", WORKER_KERNEL])
    status = out.replace("has status ", "").strip().strip('"')
    if status.startswith(WORKER_KERNEL):
        status = out.split("has status")[-1].strip().strip('"')
    return {"slug": WORKER_KERNEL, "status": status or "unknown", "configured": True, "ok": code == 0}


def list_kernels() -> list:
    if not is_configured():
        return []
    code, out = _run_cli(["kernels", "list", "--mine", "--page-size", "20"])
    rows = []
    for line in out.splitlines():
        if "ref" in line or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            rows.append({"ref": parts[0], "title": parts[1], "lastRun": parts[2] if len(parts) > 2 else ""})
    return rows


def build_worker_script(control_url: str, worker_token: str, worker_source: str) -> str:
    b64 = base64.b64encode(worker_source.encode("utf-8")).decode()
    lines = [
        "import os, subprocess, sys, base64",
        "",
        f'os.environ["CONTROL_URL"] = "{control_url}"',
        f'os.environ["WORKER_TOKEN"] = "{worker_token}"',
        "",
        'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx", "psutil"])',
        "",
        f'WORKER_B64 = "{b64}"',
        "",
        'worker_path = "/kaggle/working/worker.py"',
        'with open(worker_path, "wb") as f:',
        "    f.write(base64.b64decode(WORKER_B64))",
        "",
        "import runpy",
        'runpy.run_path(worker_path, run_name="__main__")',
    ]
    return "\n".join(lines) + "\n"


def push_worker(control_url: str, worker_token: str, accelerator: str | None = None) -> dict:
    if not is_configured():
        raise KaggleError("Kaggle API token not configured on the backend")
    acc = accelerator or KAGGLE_ACCELERATOR
    worker_source = (WORKER_DIR / "worker.py").read_text(encoding="utf-8")
    script = build_worker_script(control_url, worker_token, worker_source)

    with tempfile.TemporaryDirectory(prefix="kaggle-worker-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "worker-kernel.py").write_text(script, encoding="utf-8")
        meta = {
            "id": WORKER_KERNEL,
            "title": "rahul12-worker-t4",
            "code_file": "worker-kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "enable_tpu": "false",
            "machine_shape": acc,
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        if WORKER_KERNEL_ID:
            meta["id_no"] = WORKER_KERNEL_ID
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        code, out = _run_cli(["kernels", "push", "-p", str(tmp_path)])
        if code != 0:
            raise KaggleError(f"Kaggle push failed: {out[:800]}")
        return {"pushed": True, "message": out[:500], "slug": WORKER_KERNEL}


def kernel_output() -> dict:
    if not is_configured():
        return {"logs": "", "configured": False}
    code, out = _run_cli(["kernels", "output", WORKER_KERNEL])
    return {"logs": out[-12000:], "ok": code == 0}

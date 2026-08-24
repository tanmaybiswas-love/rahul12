import os
from fastapi.staticfiles import StaticFiles
from main import app

WEB_DIR = os.getenv("WEB_DIR", os.path.join(os.path.dirname(__file__), "..", "web"))
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("CONTROL_PORT", "8000")))

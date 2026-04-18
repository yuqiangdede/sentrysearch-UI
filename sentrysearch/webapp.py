from __future__ import annotations
import logging
import hashlib
import os, threading, traceback, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from jinja2 import Environment
from pydantic import BaseModel, Field
from .paths import CLIPS_DIR, UPLOADS_DIR
from .services import (
    clear_video_index,
    get_index_rebuild_candidates,
    get_stats,
    get_video_index_status,
    run_index,
    run_search,
    run_trim,
)
from .reranker import RerankerError
from .trimmer import create_browser_preview_clip

UPLOAD_DIR = UPLOADS_DIR
ALLOWED_UPLOAD_EXTS = {".mp4", ".mov"}
VIDEO_CONTENT_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime"}
now = lambda: datetime.now(timezone.utc).isoformat()
logger = logging.getLogger("sentrysearch.web")
runtime_logger = logging.getLogger("uvicorn.error")


def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def list_uploads() -> list[dict]:
    base = ensure_upload_dir()
    out = []
    for p in sorted(base.glob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_UPLOAD_EXTS:
            out.append({"name": p.name, "path": str(p.resolve()), "size_bytes": p.stat().st_size})
    return out


class JobManager:
    def __init__(self, workers: int = 2):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sentrysearch-web")
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.files: dict[str, str] = {}

    def submit(self, job_type: str, runner: Callable[[Callable[[dict], None]], dict]) -> str:
        jid = str(uuid.uuid4())
        with self.lock:
            self.jobs[jid] = {"job_id": jid, "job_type": job_type, "status": "queued", "created_at": now(), "updated_at": now(), "progress": {}, "result": None, "error": None}
        runtime_logger.info("job queued: id=%s type=%s", jid, job_type)

        def cb(payload: dict):
            with self.lock:
                j = self.jobs.get(jid)
                if not j:
                    return
                j["progress"] = {**j["progress"], **payload}
                j["updated_at"] = now()

        def run():
            with self.lock:
                self.jobs[jid]["status"] = "running"
                self.jobs[jid]["updated_at"] = now()
            runtime_logger.info("job started: id=%s type=%s", jid, job_type)
            try:
                res = runner(cb)
                with self.lock:
                    self.jobs[jid]["status"] = "succeeded"
                    self.jobs[jid]["result"] = res
                    self.jobs[jid]["updated_at"] = now()
                runtime_logger.info("job succeeded: id=%s type=%s", jid, job_type)
            except Exception as e:  # pragma: no cover
                with self.lock:
                    self.jobs[jid]["status"] = "failed"
                    self.jobs[jid]["error"] = f"{e}\n{traceback.format_exc()}"
                    self.jobs[jid]["updated_at"] = now()
                runtime_logger.exception("job failed: id=%s type=%s", jid, job_type)

        self.pool.submit(run)
        return jid

    def get(self, jid: str) -> dict[str, Any] | None:
        with self.lock:
            j = self.jobs.get(jid)
            return None if j is None else {**j, "progress": dict(j["progress"])}

    def reg_file(self, p: str) -> str:
        t = str(uuid.uuid4())
        with self.lock:
            self.files[t] = p
        return t

    def file(self, token: str) -> str | None:
        with self.lock:
            return self.files.get(token)


class IndexRequest(BaseModel):
    directory: str
    chunk_duration: int = Field(default=5, ge=1)
    overlap: int = Field(default=1, ge=0)
    preprocess: bool = True
    target_resolution: int = Field(default=480, ge=120)
    target_fps: int = Field(default=5, ge=1)
    skip_still: bool = False
    backend: str | None = Field(default=None, pattern="^(gemini|local)?$")
    model: str | None = None
    quantize: bool | None = None
    force_reindex: bool = False
    verbose: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    results: int = Field(default=5, ge=1, le=50)
    recall: int | None = Field(default=None, ge=1)
    threshold: float = Field(default=0.41, ge=0.0, le=1.0)
    backend: str | None = Field(default=None, pattern="^(gemini|local)?$")
    model: str | None = None
    quantize: bool | None = None
    rerank: bool = False
    verbose: bool = False


class SearchResultItem(BaseModel):
    source_file: str
    start_time: float
    end_time: float
    similarity_score: float
    vector_score: float | None = None
    preview_url: str | None = None


class TrimRequest(BaseModel):
    results: list[SearchResultItem]
    selected_indices: list[int]
    output_dir: str = str(CLIPS_DIR)
    overlay: bool = False


class IndexStatusRequest(BaseModel):
    source_files: list[str]
    backend: str | None = Field(default=None, pattern="^(gemini|local)?$")
    model: str | None = None


class ClearVideoIndexRequest(BaseModel):
    source_file: str = Field(min_length=1)
    backend: str | None = Field(default=None, pattern="^(gemini|local)?$")
    model: str | None = None


def _browser_preview_cache_path(source_file: str, start: float | None, end: float | None) -> Path:
    """Return a stable cache path for browser preview clips."""
    preview_dir = Path(__file__).resolve().parents[1] / ".sentrysearch" / "tmp" / "browser-previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    source = Path(source_file)
    try:
        mtime_ns = source.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0

    key = f"{source.resolve()}|{mtime_ns}|{start!r}|{end!r}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return preview_dir / f"{digest}.mp4"


def create_app() -> FastAPI:
    app = FastAPI(title="SentrySearch Web", version="0.1.0")
    jobs = JobManager()
    page = Environment(autoescape=True).from_string(TEMPLATE)

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return page.render(page_mode="index")

    @app.get("/search", response_class=HTMLResponse)
    def search_page() -> str:
        return page.render(page_mode="search")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/stats")
    def api_stats() -> JSONResponse:
        return JSONResponse(get_stats())

    @app.get("/api/uploads")
    def api_uploads() -> JSONResponse:
        return JSONResponse({"upload_dir": str(ensure_upload_dir().resolve()), "videos": list_uploads()})

    @app.post("/api/upload-video")
    async def api_upload(file: UploadFile = File(...)) -> JSONResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        if Path(file.filename).suffix.lower() not in ALLOWED_UPLOAD_EXTS:
            raise HTTPException(status_code=400, detail="Only .mp4/.mov are supported")
        p = ensure_upload_dir() / f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
        p.write_bytes(await file.read())
        return JSONResponse({"name": p.name, "path": str(p.resolve()), "size_bytes": p.stat().st_size})

    @app.post("/api/index")
    def api_index(payload: IndexRequest) -> JSONResponse:
        d = os.path.expanduser(payload.directory)
        if not os.path.exists(d):
            raise HTTPException(status_code=400, detail=f"Directory not found: {d}")
        pv = get_index_rebuild_candidates(directory=d, backend=payload.backend, model=payload.model)
        is_single_file_target = os.path.isfile(d)
        is_all_indexed = pv["total_videos"] > 0 and pv["indexed_count"] == pv["total_videos"]
        should_conflict = (is_single_file_target and pv["indexed_count"] > 0) or (
            not is_single_file_target and is_all_indexed
        )
        if should_conflict and not payload.force_reindex:
            return JSONResponse(status_code=409, content={"code": "indexed_exists", "indexed_count": pv["indexed_count"], "total_videos": pv["total_videos"]})

        def runner(cb):
            runtime_logger.info("index run begin: target=%s backend=%s model=%s", d, pv["backend"], pv["model"])
            return run_index(
                directory=d, chunk_duration=payload.chunk_duration, overlap=payload.overlap,
                preprocess=payload.preprocess, target_resolution=payload.target_resolution, target_fps=payload.target_fps,
                skip_still=payload.skip_still, backend=pv["backend"], model=pv["model"], quantize=payload.quantize,
                force_reindex=payload.force_reindex, verbose=payload.verbose, progress_callback=cb
            )
        return JSONResponse({"job_id": jobs.submit("index", runner), "status": "queued"})

    @app.post("/api/index-status")
    def api_index_status(payload: IndexStatusRequest) -> JSONResponse:
        if not payload.source_files:
            return JSONResponse({"backend": payload.backend, "model": payload.model, "status": {}})
        return JSONResponse(
            get_video_index_status(
                source_files=payload.source_files,
                backend=payload.backend,
                model=payload.model,
            )
        )

    @app.post("/api/index/clear")
    def api_clear_video_index(payload: ClearVideoIndexRequest) -> JSONResponse:
        return JSONResponse(
            clear_video_index(
                source_file=payload.source_file,
                backend=payload.backend,
                model=payload.model,
            )
        )

    @app.post("/api/search")
    def api_search(payload: SearchRequest) -> JSONResponse:
        try:
            res = run_search(
                query=payload.query,
                n_results=payload.results,
                recall=payload.recall,
                threshold=payload.threshold,
                backend=payload.backend,
                model=payload.model,
                quantize=payload.quantize,
                rerank=payload.rerank,
                verbose=payload.verbose,
            )
        except RerankerError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        out = []
        for x in res["results"]:
            t = jobs.reg_file(x["source_file"])
            out.append({**x, "preview_url": f"/api/media/{t}"})
        return JSONResponse({**res, "results": out})

    @app.post("/api/trim")
    def api_trim(payload: TrimRequest) -> JSONResponse:
        serializable = [x.model_dump() for x in payload.results]
        def runner(cb):
            r = run_trim(results=serializable, selected_indices=payload.selected_indices, output_dir=payload.output_dir, overlay=payload.overlay, progress_callback=cb)
            clips = []
            for p in r["clips"]:
                t = jobs.reg_file(p)
                clips.append({"path": p, "download_url": f"/api/files/{t}", "preview_url": f"/api/media/{t}"})
            return {**r, "clips": clips}
        return JSONResponse({"job_id": jobs.submit("trim", runner), "status": "queued"})

    @app.get("/api/jobs/{jid}")
    def api_job(jid: str) -> JSONResponse:
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JSONResponse(j)

    @app.get("/api/files/{token}")
    def api_file(token: str) -> FileResponse:
        p = jobs.file(token)
        if p is None or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(path=p, filename=os.path.basename(p))

    @app.get("/api/media/{token}")
    def api_media(token: str, start: float | None = None, end: float | None = None) -> FileResponse:
        p = jobs.file(token)
        if p is None or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="file not found")
        if start is not None and end is not None and end > start:
            preview_path = _browser_preview_cache_path(p, start, end)
            if not preview_path.is_file() or preview_path.stat().st_size <= 1024:
                create_browser_preview_clip(
                    source_file=p,
                    start_time=start,
                    end_time=end,
                    output_path=str(preview_path),
                )
            return FileResponse(
                path=str(preview_path),
                media_type=VIDEO_CONTENT_TYPES.get(preview_path.suffix.lower(), "video/mp4"),
            )
        return FileResponse(path=p, media_type=VIDEO_CONTENT_TYPES.get(Path(p).suffix.lower(), "application/octet-stream"))

    return app


def run_web_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    class _HideIndexStatusAccessLog(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "/api/index-status" not in record.getMessage()

    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(x, _HideIndexStatusAccessLog) for x in access_logger.filters):
        access_logger.addFilter(_HideIndexStatusAccessLog())

    uvicorn.run(
        "sentrysearch.webapp:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        access_log=True,
    )


TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SentrySearch</title>
<style>
:root{
  --bg:#000000;
  --bg-soft:#090909;
  --surface:rgba(19,19,19,.94);
  --surface-2:rgba(10,10,10,.9);
  --line:rgba(250,255,105,.18);
  --line-soft:rgba(255,255,255,.08);
  --text:#f5f5f0;
  --muted:#a1a1aa;
  --muted-2:#6f6f73;
  --accent:#faff69;
  --accent-soft:#f4f692;
  --accent-deep:#e6e84c;
  --green:#166534;
  --green-deep:#14572f;
  --success:#3ee08f;
  --danger:#ff6b6b;
  --shadow:0 20px 70px rgba(0,0,0,.52);
  --radius:18px;
  --radius-sm:12px;
  --font-sans:Inter,"Segoe UI",Arial,Helvetica,sans-serif;
  --font-mono:Inconsolata,"JetBrains Mono","SFMono-Regular",Consolas,monospace;
}
*{box-sizing:border-box}
html,body{min-height:100%}
body{
  margin:0;
  color:var(--text);
  font-family:var(--font-sans);
  background:
    radial-gradient(circle at 10% 0%, rgba(250,255,105,.10), transparent 26%),
    radial-gradient(circle at 92% 8%, rgba(22,101,52,.20), transparent 22%),
    linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px),
    linear-gradient(180deg, #000 0%, #050505 100%);
  background-size:auto,auto,46px 46px,46px 46px,auto;
  background-position:center center;
  overflow:hidden;
}
main{
  flex:1 1 auto;
  min-height:0;
}
body::before{
  content:"";
  position:fixed;
  inset:-20vh -20vw auto -20vw;
  height:48vh;
  background:radial-gradient(circle, rgba(250,255,105,.10), transparent 58%);
  filter:blur(24px);
  pointer-events:none;
  opacity:.8;
}
body::after{
  content:"";
  position:fixed;
  inset:auto -10vw -18vh auto;
  width:36vw;
  height:36vw;
  background:radial-gradient(circle, rgba(22,101,52,.18), transparent 62%);
  filter:blur(34px);
  pointer-events:none;
  opacity:.7;
}
a{color:var(--accent);text-decoration:none}
a:hover{color:var(--accent-soft)}
button,input,select{font:inherit}
.app{
  width:min(1600px,calc(100vw - 16px));
  margin:0 auto;
  height:100vh;
  padding:10px 0 10px;
  position:relative;
  z-index:1;
  display:flex;
  flex-direction:column;
  min-height:0;
  animation:pageIn .55s ease-out both;
}
.masthead{
  display:flex;
  justify-content:space-between;
  align-items:flex-end;
  gap:12px;
  padding:10px 2px 12px;
  border-bottom:1px solid var(--line-soft);
}
.brand{
  min-width:0;
}
.eyebrow{
  margin:0 0 6px;
  font-size:11px;
  letter-spacing:1.8px;
  text-transform:uppercase;
  color:var(--accent);
}
.masthead h1{
  margin:0;
  font-size:clamp(1.35rem,2vw,2.2rem);
  line-height:1.02;
  letter-spacing:-.04em;
  font-weight:900;
}
.actions{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  justify-content:flex-end;
}
.btn.secondary.active{
  background:linear-gradient(180deg, rgba(250,255,105,.16), rgba(250,255,105,.08));
  border-color:rgba(250,255,105,.42);
  color:var(--accent);
}
.workspace{
  display:grid;
  grid-template-columns:minmax(0,1fr) minmax(0,1fr);
  gap:12px;
  align-items:start;
  height:100%;
  min-height:0;
  margin-top:12px;
}
.stack{
  display:grid;
  gap:12px;
  min-height:0;
}
.stack--right{
  position:sticky;
  top:12px;
}
.panel{
  position:relative;
  overflow:hidden;
  border:1px solid var(--line);
  border-radius:16px;
  background:linear-gradient(180deg, rgba(19,19,19,.96), rgba(8,8,8,.94));
  box-shadow:var(--shadow);
  min-height:0;
  display:flex;
  flex-direction:column;
}
.panel::before{
  content:"";
  position:absolute;
  inset:0;
  border-radius:inherit;
  pointer-events:none;
  background:linear-gradient(180deg, rgba(255,255,255,.05), transparent 22%);
  opacity:.7;
}
.panel-header{
  display:flex;
  align-items:flex-start;
  gap:10px;
  padding:12px 14px 0;
}
.step-badge{
  flex:0 0 auto;
  width:30px;
  height:30px;
  border-radius:999px;
  display:grid;
  place-items:center;
  color:#000;
  background:var(--accent);
  font-weight:900;
  box-shadow:0 0 0 4px rgba(250,255,105,.12);
}
.panel-title{
  min-width:0;
  flex:1;
}
.panel-title h2{
  margin:0;
  font-size:15px;
  letter-spacing:.02em;
}
.panel-title p{
  margin:5px 0 0;
  color:var(--muted);
  line-height:1.5;
  font-size:12px;
}
.toggle{
  border:1px solid var(--line-soft);
  background:rgba(255,255,255,.03);
  color:var(--text);
  border-radius:999px;
  padding:6px 10px;
  cursor:pointer;
  transition:transform .18s ease,border-color .18s ease,background .18s ease;
}
.toggle:hover{transform:translateY(-1px);border-color:var(--line);background:rgba(255,255,255,.05)}
.panel-body{
  padding:12px 14px 14px;
  flex:1 1 auto;
  min-height:0;
}
.panel.collapsed .panel-body{
  display:none;
}
.panel.collapsed .panel-header{
  padding-bottom:12px;
}
.field{
  display:grid;
  gap:6px;
  margin-bottom:8px;
}
.field-grid{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:8px;
}
.search-config-grid{
  grid-template-columns:repeat(3,minmax(0,1fr));
}
.field label{
  color:var(--muted);
  font-size:11px;
  letter-spacing:.02em;
}
.row{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  align-items:center;
}
input[type=text],input[type=number],select{
  width:100%;
  min-height:36px;
  padding:8px 10px;
  color:var(--text);
  background:rgba(6,6,6,.96);
  border:1px solid var(--line-soft);
  border-radius:12px;
  outline:none;
  transition:border-color .18s ease,box-shadow .18s ease,transform .18s ease;
}
input[type=text]:focus,input[type=number]:focus,select:focus,input[type=file]:focus{
  border-color:rgba(250,255,105,.65);
  box-shadow:0 0 0 3px rgba(250,255,105,.10);
}
input[type=file]{
  width:100%;
  padding:8px 10px;
  border:1px dashed rgba(250,255,105,.28);
  border-radius:12px;
  background:rgba(6,6,6,.65);
  color:var(--muted);
}
button.btn,
a.btnlink{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  gap:8px;
  min-height:36px;
  padding:8px 12px;
  border-radius:12px;
  border:1px solid transparent;
  cursor:pointer;
  transition:transform .18s ease,background .18s ease,border-color .18s ease,color .18s ease,box-shadow .18s ease;
  text-decoration:none;
  white-space:nowrap;
}
.btn{
  background:linear-gradient(180deg, var(--accent), #dfe24b);
  color:#000;
  font-weight:800;
}
.btn:hover{transform:translateY(-1px);box-shadow:0 12px 26px rgba(250,255,105,.18)}
.btn.secondary{
  background:rgba(255,255,255,.03);
  color:var(--text);
  border-color:var(--line-soft);
}
.btn.secondary:hover{
  border-color:rgba(250,255,105,.35);
  background:rgba(255,255,255,.05);
}
.btn.success{
  background:linear-gradient(180deg, var(--green), var(--green-deep));
  color:#eafbea;
}
.btn:disabled{
  opacity:.55;
  cursor:not-allowed;
  transform:none;
  box-shadow:none;
}
.status{
  margin-top:8px;
  padding:9px 10px;
  border:1px solid var(--line-soft);
  border-radius:12px;
  background:rgba(6,6,6,.65);
  color:var(--muted);
  font-family:var(--font-mono);
  font-size:11px;
  line-height:1.45;
  white-space:pre-wrap;
  word-break:break-word;
  min-height:40px;
}
.status.error{
  border-color:rgba(255,107,107,.35);
  color:#ffb4b4;
}
.tool-grid{
  display:grid;
  gap:12px;
}
.tool-grid .row label{
  flex:0 0 auto;
}
.tool-grid .row input[type=text],
.tool-grid .row input[type=number],
.tool-grid .row select{
  flex:1 1 190px;
  min-width:160px;
}
.choices{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
}
.choices label{
  display:inline-flex;
  align-items:center;
  gap:8px;
  color:var(--text);
  font-size:12px;
}
.upload-list{
  margin-top:8px;
  max-height:calc(100vh - 430px);
  overflow:auto;
  border-top:1px solid var(--line-soft);
  padding-top:8px;
}
.item{
  display:flex;
  align-items:center;
  gap:8px;
  padding:8px 10px;
  margin-bottom:10px;
  border:1px solid rgba(255,255,255,.06);
  border-radius:12px;
  background:rgba(255,255,255,.02);
  transition:border-color .18s ease,transform .18s ease,background .18s ease;
}
.item:hover{
  border-color:rgba(250,255,105,.25);
  background:rgba(255,255,255,.04);
  transform:translateY(-1px);
}
.item input{
  margin:0;
}
.item code,
code{
  padding:2px 6px;
  border-radius:8px;
  background:rgba(255,255,255,.05);
  color:var(--accent-soft);
  font-family:var(--font-mono);
  font-size:11px;
  word-break:break-all;
}
.item .meta{
  color:var(--muted);
  font-family:var(--font-mono);
  font-size:11px;
}
.badge-pill{
  display:inline-flex;
  align-items:center;
  padding:4px 8px;
  border-radius:999px;
  border:1px solid rgba(250,255,105,.2);
  color:var(--accent);
  font-size:11px;
  letter-spacing:.08em;
  text-transform:uppercase;
}
.preview-panel .panel-body{
  padding-top:0;
}
.search-results-panel{
  margin-top:12px;
}
.preview{
  margin:0 14px 14px;
  border:1px solid rgba(255,255,255,.08);
  border-radius:16px;
  overflow:hidden;
  background:#050505;
}
.preview video{
  display:block;
  width:100%;
  aspect-ratio:16/8.25;
  background:#000;
}
.preview-meta{
  padding:9px 12px;
  border-top:1px solid rgba(255,255,255,.08);
  background:linear-gradient(180deg, rgba(14,14,14,.96), rgba(8,8,8,.96));
  color:var(--muted);
  font-size:11px;
  line-height:1.45;
}
.detail-grid{
  display:grid;
  gap:12px;
}
.detail-box{
  border:1px solid var(--line-soft);
  border-radius:14px;
  background:rgba(6,6,6,.7);
  padding:10px 12px;
}
.detail-box h3{
  margin:0 0 8px;
  font-size:13px;
  color:var(--accent);
  letter-spacing:.1em;
  text-transform:uppercase;
}
.detail-box .list{
  max-height:120px;
  overflow:auto;
  padding-right:4px;
  color:var(--muted);
  font-family:var(--font-mono);
  font-size:11px;
  line-height:1.45;
  white-space:pre-wrap;
  word-break:break-word;
}
.table-shell{
  overflow:auto;
  border:1px solid var(--line-soft);
  border-radius:14px;
  background:rgba(4,4,4,.8);
  max-height:220px;
}
table{
  width:100%;
  border-collapse:collapse;
  font-size:11px;
}
thead th{
  position:sticky;
  top:0;
  z-index:1;
  background:rgba(15,15,15,.98);
  color:var(--accent);
  text-align:left;
  border-bottom:1px solid rgba(250,255,105,.15);
  padding:9px 8px;
  text-transform:uppercase;
  letter-spacing:.08em;
  font-size:11px;
}
tbody td{
  padding:8px 8px;
  border-bottom:1px solid rgba(255,255,255,.06);
  color:var(--text);
  vertical-align:middle;
}
tbody tr:hover{
  background:rgba(250,255,105,.03);
}
.results-tools{
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  margin-bottom:12px;
}
.mini{
  min-height:28px;
  padding:5px 8px;
  border-radius:10px;
  border:1px solid var(--line-soft);
  background:rgba(255,255,255,.03);
  color:var(--text);
  cursor:pointer;
  transition:transform .18s ease,border-color .18s ease,background .18s ease;
}
.mini:hover{
  transform:translateY(-1px);
  border-color:rgba(250,255,105,.28);
  background:rgba(255,255,255,.06);
}
.download-links{
  display:grid;
  gap:8px;
  margin-top:8px;
}
.download-links > div{
  display:flex;
  align-items:center;
  gap:8px;
  flex-wrap:wrap;
  padding:8px 10px;
  border:1px solid rgba(255,255,255,.06);
  border-radius:12px;
  background:rgba(255,255,255,.02);
}
.download-links a{
  word-break:break-all;
}
.subtle{
  color:var(--muted-2);
  font-size:12px;
}
.collapsed .panel-body,
.collapsed .preview{
  display:none;
}
.collapsed .panel-header{
  padding-bottom:18px;
}
@keyframes pageIn{
  from{opacity:0;transform:translateY(12px)}
  to{opacity:1;transform:translateY(0)}
}
@media (max-width:1180px){
  .workspace{
    grid-template-columns:1fr;
  }
  .stack--right{
    position:static;
  }
}
@media (max-width:760px){
  .app{width:min(100vw - 16px,1600px);padding-top:10px;height:100dvh}
  .masthead{
    flex-direction:column;
    align-items:flex-start;
  }
  .actions{
    justify-content:flex-start;
  }
  .panel-header,
  .panel-body{padding-left:12px;padding-right:12px}
  .preview{margin-left:12px;margin-right:12px}
  .field-grid{
    grid-template-columns:1fr;
  }
  .search-config-grid{
    grid-template-columns:1fr;
  }
  .tool-grid .row{
    align-items:stretch;
  }
  .tool-grid .row label{
    width:100%;
  }
}
</style>
</head>
<body>
<div class="app">
  <header class="masthead">
    <div class="brand">
      {% if page_mode == "search" %}
      <div class="eyebrow">SentrySearch / Video Intelligence</div>
      <h1>搜索</h1>
      {% else %}
      <div class="eyebrow">SentrySearch / Video Intelligence</div>
      <h1>索引</h1>
      {% endif %}
    </div>
    <div class="actions">
      <a class="btn secondary{% if page_mode == 'index' %} active{% endif %}" href="/">索引</a>
      <a class="btn secondary{% if page_mode == 'search' %} active{% endif %}" href="/search">搜索</a>
    </div>
  </header>

  <main>
    {% if page_mode == "index" %}
    <section class="page-panel page-panel--index">
      <div class="workspace">
        <div class="stack stack--left">
          <section class="panel" id="step1">
            <div class="panel-header">
              <div class="step-badge">1</div>
              <div class="panel-title">
                <h2>上传视频</h2>
                <p>把源视频放进本地上传目录，或直接把选中的视频作为索引目标。</p>
              </div>
              <button class="toggle" data-step="step1">收起</button>
            </div>
            <div class="panel-body tool-grid">
              <div class="field">
                <input id="uploadFile" type="file" accept=".mp4,.mov,video/mp4,video/quicktime"/>
              </div>
              <div class="row">
                <button class="btn" id="btnUpload">上传</button>
                <button class="btn secondary" id="btnUseUploadDir">用目录建索引</button>
                <button class="btn secondary" id="btnUseSelectedUpload">用选中视频建索引</button>
                <button class="btn secondary" id="btnClearSelectedIndex">清空选中视频索引</button>
              </div>
              <div class="status" id="uploadStatus"></div>
              <div class="subtle" id="uploadCount"></div>
              <div class="upload-list" id="uploadList"></div>
            </div>
          </section>
        </div>

        <div class="stack stack--right">
          <section class="panel" id="step2">
            <div class="panel-header">
              <div class="step-badge">2</div>
              <div class="panel-title">
                <h2>建立索引</h2>
                <p>选择目标目录、引擎和预处理参数。默认值偏向本地 2B 路径，适合直接跑通工作流。</p>
              </div>
              <button class="toggle" data-step="step2">收起</button>
            </div>
            <div class="panel-body tool-grid">
              <div class="field">
                <label for="idxDirectory">目标</label>
                <input id="idxDirectory" type="text" value="./sample_data"/>
              </div>
              <div class="field">
                <label for="idxEngine">引擎</label>
                <select id="idxEngine">
                  <option value="qwen2b">Qwen2B 本地</option>
                  <option value="qwen8b">Qwen3-VL-Embedding-8B 本地</option>
                  <option value="customLocal">自定义本地路径</option>
                  <option value="gemini">Gemini 云端</option>
                </select>
              </div>
              <div class="field" id="idxCustomModelRow" style="display:none">
                <label for="idxCustomModelPath">模型路径</label>
                <input id="idxCustomModelPath" type="text" value="./models/Qwen3-VL-Embedding-2B"/>
              </div>
              <div class="field-grid">
                <div class="field">
                  <label for="idxChunkDuration">片段秒数</label>
                  <input id="idxChunkDuration" type="number" value="5"/>
                </div>
                <div class="field">
                  <label for="idxOverlap">重叠秒数</label>
                  <input id="idxOverlap" type="number" value="1"/>
                </div>
              </div>
              <div class="field-grid">
                <div class="field">
                  <label for="idxResolution">分辨率高</label>
                  <input id="idxResolution" type="number" value="480"/>
                </div>
                <div class="field">
                  <label for="idxFps">目标 FPS</label>
                  <input id="idxFps" type="number" value="5"/>
                </div>
              </div>
              <div class="choices">
                <label><input id="idxPreprocess" type="checkbox" checked/>预处理</label>
                <label><input id="idxSkipStill" type="checkbox"/>跳过静态</label>
                <label><input id="idxQuantize" type="checkbox" checked/>4bit 量化</label>
              </div>
              <div class="row">
                <button class="btn success" id="btnIndex">开始建索引</button>
              </div>
              <div class="status" id="indexStatus"></div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-body">
              <div class="detail-grid">
                <div class="detail-box">
                </div>
                <div class="detail-box">
                  <div class="list" id="statsDetail"></div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </section>
    {% else %}
    <section class="page-panel page-panel--search">
      <div class="workspace">
        <div class="stack stack--left">
          <section class="panel" id="step3">
            <div class="panel-header">
              <div class="step-badge">3</div>
              <div class="panel-title">
                <h2>搜索并裁剪</h2>
              </div>
              <button class="toggle" data-step="step3">收起</button>
            </div>
            <div class="panel-body tool-grid">
              <div class="field">
                <label for="searchQuery">查询</label>
                <input id="searchQuery" type="text" placeholder="例如：前车突然变道"/>
              </div>
              <div class="field-grid search-config-grid">
                <div class="field">
                  <label for="searchResults">返回条数</label>
                  <input id="searchResults" type="number" min="1" value="5"/>
                </div>
                <div class="field">
                  <label for="searchRecall">召回数量</label>
                  <input id="searchRecall" type="number" min="1" value="25"/>
                </div>
                <div class="field">
                  <label for="searchThreshold">最低分</label>
                  <input id="searchThreshold" type="number" value="0.41" step="0.01"/>
                </div>
              </div>
              <div class="field">
                <label for="searchEngine">引擎</label>
                <select id="searchEngine">
                  <option value="qwen2b">Qwen2B 本地</option>
                  <option value="qwen8b">Qwen3-VL-Embedding-8B 本地</option>
                  <option value="customLocal">自定义本地路径</option>
                  <option value="gemini">Gemini 云端</option>
                </select>
              </div>
              <div class="field" id="searchCustomModelRow" style="display:none">
                <label for="searchCustomModelPath">模型路径</label>
                <input id="searchCustomModelPath" type="text" value="./models/Qwen3-VL-Embedding-2B"/>
              </div>
              <div class="choices">
                <label><input id="searchQuantize" type="checkbox" checked/>4bit 量化</label>
                <label><input id="searchRerank" type="checkbox" checked/>是否重排</label>
              </div>
              <div class="row">
                <button class="btn success" id="btnSearch">执行搜索</button>
                <button class="btn secondary" id="btnSelectAll">全选</button>
                <button class="btn secondary" id="btnClearSelect">清空</button>
              </div>
              <div class="status" id="searchStatus"></div>
            </div>
          </section>
        </div>

        <div class="stack stack--right">
          <section class="panel preview-panel">
            <div class="panel-header">
              <div class="step-badge">预</div>
              <div class="panel-title">
                <h2>片段预览</h2>
              </div>
            </div>
            <div class="panel-body">
              <div class="preview">
                <video id="previewPlayer" controls preload="metadata"></video>
                <div class="preview-meta" id="playerMeta">尚未选择片段。</div>
              </div>
            </div>
          </section>
        </div>
      </div>

      <section class="panel search-results-panel">
        <div class="panel-header">
          <div class="step-badge">结</div>
          <div class="panel-title">
            <h2>搜索结果</h2>
          </div>
        </div>
        <div class="panel-body tool-grid">
          <div class="table-shell">
            <table id="resultsTable" style="display:none">
              <thead>
                <tr>
                  <th>勾选</th>
                  <th>#</th>
                  <th>分数</th>
                  <th>时间段</th>
                  <th>文件</th>
                  <th>播放</th>
                </tr>
              </thead>
              <tbody id="resultsBody"></tbody>
            </table>
          </div>
          <div class="field-grid">
            <div class="field">
              <label for="trimOutputDir">输出目录</label>
              <input id="trimOutputDir" type="text" value="./clips_output"/>
            </div>
            <div class="field">
              <label for="trimOverlay">裁剪选项</label>
              <div class="choices" style="padding-top:10px">
                <label><input id="trimOverlay" type="checkbox"/>叠加 overlay</label>
              </div>
            </div>
          </div>
          <div class="row">
            <button class="btn success" id="btnTrim">裁剪已勾选</button>
          </div>
          <div class="status" id="trimStatus"></div>
          <div class="download-links" id="downloadLinks"></div>
        </div>
      </section>
    </section>
    {% endif %}
  </main>
</div>

<script>
let lastSearchResults = [];
let selectedUploadPath = null;
let uploadIndexStatus = {};
let indexSubmitting = false;
const pollTokens = {};
let uploadRefreshToken = 0;
const PAGE_MODE = "{{ page_mode }}";

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = "sentrysearch-ui-state";

function loadUiState() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") || {};
  } catch (_) {
    return {};
  }
}

function saveUiState(patch) {
  try {
    const current = loadUiState();
    localStorage.setItem(STORAGE_KEY, JSON.stringify({...current, ...patch}));
  } catch (_) {}
}

function restoreValue(id, value) {
  const el = $(id);
  if (!el || value === undefined || value === null) return;
  el.value = value;
}

const uiState = loadUiState();
restoreValue("idxEngine", uiState.idxEngine);
restoreValue("searchEngine", uiState.searchEngine);
restoreValue("searchResults", uiState.searchResults);
restoreValue("searchRecall", uiState.searchRecall);
restoreValue("idxCustomModelPath", uiState.idxCustomModelPath);
restoreValue("searchCustomModelPath", uiState.searchCustomModelPath);

let searchRecallManuallyEdited = uiState.searchRecallManual === true;

function syncSearchRecallDefault() {
  const searchResults = $("searchResults");
  const searchRecall = $("searchRecall");
  if (!searchResults || !searchRecall) return;
  if (searchRecallManuallyEdited) return;
  const resultsValue = Math.max(1, Math.floor(Number(searchResults.value) || 1));
  const defaultRecall = Math.max(resultsValue, resultsValue * 5);
  searchRecall.value = String(defaultRecall);
  saveUiState({
    searchResults: searchResults.value,
    searchRecall: searchRecall.value,
    searchRecallManual: searchRecallManuallyEdited,
  });
}

if (!searchRecallManuallyEdited) {
  syncSearchRecallDefault();
}

const setStatus = (id, text, error = false) => {
  const el = $(id);
  if (!el) return;
  el.textContent = text || "";
  el.className = error ? "status error" : "status";
};

const bytes = (value) => {
  if (!value || value < 1024) return `${value || 0} B`;
  if (value < 1048576) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1048576).toFixed(1)} MB`;
};

const begin = (id) => {
  pollTokens[id] = (pollTokens[id] || 0) + 1;
  return pollTokens[id];
};

const isLatest = (id, token) => pollTokens[id] === token;

const engine = (value) => {
  if (value === "customLocal") {
    return {backend: "local", model: "__CUSTOM__"};
  }
  if (value === "qwen2b" || value === "qwen8b") {
    return {backend: "local", model: value};
  }
  return {backend: "gemini", model: null};
};

const setCollapsed = (id, collapsed) => {
  const card = $(id);
  if (!card) return;
  card.classList.toggle("collapsed", !!collapsed);
  const button = card.querySelector(".toggle");
  if (button) button.textContent = collapsed ? "展开" : "收起";
};

function bindToggles() {
  document.querySelectorAll(".toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const id = button.getAttribute("data-step");
      const card = $(id);
      if (card) setCollapsed(id, !card.classList.contains("collapsed"));
    });
  });
}

function on(id, event, handler) {
  const el = $(id);
  if (el) el.addEventListener(event, handler);
}

function updateRows() {
  const idxCustomModelRow = $("idxCustomModelRow");
  const idxEngine = $("idxEngine");
  if (idxCustomModelRow && idxEngine) {
    idxCustomModelRow.style.display = idxEngine.value === "customLocal" ? "grid" : "none";
  }
  const searchCustomModelRow = $("searchCustomModelRow");
  const searchEngine = $("searchEngine");
  if (searchCustomModelRow && searchEngine) {
    searchCustomModelRow.style.display = searchEngine.value === "customLocal" ? "grid" : "none";
  }
}

function play(url, startTime, endTime, label) {
  const video = $("previewPlayer");
  const start = Number(startTime || 0).toFixed(1);
  const end = Number(endTime || 0).toFixed(1);
  $("playerMeta").textContent = `${label} - ${start}s 到 ${end}s`;
  const mediaUrl = new URL(url, window.location.origin);
  if (Number(endTime || 0) > Number(startTime || 0)) {
    mediaUrl.searchParams.set("start", String(Number(startTime || 0)));
    mediaUrl.searchParams.set("end", String(Number(endTime || 0)));
  }
  video.src = mediaUrl.pathname + mediaUrl.search;
  video.load();
  video.play().catch(() => {});
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      message = data.detail || data.message || JSON.stringify(data);
    } catch (_) {}
    throw new Error(message);
  }
  return await response.json();
}

async function refreshStats() {
  try {
    const data = await fetchJson("/api/stats");
    const statsDetail = $("statsDetail");
    if (!statsDetail) return;
    const tagBackend = $("tagBackend");
    const tagModel = $("tagModel");
    const tagChunks = $("tagChunks");
    const tagFiles = $("tagFiles");
    if (tagBackend) tagBackend.textContent = data.backend || "-";
    if (tagModel) tagModel.textContent = data.model || "-";
    if (tagChunks) tagChunks.textContent = data.total_chunks || 0;
    if (tagFiles) tagFiles.textContent = data.unique_source_files || 0;
    statsDetail.textContent = (data.source_files || []).join("\\n") || "暂无已建立索引的视频。";
  } catch (error) {
    if ($("statsDetail")) setStatus("statsDetail", "读取统计失败: " + error.message, true);
  }
}

async function refreshUploads() {
  const token = ++uploadRefreshToken;
  try {
    const list = $("uploadList");
    const uploadCount = $("uploadCount");
    if (!list || !uploadCount) return;
    const data = await fetchJson("/api/uploads");
    if (token !== uploadRefreshToken) return;
    const videos = data.videos || [];
    list.innerHTML = "";

    if (!videos.length) {
      if (token !== uploadRefreshToken) return;
      selectedUploadPath = null;
      uploadIndexStatus = {};
      uploadCount.textContent = "0 个视频";
      list.innerHTML = '<div class="subtle">暂无上传视频</div>';
      return;
    }

    if (!selectedUploadPath || !videos.some((video) => video.path === selectedUploadPath)) {
      selectedUploadPath = videos[0].path;
    }
    if (token !== uploadRefreshToken) return;
    uploadCount.textContent = `${videos.length} 个视频`;

    const idxEngine = $("idxEngine");
    if (!idxEngine) return;
    const idxCfg = engine(idxEngine.value);
    let idxModel = idxCfg.model;
    if (idxModel === "__CUSTOM__") {
      const idxCustomModelPath = $("idxCustomModelPath");
      idxModel = idxCustomModelPath ? idxCustomModelPath.value.trim() || null : null;
    }

    let statusResp = {status: {}};
    try {
      statusResp = await fetchJson("/api/index-status", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          source_files: videos.map((video) => video.path),
          backend: idxCfg.backend,
          model: idxModel,
        }),
      });
    } catch (_) {
      statusResp = {status: {}};
    }
    if (token !== uploadRefreshToken) return;
    uploadIndexStatus = statusResp.status || {};

    for (const video of videos) {
      if (token !== uploadRefreshToken) return;
      const row = document.createElement("label");
      row.className = "item";

      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "up";
      radio.checked = video.path === selectedUploadPath;
      radio.addEventListener("change", () => {
        selectedUploadPath = video.path;
      });

      const name = document.createElement("code");
      name.textContent = video.name;

      const size = document.createElement("span");
      size.className = "meta";
      size.textContent = bytes(video.size_bytes);

      const tag = document.createElement("span");
      tag.className = "badge-pill";
      tag.textContent = uploadIndexStatus[video.path] ? "已索引" : "未索引";

      row.appendChild(radio);
      row.appendChild(name);
      row.appendChild(size);
      row.appendChild(tag);
      list.appendChild(row);
    }
  } catch (error) {
    if (token !== uploadRefreshToken) return;
    setStatus("uploadStatus", "读取上传列表失败: " + error.message, true);
    const uploadCount = $("uploadCount");
    if (uploadCount) uploadCount.textContent = "";
  }
}

async function poll(jid, id, onSuccess) {
  const token = begin(id);
  while (true) {
    if (!isLatest(id, token)) return;

    let job = null;
    try {
      job = await fetchJson("/api/jobs/" + jid);
    } catch (error) {
      setStatus(id, "Polling failed, retrying...\\n" + error.message, true);
      await new Promise((resolve) => setTimeout(resolve, 1200));
      continue;
    }

    if (!isLatest(id, token)) return;

    const progress = job.progress || {};
    const lines = [
      "Job: " + job.job_type,
      "Status: " + job.status,
      progress.backend ? "Backend: " + progress.backend : "",
      progress.model ? "Model: " + progress.model : "",
      progress.phase ? "Phase: " + progress.phase : "",
      progress.current_file ? "File: " + progress.current_file : "",
      progress.file_progress_percent !== undefined ? "File progress: " + progress.file_progress_percent + "% (" + (progress.current_chunk || 0) + "/" + (progress.total_chunks_in_file || "?") + ")" : "",
      progress.processed_files !== undefined ? "Files: " + progress.processed_files + "/" + (progress.total_files || "?") : "",
      progress.current !== undefined ? "Chunks: " + progress.current + "/" + (progress.total || "?") : "",
    ].filter(Boolean).join("\\n");

    setStatus(id, lines);

    if (job.status === "failed") {
      setStatus(id, lines + "\\n\\nError:\\n" + job.error, true);
      return;
    }
    if (job.status === "succeeded") {
      onSuccess(job.result || {});
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

on("idxEngine", "change", () => {
  saveUiState({idxEngine: $("idxEngine") ? $("idxEngine").value : null});
  updateRows();
  refreshUploads();
});
on("searchEngine", "change", () => {
  saveUiState({searchEngine: $("searchEngine") ? $("searchEngine").value : null});
  updateRows();
});
on("searchResults", "input", () => {
  saveUiState({searchResults: $("searchResults") ? $("searchResults").value : null});
  if (!searchRecallManuallyEdited) {
    syncSearchRecallDefault();
  }
});
on("searchRecall", "input", () => {
  searchRecallManuallyEdited = true;
  saveUiState({
    searchRecall: $("searchRecall") ? $("searchRecall").value : null,
    searchRecallManual: true,
  });
});
on("idxCustomModelPath", "input", () => {
  saveUiState({idxCustomModelPath: $("idxCustomModelPath") ? $("idxCustomModelPath").value : null});
});
on("searchCustomModelPath", "input", () => {
  saveUiState({searchCustomModelPath: $("searchCustomModelPath") ? $("searchCustomModelPath").value : null});
});

on("btnUseUploadDir", "click", async () => {
  try {
    const data = await fetchJson("/api/uploads");
    $("idxDirectory").value = data.upload_dir;
    setStatus("uploadStatus", "已将索引目标设置为上传目录。");
  } catch (error) {
    setStatus("uploadStatus", "设置目录失败: " + error.message, true);
  }
});

on("btnUseSelectedUpload", "click", () => {
  if (!selectedUploadPath) {
    setStatus("uploadStatus", "请先上传并选中一个视频。", true);
    return;
  }
  $("idxDirectory").value = selectedUploadPath;
  setStatus("uploadStatus", "已将索引目标设置为选中视频。");
});

on("btnClearSelectedIndex", "click", async () => {
  if (!selectedUploadPath) {
    setStatus("uploadStatus", "请先选中一个视频。", true);
    return;
  }
  const idxCfg = engine($("idxEngine").value);
  let idxModel = idxCfg.model;
  if (idxModel === "__CUSTOM__") idxModel = $("idxCustomModelPath").value.trim() || null;

  setStatus("uploadStatus", "正在清空选中视频索引...");
  try {
    const data = await fetchJson("/api/index/clear", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        source_file: selectedUploadPath,
        backend: idxCfg.backend,
        model: idxModel,
      }),
    });
    setStatus("uploadStatus", "清空完成，已删除片段: " + (data.removed_chunks || 0));
    await refreshUploads();
    await refreshStats();
  } catch (error) {
    setStatus("uploadStatus", "清空索引失败: " + error.message, true);
  }
});

on("btnUpload", "click", async () => {
  const input = $("uploadFile");
  if (!input.files || !input.files.length) {
    setStatus("uploadStatus", "请先选择视频文件。", true);
    return;
  }

  const form = new FormData();
  form.append("file", input.files[0]);
  setStatus("uploadStatus", "上传中...");

  try {
    const response = await fetch("/api/upload-video", {method: "POST", body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "上传失败");
    selectedUploadPath = data.path;
    input.value = "";
    setStatus("uploadStatus", "上传成功: " + data.name);
    await refreshUploads();
    setCollapsed("step1", true);
  } catch (error) {
    setStatus("uploadStatus", "上传失败: " + error.message, true);
  }
});

on("btnIndex", "click", async () => {
  if (indexSubmitting) return;
  indexSubmitting = true;

  const button = $("btnIndex");
  button.disabled = true;
  setStatus("indexStatus", "Submitting index job...");

  const cfg = engine($("idxEngine").value);
  let model = cfg.model;
  if (model === "__CUSTOM__") {
    model = $("idxCustomModelPath").value.trim();
    if (!model) {
      setStatus("indexStatus", "Please fill local model path.", true);
      button.disabled = false;
      indexSubmitting = false;
      return;
    }
  }

  const base = {
    directory: $("idxDirectory").value,
    chunk_duration: Number($("idxChunkDuration").value),
    overlap: Number($("idxOverlap").value),
    preprocess: $("idxPreprocess").checked,
    target_resolution: Number($("idxResolution").value),
    target_fps: Number($("idxFps").value),
    skip_still: $("idxSkipStill").checked,
    backend: cfg.backend,
    model: model,
    quantize: $("idxQuantize").checked ? true : null,
    force_reindex: false,
  };

  try {
    let response = await fetch("/api/index", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(base),
    });
    let data = await response.json();

    if (response.status === 409 && data.code === "indexed_exists") {
      if (!confirm(`Detected ${data.indexed_count}/${data.total_videos} videos already indexed.\nClear old index and rebuild?`)) {
        setStatus("indexStatus", "Rebuild canceled.");
        return;
      }
      response = await fetch("/api/index", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...base, force_reindex: true}),
      });
      data = await response.json();
    }

    if (!response.ok) throw new Error(data.detail || data.message || "Submit failed");

    setStatus("indexStatus", "Job queued: " + data.job_id);
    await poll(data.job_id, "indexStatus", (result) => {
      const noWork = (result.indexed_files || 0) === 0 && (result.rebuilt_files || 0) === 0 && (result.removed_chunks || 0) === 0;
      const message = noWork
        ? "No new indexing executed. All target videos are already indexed.\\nUse force reindex or clear index first."
        : "Index completed\\nNew videos:" + result.indexed_files + "\\nRebuilt videos:" + (result.rebuilt_files || 0) + "\\nRemoved old chunks:" + (result.removed_chunks || 0) + "\\nNew chunks:" + result.indexed_chunks + "\\nSkipped still:" + result.skipped_chunks;
      setStatus("indexStatus", message);
      refreshStats();
      refreshUploads();
    });
  } catch (error) {
    setStatus("indexStatus", "Index failed: " + error.message, true);
  } finally {
    button.disabled = false;
    indexSubmitting = false;
  }
});

on("btnSearch", "click", async () => {
  setStatus("searchStatus", "正在搜索...");
  const cfg = engine($("searchEngine").value);
  let model = cfg.model;
  if (model === "__CUSTOM__") {
    model = $("searchCustomModelPath").value.trim();
    if (!model) {
      setStatus("searchStatus", "请填写本地模型路径。", true);
      return;
    }
  }

  const payload = {
    query: $("searchQuery").value,
    results: Number($("searchResults").value),
    recall: Number($("searchRecall").value),
    threshold: Number($("searchThreshold").value),
    backend: cfg.backend,
    model: model,
    quantize: $("searchQuantize").checked ? true : null,
    rerank: $("searchRerank").checked,
  };

  try {
    const data = await fetchJson("/api/search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });

    lastSearchResults = data.results || [];
    const body = $("resultsBody");
    body.innerHTML = "";

    if (!lastSearchResults.length) {
      $("resultsTable").style.display = "none";
      setStatus("searchStatus", data.message || "没有找到结果。");
      return;
    }

    $("resultsTable").style.display = "table";
    lastSearchResults.forEach((result, index) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><input type="checkbox" class="trim-select" data-idx="${index}"/></td>
        <td>${index + 1}</td>
        <td>${result.similarity_score.toFixed(3)}</td>
        <td>${result.start_time.toFixed(1)}s - ${result.end_time.toFixed(1)}s</td>
        <td><code>${result.source_file}</code></td>
        <td><button class="mini" data-play="${index}">播放</button></td>
      `;
      body.appendChild(row);
    });

    body.querySelectorAll("button[data-play]").forEach((button) => {
      button.addEventListener("click", () => {
        const index = Number(button.getAttribute("data-play"));
        const result = lastSearchResults[index];
        play(result.preview_url, result.start_time, result.end_time, "搜索结果#" + (index + 1));
      });
    });

    setStatus("searchStatus", "搜索完成，共 " + lastSearchResults.length + " 条");
  } catch (error) {
    setStatus("searchStatus", "搜索失败: " + error.message, true);
  }
});

on("btnSelectAll", "click", () => {
  document.querySelectorAll(".trim-select").forEach((checkbox) => {
    checkbox.checked = true;
  });
});

on("btnClearSelect", "click", () => {
  document.querySelectorAll(".trim-select").forEach((checkbox) => {
    checkbox.checked = false;
  });
});

on("btnTrim", "click", async () => {
  const button = $("btnTrim");
  const checked = [...document.querySelectorAll(".trim-select:checked")];
  if (!checked.length) {
    setStatus("trimStatus", "请先勾选要裁剪的结果。", true);
    return;
  }

  button.disabled = true;
  setStatus("trimStatus", "正在提交裁剪任务...");

  try {
    const job = await fetchJson("/api/trim", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        results: lastSearchResults,
        selected_indices: checked.map((checkbox) => Number(checkbox.getAttribute("data-idx"))),
        output_dir: $("trimOutputDir").value,
        overlay: $("trimOverlay").checked,
      }),
    });

    await poll(job.job_id, "trimStatus", (result) => {
      const box = $("downloadLinks");
      box.innerHTML = "";
      (result.clips || []).forEach((clip, index) => {
        const row = document.createElement("div");
        const preview = document.createElement("button");
        preview.className = "mini";
        preview.textContent = "播放片段 " + (index + 1);
        preview.addEventListener("click", () => play(clip.preview_url, 0, 0, "裁剪结果#" + (index + 1)));

        const link = document.createElement("a");
        link.href = clip.download_url;
        link.textContent = "下载片段 " + (index + 1) + " - " + clip.path;

        row.appendChild(preview);
        row.appendChild(link);
        box.appendChild(row);
      });

      setStatus("trimStatus", "裁剪完成，输出目录: " + (result.output_dir || "-"));
      setCollapsed("step3", true);
    });
  } catch (error) {
    setStatus("trimStatus", "裁剪失败: " + error.message, true);
  } finally {
    button.disabled = false;
  }
});

bindToggles();
if (PAGE_MODE === "index") {
  refreshStats();
  refreshUploads();
  setTimeout(() => {
    refreshUploads();
  }, 300);
}
updateRows();
</script>
</body>
</html>
"""



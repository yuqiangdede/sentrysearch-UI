from __future__ import annotations
import logging
import os, threading, traceback, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from jinja2 import Environment
from pydantic import BaseModel, Field
from .services import (
    clear_video_index,
    get_index_rebuild_candidates,
    get_stats,
    get_video_index_status,
    run_index,
    run_search,
    run_trim,
)

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads" / "videos"
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
    threshold: float = Field(default=0.41, ge=0.0, le=1.0)
    backend: str | None = Field(default=None, pattern="^(gemini|local)?$")
    model: str | None = None
    quantize: bool | None = None
    verbose: bool = False


class SearchResultItem(BaseModel):
    source_file: str
    start_time: float
    end_time: float
    similarity_score: float
    preview_url: str | None = None


class TrimRequest(BaseModel):
    results: list[SearchResultItem]
    selected_indices: list[int]
    output_dir: str = "~/sentrysearch_clips"
    overlay: bool = False


class IndexStatusRequest(BaseModel):
    source_files: list[str]
    backend: str | None = Field(default=None, pattern="^(gemini|local)?$")
    model: str | None = None


class ClearVideoIndexRequest(BaseModel):
    source_file: str = Field(min_length=1)
    backend: str | None = Field(default=None, pattern="^(gemini|local)?$")
    model: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="SentrySearch Web", version="0.1.0")
    jobs = JobManager()
    page = Environment(autoescape=True).from_string(TEMPLATE)

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return page.render()

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
        res = run_search(query=payload.query, n_results=payload.results, threshold=payload.threshold, backend=payload.backend, model=payload.model, quantize=payload.quantize, verbose=payload.verbose)
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
    def api_media(token: str) -> FileResponse:
        p = jobs.file(token)
        if p is None or not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="file not found")
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
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SentrySearch</title>
<style>
body{margin:0;font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f6f8fc}.app{max-width:1220px;margin:0 auto;padding:14px}
.hero{display:none}.topbar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;background:#fff;border:1px solid #dbe2ef;border-radius:12px;padding:8px 10px}.chip{background:#eef2ff;padding:4px 10px;border-radius:999px;font-size:12px}
.btn{border:0;border-radius:8px;padding:7px 11px;background:#0f766e;color:#fff;cursor:pointer}.btn.secondary{background:#334155}.btn:disabled{opacity:.5}
.layout{display:grid;grid-template-columns:1.25fr 1fr;gap:12px;margin-top:12px}.left,.right{display:grid;gap:12px}.right{position:sticky;top:10px;max-height:calc(100vh - 20px)}
.card{background:#fff;border:1px solid #dbe2ef;border-radius:12px;padding:10px}.head{display:flex;align-items:center;gap:8px;margin-bottom:6px}.badge{width:22px;height:22px;border-radius:999px;background:#0f766e;color:#fff;display:inline-flex;justify-content:center;align-items:center;font-size:11px}
.toggle{margin-left:auto;border:1px solid #dbe2ef;background:#fff;border-radius:7px;padding:2px 8px;font-size:12px;cursor:pointer}.card.collapsed>:not(.head){display:none}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:7px}.row label{width:86px;color:#64748b;font-size:12px}input[type=text],input[type=number],select{flex:1;min-width:135px;border:1px solid #dbe2ef;border-radius:8px;padding:7px}input[type=file]{width:100%;border:1px dashed #dbe2ef;border-radius:8px;padding:7px}
.status{font-size:12px;color:#64748b;white-space:pre-wrap;word-break:break-word}.status.error{color:#dc2626}.list{max-height:150px;overflow:auto;border:1px solid #dbe2ef;border-radius:8px;padding:7px}.item{display:block;font-size:12px;margin-bottom:6px}
.results-wrap{max-height:260px;overflow:auto;border:1px solid #dbe2ef;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border-bottom:1px solid #dbe2ef;padding:5px;text-align:left}th{background:#ccfbf1}code{background:#eef2ff;border-radius:6px;padding:1px 4px}
.mini{border:1px solid #dbe2ef;background:#fff;border-radius:7px;padding:3px 7px;cursor:pointer}.player{border:1px solid #dbe2ef;border-radius:10px;overflow:hidden;background:#0b1220}.player video{width:100%;aspect-ratio:16/9;background:#020617}.meta{padding:7px;color:#cbd5e1;background:#0f172a;font-size:12px}
@media(max-width:980px){.layout{grid-template-columns:1fr}.right{position:static;max-height:none}}
</style></head><body><div class="app">
<section class="hero"><h2 style="margin:0">SentrySearch 视频检索助手</h2><div class="chips"><span class="chip" id="tagBackend">索引引擎:-</span><span class="chip" id="tagModel">模型:-</span><span class="chip" id="tagChunks">片段数:-</span><span class="chip" id="tagFiles">视频数:-</span><button class="btn secondary" id="btnRefreshStats">刷新统计</button></div><div class="status" id="statsDetail"></div></section>
<div class="layout"><div class="left">
<section class="card" id="step1"><div class="head"><span class="badge">1</span><b>上传视频</b><button class="toggle" data-step="step1">收起</button></div><div class="row"><input id="uploadFile" type="file" accept=".mp4,.mov,video/mp4,video/quicktime"/></div><div class="row"><button class="btn" id="btnUpload">上传</button><button class="btn secondary" id="btnRefreshUploads">刷新</button><button class="btn secondary" id="btnUseUploadDir">用目录建索引</button><button class="btn secondary" id="btnUseSelectedUpload">用选中视频建索引</button><button class="btn secondary" id="btnClearSelectedIndex">清空选中视频索引</button></div><div class="status" id="uploadStatus"></div><div class="list" id="uploadList"></div></section>
<section class="card" id="step2"><div class="head"><span class="badge">2</span><b>建立索引</b><button class="toggle" data-step="step2">收起</button></div><div class="row"><label>目标</label><input id="idxDirectory" type="text" value="./sample_data"/></div><div class="row"><label>引擎</label><select id="idxEngine"><option value="qwen2b">Qwen2B 本地</option><option value="customLocal">自定义本地路径</option><option value="gemini">Gemini 云端</option></select></div><div class="row" id="idxCustomModelRow" style="display:none"><label>模型路径</label><input id="idxCustomModelPath" type="text" value="./models/Qwen3-VL-Embedding-2B"/></div><div class="row"><label>片段秒数</label><input id="idxChunkDuration" type="number" value="5"/><label>重叠秒数</label><input id="idxOverlap" type="number" value="1"/></div><div class="row"><label>分辨率高</label><input id="idxResolution" type="number" value="480"/><label>目标FPS</label><input id="idxFps" type="number" value="5"/></div><div class="row"><label><input id="idxPreprocess" type="checkbox" checked/>预处理</label><label><input id="idxSkipStill" type="checkbox"/>跳过静态</label><label><input id="idxQuantize" type="checkbox"/>4bit量化</label></div><div class="row"><button class="btn" id="btnIndex">开始建索引</button></div><div class="status" id="indexStatus"></div></section>
</div>
<div class="right">
<section class="card"><div class="head"><span class="badge">预</span><b>片段预览</b></div><div class="player"><video id="previewPlayer" controls preload="metadata"></video><div class="meta" id="playerMeta">尚未选择片段。</div></div></section>
<section class="card" id="step3"><div class="head"><span class="badge">3</span><b>搜索并裁剪</b><button class="toggle" data-step="step3">收起</button></div><div class="row"><label>查询</label><input id="searchQuery" type="text" placeholder="例如：前车突然变道"/></div><div class="row"><label>返回条数</label><input id="searchResults" type="number" value="5"/><label>最低分</label><input id="searchThreshold" type="number" value="0.41" step="0.01"/></div><div class="row"><label>引擎</label><select id="searchEngine"><option value="qwen2b">Qwen2B 本地</option><option value="customLocal">自定义本地路径</option><option value="gemini">Gemini 云端</option></select></div><div class="row" id="searchCustomModelRow" style="display:none"><label>模型路径</label><input id="searchCustomModelPath" type="text" value="./models/Qwen3-VL-Embedding-2B"/></div><div class="row"><label><input id="searchQuantize" type="checkbox"/>4bit量化</label></div><div class="row"><button class="btn" id="btnSearch">执行搜索</button><button class="btn secondary" id="btnSelectAll">全选</button><button class="btn secondary" id="btnClearSelect">清空</button></div><div class="status" id="searchStatus"></div><div class="results-wrap"><table id="resultsTable" style="display:none"><thead><tr><th>勾选</th><th>#</th><th>分数</th><th>时间段</th><th>文件</th><th>播放</th></tr></thead><tbody id="resultsBody"></tbody></table></div><div class="row"><label>输出目录</label><input id="trimOutputDir" type="text" value="~/sentrysearch_clips"/></div><div class="row"><label><input id="trimOverlay" type="checkbox"/>叠加overlay</label></div><div class="row"><button class="btn" id="btnTrim">裁剪已勾选</button></div><div class="status" id="trimStatus"></div><div id="downloadLinks"></div></section>
</div></div></div>
<script>
let lastSearchResults=[],selectedUploadPath=null,uploadIndexStatus={},indexSubmitting=false;const pollTokens={};
const setStatus=(id,t,e=false)=>{const el=document.getElementById(id);el.textContent=t||"";el.className=e?"status error":"status";};
const bytes=v=>!v||v<1024?(v||0)+" B":v<1048576?(v/1024).toFixed(1)+" KB":(v/1048576).toFixed(1)+" MB";
const begin=id=>pollTokens[id]=(pollTokens[id]||0)+1,isLatest=(id,t)=>pollTokens[id]===t;
const engine=v=>v==="customLocal"?{backend:"local",model:"__CUSTOM__"}:(v==="qwen2b")?{backend:"local",model:v}:{backend:"gemini",model:null};
const setCollapsed=(id,c)=>{const card=document.getElementById(id);if(!card)return;card.classList.toggle("collapsed",!!c);const b=card.querySelector(".toggle");if(b)b.textContent=c?"展开":"收起";};
function bindToggles(){document.querySelectorAll(".toggle").forEach(b=>b.addEventListener("click",()=>{const id=b.getAttribute("data-step"),card=document.getElementById(id);if(card)setCollapsed(id,!card.classList.contains("collapsed"));}));}
function updateRows(){document.getElementById("idxCustomModelRow").style.display=document.getElementById("idxEngine").value==="customLocal"?"flex":"none";document.getElementById("searchCustomModelRow").style.display=document.getElementById("searchEngine").value==="customLocal"?"flex":"none";}
function play(url,s,e,label){const v=document.getElementById("previewPlayer");document.getElementById("playerMeta").textContent=`${label} - ${Number(s||0).toFixed(1)}s 到 ${Number(e||0).toFixed(1)}s`;v.src=`${url}#t=${Number(s||0).toFixed(1)},${Number(e||0).toFixed(1)}`;v.load();v.play().catch(()=>{});}
async function fetchJson(url,options){const r=await fetch(url,options);if(!r.ok){let m=r.status+" "+r.statusText;try{const d=await r.json();m=d.detail||d.message||JSON.stringify(d);}catch(_){}throw new Error(m);}return await r.json();}
async function refreshStats(){try{const d=await fetchJson("/api/stats");document.getElementById("tagBackend").textContent="索引引擎:"+(d.backend||"-");document.getElementById("tagModel").textContent="模型:"+(d.model||"-");document.getElementById("tagChunks").textContent="片段数:"+(d.total_chunks||0);document.getElementById("tagFiles").textContent="视频数:"+(d.unique_source_files||0);document.getElementById("statsDetail").textContent=(d.source_files||[]).join("\\n");}catch(e){setStatus("statsDetail","读取统计失败: "+e.message,true);}}
async function refreshUploads(){
  try{
    const d=await fetchJson("/api/uploads"),w=document.getElementById("uploadList"),vs=d.videos||[];
    w.innerHTML="";
    if(!vs.length){selectedUploadPath=null;uploadIndexStatus={};w.innerHTML='<div class="item">暂无上传视频</div>';return;}
    if(!selectedUploadPath||!vs.some(v=>v.path===selectedUploadPath))selectedUploadPath=vs[0].path;

    const idxCfg=engine(document.getElementById("idxEngine").value);
    let idxModel=idxCfg.model;
    if(idxModel==="__CUSTOM__")idxModel=document.getElementById("idxCustomModelPath").value.trim()||null;
    let statusResp={status:{}};
    try{
      statusResp=await fetchJson("/api/index-status",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({source_files:vs.map(v=>v.path),backend:idxCfg.backend,model:idxModel})});
    }catch(_){statusResp={status:{}};}
    uploadIndexStatus=statusResp.status||{};

    vs.forEach(v=>{
      const row=document.createElement("label");row.className="item";
      const r=document.createElement("input");r.type="radio";r.name="up";r.checked=v.path===selectedUploadPath;r.style.marginRight="6px";r.addEventListener("change",()=>selectedUploadPath=v.path);
      const n=document.createElement("code");n.textContent=v.name;
      const s=document.createElement("span");s.textContent=" | "+bytes(v.size_bytes);
      const tag=document.createElement("span");tag.textContent=uploadIndexStatus[v.path]?" [已建立索引]":" [未建立索引]";
      row.appendChild(r);row.appendChild(n);row.appendChild(s);row.appendChild(tag);w.appendChild(row);
    });
  }catch(e){setStatus("uploadStatus","读取上传列表失败: "+e.message,true);}
}
async function poll(jid,id,onSuccess){const tk=begin(id);while(true){if(!isLatest(id,tk))return;let j=null;try{j=await fetchJson("/api/jobs/"+jid);}catch(e){setStatus(id,"Polling failed, retrying...\\n"+e.message,true);await new Promise(r=>setTimeout(r,1200));continue;}if(!isLatest(id,tk))return;const p=j.progress||{};const txt=["Job:"+j.job_type,"Status:"+j.status,p.backend?("Backend:"+p.backend):"",p.model?("Model:"+p.model):"",p.phase?("Phase:"+p.phase):"",p.current_file?("File:"+p.current_file):"",p.file_progress_percent!==undefined?("File progress:"+p.file_progress_percent+"% ("+(p.current_chunk||0)+"/"+(p.total_chunks_in_file||"?")+")"):"",p.processed_files!==undefined?("Files:"+p.processed_files+"/"+(p.total_files||"?")):"",p.current!==undefined?("Chunks:"+p.current+"/"+(p.total||"?")):""].filter(Boolean).join("\\n");setStatus(id,txt);if(j.status==="failed"){setStatus(id,txt+"\\n\\nError:\\n"+j.error,true);return;}if(j.status==="succeeded"){onSuccess(j.result||{});return;}await new Promise(r=>setTimeout(r,1200));}}
document.getElementById("btnRefreshStats").addEventListener("click",refreshStats);
document.getElementById("btnRefreshUploads").addEventListener("click",refreshUploads);
document.getElementById("idxEngine").addEventListener("change",()=>{updateRows();refreshUploads();});document.getElementById("searchEngine").addEventListener("change",updateRows);
document.getElementById("btnUseUploadDir").addEventListener("click",async()=>{try{const d=await fetchJson("/api/uploads");document.getElementById("idxDirectory").value=d.upload_dir;setStatus("uploadStatus","已将索引目标设置为上传目录。");}catch(e){setStatus("uploadStatus","设置目录失败: "+e.message,true);}});
document.getElementById("btnUseSelectedUpload").addEventListener("click",()=>{if(!selectedUploadPath){setStatus("uploadStatus","请先上传并选中一个视频。",true);return;}document.getElementById("idxDirectory").value=selectedUploadPath;setStatus("uploadStatus","已将索引目标设置为选中视频。");});document.getElementById("btnClearSelectedIndex").addEventListener("click",async()=>{if(!selectedUploadPath){setStatus("uploadStatus","请先选中一个视频。",true);return;}const idxCfg=engine(document.getElementById("idxEngine").value);let idxModel=idxCfg.model;if(idxModel==="__CUSTOM__")idxModel=document.getElementById("idxCustomModelPath").value.trim()||null;setStatus("uploadStatus","正在清空选中视频索引...",false);try{const d=await fetchJson("/api/index/clear",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({source_file:selectedUploadPath,backend:idxCfg.backend,model:idxModel})});setStatus("uploadStatus","清空完成，已删除片段: "+(d.removed_chunks||0));await refreshUploads();await refreshStats();}catch(e){setStatus("uploadStatus","清空索引失败: "+e.message,true);}});
document.getElementById("btnUpload").addEventListener("click",async()=>{const inp=document.getElementById("uploadFile");if(!inp.files||!inp.files.length){setStatus("uploadStatus","请先选择视频文件。",true);return;}const f=new FormData();f.append("file",inp.files[0]);setStatus("uploadStatus","上传中...");try{const r=await fetch("/api/upload-video",{method:"POST",body:f});const d=await r.json();if(!r.ok)throw new Error(d.detail||"上传失败");selectedUploadPath=d.path;inp.value="";setStatus("uploadStatus","上传成功: "+d.name);await refreshUploads();setCollapsed("step1",true);}catch(e){setStatus("uploadStatus","上传失败: "+e.message,true);}});
document.getElementById("btnIndex").addEventListener("click",async()=>{if(indexSubmitting)return;indexSubmitting=true;const btn=document.getElementById("btnIndex");btn.disabled=true;setStatus("indexStatus","Submitting index job...");const c=engine(document.getElementById("idxEngine").value);let m=c.model;if(m==="__CUSTOM__"){m=document.getElementById("idxCustomModelPath").value.trim();if(!m){setStatus("indexStatus","Please fill local model path.",true);btn.disabled=false;indexSubmitting=false;return;}}const base={directory:document.getElementById("idxDirectory").value,chunk_duration:Number(document.getElementById("idxChunkDuration").value),overlap:Number(document.getElementById("idxOverlap").value),preprocess:document.getElementById("idxPreprocess").checked,target_resolution:Number(document.getElementById("idxResolution").value),target_fps:Number(document.getElementById("idxFps").value),skip_still:document.getElementById("idxSkipStill").checked,backend:c.backend,model:m,quantize:document.getElementById("idxQuantize").checked?true:null,force_reindex:false};try{let r=await fetch("/api/index",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(base)}),d=await r.json();if(r.status===409&&d.code==="indexed_exists"){if(!confirm(`Detected ${d.indexed_count}/${d.total_videos} videos already indexed.\\nClear old index and rebuild?`)){setStatus("indexStatus","Rebuild canceled.");return;}r=await fetch("/api/index",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...base,force_reindex:true})});d=await r.json();}if(!r.ok)throw new Error(d.detail||d.message||"Submit failed");setStatus("indexStatus","Job queued: "+d.job_id);await poll(d.job_id,"indexStatus",(x)=>{const noWork=(x.indexed_files||0)===0&&(x.rebuilt_files||0)===0&&(x.removed_chunks||0)===0;const msg=noWork?("No new indexing executed. All target videos are already indexed.\\nUse force reindex or clear index first."):("Index completed\\nNew videos:"+x.indexed_files+"\\nRebuilt videos:"+(x.rebuilt_files||0)+"\\nRemoved old chunks:"+(x.removed_chunks||0)+"\\nNew chunks:"+x.indexed_chunks+"\\nSkipped still:"+x.skipped_chunks);setStatus("indexStatus",msg);refreshStats();refreshUploads();});}catch(e){setStatus("indexStatus","Index failed: "+e.message,true);}finally{btn.disabled=false;indexSubmitting=false;}});
document.getElementById("btnSearch").addEventListener("click",async()=>{setStatus("searchStatus","正在搜索...");const c=engine(document.getElementById("searchEngine").value);let m=c.model;if(m==="__CUSTOM__"){m=document.getElementById("searchCustomModelPath").value.trim();if(!m){setStatus("searchStatus","请填写本地模型路径。",true);return;}}const payload={query:document.getElementById("searchQuery").value,results:Number(document.getElementById("searchResults").value),threshold:Number(document.getElementById("searchThreshold").value),backend:c.backend,model:m,quantize:document.getElementById("searchQuantize").checked?true:null};try{const d=await fetchJson("/api/search",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});lastSearchResults=d.results||[];const b=document.getElementById("resultsBody");b.innerHTML="";if(!lastSearchResults.length){document.getElementById("resultsTable").style.display="none";setStatus("searchStatus",d.message||"没有找到结果。");return;}document.getElementById("resultsTable").style.display="table";lastSearchResults.forEach((r,i)=>{const tr=document.createElement("tr");tr.innerHTML=`<td><input type="checkbox" class="trim-select" data-idx="${i}"/></td><td>${i+1}</td><td>${r.similarity_score.toFixed(3)}</td><td>${r.start_time.toFixed(1)}s - ${r.end_time.toFixed(1)}s</td><td><code>${r.source_file}</code></td><td><button class="mini" data-play="${i}">播放</button></td>`;b.appendChild(tr);});b.querySelectorAll("button[data-play]").forEach(el=>el.addEventListener("click",()=>{const i=Number(el.getAttribute("data-play"));const r=lastSearchResults[i];play(r.preview_url,r.start_time,r.end_time,"搜索结果#"+(i+1));}));setStatus("searchStatus","搜索完成，共 "+lastSearchResults.length+" 条");}catch(e){setStatus("searchStatus","搜索失败: "+e.message,true);}});
document.getElementById("btnSelectAll").addEventListener("click",()=>document.querySelectorAll(".trim-select").forEach(x=>x.checked=true));
document.getElementById("btnClearSelect").addEventListener("click",()=>document.querySelectorAll(".trim-select").forEach(x=>x.checked=false));
document.getElementById("btnTrim").addEventListener("click",async()=>{const btn=document.getElementById("btnTrim"),c=[...document.querySelectorAll(".trim-select:checked")];if(!c.length){setStatus("trimStatus","请先勾选要裁剪的结果。",true);return;}btn.disabled=true;setStatus("trimStatus","正在提交裁剪任务...");try{const j=await fetchJson("/api/trim",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({results:lastSearchResults,selected_indices:c.map(x=>Number(x.getAttribute("data-idx"))),output_dir:document.getElementById("trimOutputDir").value,overlay:document.getElementById("trimOverlay").checked})});await poll(j.job_id,"trimStatus",(r)=>{const box=document.getElementById("downloadLinks");box.innerHTML="";(r.clips||[]).forEach((x,i)=>{const row=document.createElement("div");const pb=document.createElement("button");pb.className="mini";pb.textContent="播放片段"+(i+1);pb.addEventListener("click",()=>play(x.preview_url,0,0,"裁剪结果#"+(i+1)));const dl=document.createElement("a");dl.href=x.download_url;dl.textContent=" 下载片段"+(i+1)+" - "+x.path;row.appendChild(pb);row.appendChild(dl);box.appendChild(row);});setStatus("trimStatus","裁剪完成，输出目录: "+(r.output_dir||"-"));setCollapsed("step3",true);});}catch(e){setStatus("trimStatus","裁剪失败: "+e.message,true);}finally{btn.disabled=false;}});
bindToggles();refreshStats();refreshUploads();updateRows();
</script></body></html>
"""



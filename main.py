import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import desc

import requests


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mteam")


# ----------------------------
# Database models
# ----------------------------
class Config(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    api_key: Optional[str] = None
    qb_url: Optional[str] = None  # e.g. http://127.0.0.1:8080
    qb_username: Optional[str] = None
    qb_password: Optional[str] = None
    # Web admin password (PBKDF2)
    admin_password_hash: Optional[str] = None
    admin_password_salt: Optional[str] = None
    poll_interval_sec: int = 300
    enabled: bool = True
    last_poll_at: Optional[datetime] = None
    last_poll_message: Optional[str] = None


class SeenTorrent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    torrent_id: str
    keyword: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DownloadTask(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    torrent_id: str
    name: Optional[str] = None
    status: str = "queued"  # queued | added | failed
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SavedSearch(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    keyword: str
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ----------------------------
# DB setup
# ----------------------------
DB_URL = os.getenv("DB_URL", "sqlite:///app.db")
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})


def init_db():
    SQLModel.metadata.create_all(engine)
    migrate_db()
    with Session(engine) as session:
        cfg = session.get(Config, 1)
        if not cfg:
            cfg = Config()
            session.add(cfg)
            session.commit()


def migrate_db():
    """Lightweight, idempotent migrations for SQLite."""
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            # Ensure new admin password columns exist on config
            cols = set()
            for row in conn.exec_driver_sql("PRAGMA table_info('config')"):
                # row: (cid, name, type, notnull, dflt_value, pk)
                cols.add(row[1])
            if "admin_password_hash" not in cols:
                conn.exec_driver_sql("ALTER TABLE config ADD COLUMN admin_password_hash VARCHAR")
            if "admin_password_salt" not in cols:
                conn.exec_driver_sql("ALTER TABLE config ADD COLUMN admin_password_salt VARCHAR")
    except Exception as e:
        logger.warning("数据库迁移检查失败: %s", e)


def get_session():
    with Session(engine) as session:
        yield session


# ----------------------------
# External clients (M-Team & qBittorrent)
# ----------------------------
class MTeamClient:
    def __init__(self, api_key: str, base: str = "https://api.m-team.cc"):
        self.base = base.rstrip("/")
        self.api_key = api_key

    def _headers(self, json_mode: bool = True):
        common = {"Accept": "application/json", "x-api-key": self.api_key}
        if json_mode:
            return {**common, "Content-Type": "application/json"}
        return {**common, "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}

    def search(self, keyword: str, page_size: int = 50) -> List[dict]:
        url = f"{self.base}/api/torrent/search"
        payload = {"keyword": keyword, "pageNumber": 1, "pageSize": page_size, "unread": False}
        r = requests.post(url, headers=self._headers(True), json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        # detect explicit error with code/message
        if isinstance(data, dict) and "code" in data and data.get("code") not in (0, "0", None):
            msg = data.get("message") or "API 错误"
            if ("key" in str(msg).lower()) or ("無效" in str(msg)) or ("invalid" in str(msg).lower()):
                raise RuntimeError(f"M-Team API Key 无效: {msg}")
            raise RuntimeError(msg)

        def pick_list(obj) -> List[dict]:
            # Try common shapes first
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj
            if isinstance(obj, dict):
                if "data" in obj:
                    return pick_list(obj["data"])
                if "content" in obj:
                    return pick_list(obj["content"])
                if "result" in obj:
                    return pick_list(obj["result"])
                if "results" in obj:
                    return pick_list(obj["results"])
                # Fallback: search first list of dicts
                for v in obj.values():
                    cand = pick_list(v)
                    if cand:
                        return cand
                # Single object case
                if "id" in obj or "name" in obj or "title" in obj:
                    return [obj]
            return []

        results = pick_list(data)
        return results or []

    def gen_dl_token(self, torrent_id: str) -> str:
        url = f"{self.base}/api/torrent/genDlToken?id={torrent_id}"
        r = requests.post(url, headers=self._headers(False), timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if data.get("code") in ("0", 0) and data.get("data"):
                return data["data"]
            if data.get("code") not in ("0", 0):
                msg = data.get("message") or "genDlToken 失败"
                if ("key" in str(msg).lower()) or ("無效" in str(msg)) or ("invalid" in str(msg).lower()):
                    raise RuntimeError(f"M-Team API Key 无效: {msg}")
                raise RuntimeError(msg)
        raise RuntimeError(f"genDlToken failed: {data}")


class QbClient:
    def __init__(self, url: str, username: str, password: str):
        self.url = url.rstrip("/")
        self.username = username
        self.password = password
        self._client = None

    def _get_client(self):
        try:
            import qbittorrentapi
        except Exception as e:
            raise RuntimeError("qbittorrent-api 未安装，请先安装依赖。") from e
        if self._client is None:
            self._client = qbittorrentapi.Client(host=self.url, username=self.username, password=self.password)
            # This will lazy login on first request; force a check
            self._client.auth_log_in()
        return self._client

    def add_url(self, url: str) -> bool:
        client = self._get_client()
        res = client.torrents_add(urls=url)
        if res is None:
            return True
        if isinstance(res, str) and res.strip().lower().startswith("ok"):
            return True
        try:
            txt = getattr(res, "text", None)
            if isinstance(txt, str) and txt.strip().lower().startswith("ok"):
                return True
        except Exception:
            pass
        raise RuntimeError(f"qBittorrent 返回异常: {res}")

    def check_connectivity(self) -> dict:
        client = self._get_client()
        try:
            ver = str(client.app_version())
            return {"ok": True, "version": ver}
        except Exception as e:
            raise RuntimeError(f"无法连接 qBittorrent: {e}")


# ----------------------------
# Poller
# ----------------------------
class Poller:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _run_loop(self):
        while self._running:
            try:
                await self.run_once()
            except Exception as e:
                logger.exception("poll loop error: %s", e)
            with Session(engine) as session:
                cfg = session.get(Config, 1)
                interval = cfg.poll_interval_sec if cfg else 300
            await asyncio.sleep(max(5, interval))

    async def run_once(self, force: bool = False):
        with Session(engine) as session:
            cfg = session.get(Config, 1)
            if not cfg:
                return
            if not cfg.enabled and not force:
                return
            if not cfg.api_key or not cfg.qb_url or not cfg.qb_username or not cfg.qb_password:
                cfg.last_poll_at = datetime.utcnow()
                cfg.last_poll_message = "配置不完整，跳过轮询"
                session.add(cfg)
                session.commit()
                return

        try:
            added_count = self._do_poll()
            msg = f"轮询完成：新增任务 {added_count} 个"
            logger.info(msg)
            with Session(engine) as session:
                cfg = session.get(Config, 1)
                if cfg:
                    cfg.last_poll_at = datetime.utcnow()
                    cfg.last_poll_message = msg
                    session.add(cfg)
                    session.commit()
        except Exception as e:
            logger.exception("轮询失败: %s", e)
            with Session(engine) as session:
                cfg = session.get(Config, 1)
                if cfg:
                    cfg.last_poll_at = datetime.utcnow()
                    cfg.last_poll_message = f"轮询失败: {e}"
                    session.add(cfg)
                    session.commit()

    def _do_poll(self) -> int:
        added = 0
        with Session(engine) as session:
            cfg = session.get(Config, 1)
            assert cfg is not None
            # Only use per-task keywords (no global keywords)
            active_tasks = session.exec(select(SavedSearch).where(SavedSearch.enabled == True)).all()  # noqa: E712
            keywords = [t.keyword.strip() for t in active_tasks if t.keyword and t.keyword.strip()]
            if not keywords:
                return 0

            mt = MTeamClient(api_key=cfg.api_key)
            qb = QbClient(cfg.qb_url, cfg.qb_username, cfg.qb_password)

            for kw in keywords:
                try:
                    results = mt.search(kw)
                except Exception as e:
                    logger.warning("搜索关键字 '%s' 失败: %s", kw, e)
                    continue

                for item in results:
                    tid = str(item.get("id") or item.get("torrentId") or item.get("tid") or "")
                    if not tid:
                        continue
                    # skip if already successfully added
                    added_before = session.exec(
                        select(DownloadTask).where((DownloadTask.torrent_id == tid) & (DownloadTask.status == "added"))
                    ).first()
                    if added_before:
                        continue

                    name = item.get("name") or item.get("title") or ""
                    task = DownloadTask(torrent_id=tid, name=name, status="queued")
                    session.add(task)
                    session.commit()

                    try:
                        url = mt.gen_dl_token(tid)
                        ok = QbClient(cfg.qb_url, cfg.qb_username, cfg.qb_password).add_url(url)
                        task.status = "added"
                        session.add(task)
                        session.commit()
                        # mark seen only after success
                        st = SeenTorrent(torrent_id=tid, keyword=kw)
                        session.add(st)
                        session.commit()
                        added += 1
                        logger.info("已添加下载: %s (%s)", name, tid)
                    except Exception as e:
                        task.status = "failed"
                        task.error = str(e)
                        session.add(task)
                        session.commit()
                        logger.warning("添加失败 %s: %s", tid, e)
        return added


poller = Poller()


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="M-Team Auto Downloader")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------------------
# Auth helpers and middleware
# ----------------------------
import base64
import hmac
import hashlib
import json as _json


SESSION_COOKIE = "session"
SESSION_MAX_AGE_SEC = 7 * 24 * 3600


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64u_decode(s: str) -> bytes:
    pad = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _get_secret() -> bytes:
    # Ephemeral secret per process unless APP_SECRET set
    sec = getattr(app.state, "secret_key", None)
    if sec is None:
        env = os.getenv("APP_SECRET")
        if env:
            sec = env.encode()
        else:
            sec = os.urandom(32)
        app.state.secret_key = sec
    return sec


def create_session_token() -> str:
    payload = {"ts": int(datetime.utcnow().timestamp())}
    data = _b64u(_json.dumps(payload).encode())
    sig = hmac.new(_get_secret(), msg=data.encode(), digestmod=hashlib.sha256).digest()
    return f"{data}.{_b64u(sig)}"


def verify_session_token(token: str) -> bool:
    try:
        data_b64, sig_b64 = token.split(".", 1)
        sig = _b64u(hmac.new(_get_secret(), msg=data_b64.encode(), digestmod=hashlib.sha256).digest())
        if not hmac.compare_digest(sig, sig_b64):
            return False
        payload = _json.loads(_b64u_decode(data_b64))
        ts = int(payload.get("ts", 0))
        if int(datetime.utcnow().timestamp()) - ts > SESSION_MAX_AGE_SEC:
            return False
        return True
    except Exception:
        return False


def hash_password(password: str, salt_b64: Optional[str] = None) -> tuple[str, str]:
    if not salt_b64:
        salt = os.urandom(16)
        salt_b64 = _b64u(salt)
    else:
        salt = _b64u_decode(salt_b64)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return _b64u(dk), salt_b64


def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    calc, _ = hash_password(password, salt_b64)
    return hmac.compare_digest(calc, hash_b64)


EXEMPT_PATHS = {"/login", "/api/login", "/setup", "/api/setup", "/favicon.ico"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static/"):
        return await call_next(request)
    # Read config
    with Session(engine) as s:
        cfg = s.get(Config, 1)
    has_pw = bool(cfg and cfg.admin_password_hash)

    # Exempt some paths always
    if path in EXEMPT_PATHS:
        return await call_next(request)

    # If no password set yet
    if not has_pw:
        # Allow setup endpoints and setup page; redirect others to /setup
        if path.startswith("/api/") and path != "/api/setup":
            return HTMLResponse(status_code=403, content="未设置管理密码")
        if path not in ("/setup", "/api/setup"):
            return HTMLResponse(status_code=302, headers={"Location": "/setup"})
        return await call_next(request)

    # Password is set: require session
    token = request.cookies.get(SESSION_COOKIE)
    if token and verify_session_token(token):
        return await call_next(request)

    # Not authenticated
    if path.startswith("/api/"):
        return HTMLResponse(status_code=401, content="unauthorized")
    # HTML -> redirect to login
    return HTMLResponse(status_code=302, headers={"Location": "/login"})


@app.on_event("startup")
async def on_startup():
    init_db()
    await poller.start()


@app.on_event("shutdown")
async def on_shutdown():
    await poller.stop()


# ----------------------------
# Pydantic schemas
# ----------------------------
class ConfigOut(BaseModel):
    api_key: Optional[str]
    qb_url: Optional[str]
    qb_username: Optional[str]
    qb_password: Optional[str]  # write-only in UI; returned as masked
    poll_interval_sec: int
    enabled: bool
    last_poll_at: Optional[datetime]
    last_poll_message: Optional[str]


class ConfigIn(BaseModel):
    api_key: Optional[str] = None
    qb_url: Optional[str] = None
    qb_username: Optional[str] = None
    qb_password: Optional[str] = None  # empty string means keep existing
    poll_interval_sec: Optional[int] = None
    enabled: Optional[bool] = None


def mask(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return "***" if s else s


def get_effective_api_key(cfg: Config) -> Optional[str]:
    # Prefer stored key, fallback to environment variable
    return cfg.api_key or os.getenv("MTEAM_API_KEY")


@app.get("/api/config", response_model=ConfigOut)
def get_config(session: Session = Depends(get_session)):
    cfg = session.get(Config, 1)
    if not cfg:
        raise HTTPException(404, "config not found")
    return ConfigOut(
        api_key=mask(cfg.api_key),
        qb_url=cfg.qb_url,
        qb_username=cfg.qb_username,
        qb_password=mask(cfg.qb_password),
        poll_interval_sec=cfg.poll_interval_sec,
        enabled=cfg.enabled,
        last_poll_at=cfg.last_poll_at,
        last_poll_message=cfg.last_poll_message,
    )


@app.put("/api/config", response_model=ConfigOut)
def update_config(body: ConfigIn, session: Session = Depends(get_session)):
    cfg = session.get(Config, 1)
    if not cfg:
        cfg = Config()
    changed = False
    # Ignore masked api key value ("***")
    if body.api_key is not None and body.api_key.strip() != "***" and body.api_key.strip() != "":
        cfg.api_key = body.api_key.strip()
        changed = True
    if body.qb_url is not None:
        cfg.qb_url = body.qb_url
        changed = True
    if body.qb_username is not None:
        cfg.qb_username = body.qb_username
        changed = True
    if body.qb_password is not None:
        if body.qb_password != "":
            cfg.qb_password = body.qb_password
            changed = True
    if body.poll_interval_sec is not None and body.poll_interval_sec > 0:
        cfg.poll_interval_sec = body.poll_interval_sec
        changed = True
    if body.enabled is not None:
        cfg.enabled = body.enabled
        changed = True
    if changed:
        session.add(cfg)
        session.commit()
    return get_config(session)


class DownloadTaskOut(BaseModel):
    id: int
    torrent_id: str
    name: Optional[str]
    status: str
    error: Optional[str]
    created_at: datetime


@app.get("/api/downloads", response_model=List[DownloadTaskOut])
def list_downloads(session: Session = Depends(get_session)):
    items = session.exec(select(DownloadTask).order_by(desc(DownloadTask.created_at))).all()
    return [
        DownloadTaskOut(
            id=i.id, torrent_id=i.torrent_id, name=i.name, status=i.status, error=i.error, created_at=i.created_at
        )
        for i in items
    ]


class DeleteDownloadOut(BaseModel):
    ok: bool
    deleted_id: int
    removed_seen: int


@app.delete("/api/downloads/{did}", response_model=DeleteDownloadOut)
def delete_download(did: int, also_unsee: bool = True, session: Session = Depends(get_session)):
    row = session.get(DownloadTask, did)
    if not row:
        raise HTTPException(404, "not found")
    tid = row.torrent_id
    session.delete(row)
    removed_seen = 0
    if also_unsee and tid:
        seen_rows = session.exec(select(SeenTorrent).where(SeenTorrent.torrent_id == tid)).all()
        for s in seen_rows:
            session.delete(s)
            removed_seen += 1
    session.commit()
    return DeleteDownloadOut(ok=True, deleted_id=did, removed_seen=removed_seen)


class TriggerOut(BaseModel):
    message: str


@app.post("/api/trigger", response_model=TriggerOut)
async def trigger_once():
    await poller.run_once(force=True)
    return TriggerOut(message="已触发一次轮询")


@app.get("/api/qb/check")
def qb_check(session: Session = Depends(get_session)):
    cfg = session.get(Config, 1)
    if not cfg or not cfg.qb_url or not cfg.qb_username or not cfg.qb_password:
        raise HTTPException(400, "未配置 qBittorrent WebUI")
    try:
        info = QbClient(cfg.qb_url, cfg.qb_username, cfg.qb_password).check_connectivity()
        return info
    except Exception as e:
        raise HTTPException(502, f"连接失败: {e}")


class StateOut(BaseModel):
    running: bool
    last_poll_at: Optional[datetime]
    last_poll_message: Optional[str]


@app.get("/api/state", response_model=StateOut)
def get_state(session: Session = Depends(get_session)):
    cfg = session.get(Config, 1)
    return StateOut(running=True, last_poll_at=cfg.last_poll_at if cfg else None, last_poll_message=cfg.last_poll_message if cfg else None)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return FileResponse("templates/index.html")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


class LoginIn(BaseModel):
    password: str


@app.post("/api/login")
def api_login(body: LoginIn, session: Session = Depends(get_session)):
    cfg = session.get(Config, 1)
    if not cfg or not cfg.admin_password_hash:
        raise HTTPException(400, "未设置密码")
    if not verify_password(body.password, cfg.admin_password_salt or "", cfg.admin_password_hash or ""):
        raise HTTPException(401, "密码错误")
    token = create_session_token()
    from fastapi import Response

    resp = Response(content=_json.dumps({"ok": True}), media_type="application/json")
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        max_age=SESSION_MAX_AGE_SEC,
        path="/",
        samesite="lax",
    )
    return resp


@app.post("/api/logout")
def api_logout():
    from fastapi import Response

    resp = Response(content=_json.dumps({"ok": True}), media_type="application/json")
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request, session: Session = Depends(get_session)):
    cfg = session.get(Config, 1)
    if cfg and cfg.admin_password_hash:
        # Already set -> redirect to /
        return HTMLResponse(status_code=302, headers={"Location": "/"})
    return templates.TemplateResponse("setup.html", {"request": request})


class SetupIn(BaseModel):
    password: str
    confirm: str


@app.post("/api/setup")
def api_setup(body: SetupIn, session: Session = Depends(get_session)):
    if not body.password or len(body.password) < 6:
        raise HTTPException(400, "密码至少 6 位")
    if body.password != body.confirm:
        raise HTTPException(400, "两次输入不一致")
    cfg = session.get(Config, 1)
    if not cfg:
        cfg = Config()
    if cfg.admin_password_hash:
        raise HTTPException(400, "已设置密码")
    h, salt = hash_password(body.password)
    cfg.admin_password_hash = h
    cfg.admin_password_salt = salt
    session.add(cfg)
    session.commit()
    return {"ok": True}


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str
    confirm: str


@app.post("/api/change_password")
def api_change_password(body: ChangePasswordIn, session: Session = Depends(get_session)):
    cfg = session.get(Config, 1)
    if not cfg or not cfg.admin_password_hash:
        raise HTTPException(400, "未设置密码")
    if not verify_password(body.current_password, cfg.admin_password_salt or "", cfg.admin_password_hash or ""):
        raise HTTPException(401, "当前密码错误")
    if not body.new_password or len(body.new_password) < 6:
        raise HTTPException(400, "新密码至少 6 位")
    if body.new_password != body.confirm:
        raise HTTPException(400, "两次输入不一致")
    h, salt = hash_password(body.new_password)
    cfg.admin_password_hash = h
    cfg.admin_password_salt = salt
    session.add(cfg)
    session.commit()
    return {"ok": True}


# ----------------------------
# Preview & Saved Searches CRUD
# ----------------------------
class PreviewIn(BaseModel):
    keyword: str
    page_size: Optional[int] = 20


class PreviewItem(BaseModel):
    id: str
    name: Optional[str] = None
    size: Optional[str] = None
    seeders: Optional[str] = None
    discount: Optional[str] = None


class PreviewOut(BaseModel):
    items: List[PreviewItem]


@app.post("/api/preview", response_model=PreviewOut)
def preview(body: PreviewIn, session: Session = Depends(get_session)):
    cfg = session.get(Config, 1)
    if not cfg:
        raise HTTPException(400, "未初始化配置")
    key = get_effective_api_key(cfg)
    if not key:
        raise HTTPException(400, "请先配置 API Key（或设置环境变量 MTEAM_API_KEY）")
    mt = MTeamClient(api_key=key)
    results = mt.search(body.keyword, page_size=body.page_size or 20)
    items: List[PreviewItem] = []
    for it in results:
        tid = str(it.get("id") or it.get("torrentId") or it.get("tid") or "")
        if not tid:
            continue
        items.append(
            PreviewItem(
                id=tid,
                name=it.get("name") or it.get("title"),
                size=it.get("size"),
                seeders=(it.get("status") or {}).get("seeders") if isinstance(it.get("status"), dict) else it.get("seeders"),
                discount=(it.get("status") or {}).get("discount") if isinstance(it.get("status"), dict) else it.get("discount"),
            )
        )
    return PreviewOut(items=items)


class SavedSearchIn(BaseModel):
    keyword: str
    enabled: Optional[bool] = True


class SavedSearchOut(BaseModel):
    id: int
    keyword: str
    enabled: bool
    created_at: datetime


@app.get("/api/saved_searches", response_model=List[SavedSearchOut])
def list_saved_searches(session: Session = Depends(get_session)):
    rows = session.exec(select(SavedSearch).order_by(desc(SavedSearch.created_at))).all()
    return [SavedSearchOut(id=r.id, keyword=r.keyword, enabled=r.enabled, created_at=r.created_at) for r in rows]


@app.post("/api/saved_searches", response_model=SavedSearchOut)
def create_saved_search(body: SavedSearchIn, session: Session = Depends(get_session)):
    row = SavedSearch(keyword=body.keyword.strip(), enabled=True if body.enabled is None else body.enabled)
    session.add(row)
    session.commit()
    session.refresh(row)
    return SavedSearchOut(id=row.id, keyword=row.keyword, enabled=row.enabled, created_at=row.created_at)


@app.put("/api/saved_searches/{sid}", response_model=SavedSearchOut)
def update_saved_search(sid: int, body: SavedSearchIn, session: Session = Depends(get_session)):
    row = session.get(SavedSearch, sid)
    if not row:
        raise HTTPException(404, "not found")
    if body.keyword is not None:
        row.keyword = body.keyword.strip()
    if body.enabled is not None:
        row.enabled = body.enabled
    session.add(row)
    session.commit()
    session.refresh(row)
    return SavedSearchOut(id=row.id, keyword=row.keyword, enabled=row.enabled, created_at=row.created_at)


@app.delete("/api/saved_searches/{sid}")
def delete_saved_search(sid: int, session: Session = Depends(get_session)):
    row = session.get(SavedSearch, sid)
    if not row:
        raise HTTPException(404, "not found")
    session.delete(row)
    session.commit()
    return {"ok": True}


class RunOut(BaseModel):
    message: str
    attempted: int
    added: int
    failed: int
    skipped_added: int
    errors: List[str] = []


@app.post("/api/saved_searches/{sid}/run", response_model=RunOut)
async def run_saved_search(sid: int, session: Session = Depends(get_session)):
    row = session.get(SavedSearch, sid)
    if not row:
        raise HTTPException(404, "not found")

    def do_once_for(keyword: str):
        counts = {"attempted": 0, "added": 0, "failed": 0, "skipped_added": 0}
        errors: List[str] = []
        cfg = session.get(Config, 1)
        if not cfg or not cfg.api_key or not cfg.qb_url or not cfg.qb_username or not cfg.qb_password:
            raise HTTPException(400, "配置不完整：需提供 API Key 与 qBittorrent 地址/用户名/密码")
        mt = MTeamClient(api_key=cfg.api_key)
        qb = QbClient(cfg.qb_url, cfg.qb_username, cfg.qb_password)
        results = mt.search(keyword)
        for item in results:
            tid = str(item.get("id") or item.get("torrentId") or item.get("tid") or "")
            if not tid:
                continue
            already = session.exec(
                select(DownloadTask).where((DownloadTask.torrent_id == tid) & (DownloadTask.status == "added"))
            ).first()
            if already:
                counts["skipped_added"] += 1
                continue
            name = item.get("name") or item.get("title") or ""
            task = DownloadTask(torrent_id=tid, name=name, status="queued")
            session.add(task)
            session.commit()
            counts["attempted"] += 1
            try:
                url = mt.gen_dl_token(tid)
                ok = qb.add_url(url)
                task.status = "added"
                session.add(task)
                session.commit()
                st = SeenTorrent(torrent_id=tid, keyword=keyword)
                session.add(st)
                session.commit()
                counts["added"] += 1
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                session.add(task)
                session.commit()
                counts["failed"] += 1
                errors.append(f"{tid}: {e}")
        return counts, errors

    counts, errors = do_once_for(row.keyword)
    msg = f"已执行：新增 {counts['added']}，失败 {counts['failed']}，已存在 {counts['skipped_added']}"
    return RunOut(
        message=msg,
        attempted=counts['attempted'],
        added=counts['added'],
        failed=counts['failed'],
        skipped_added=counts['skipped_added'],
        errors=errors,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


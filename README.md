# M-Team 自动下载器 (FastAPI + Vue)

基于 FastAPI + SQLModel 的后端和 Vue 3 前端的自动化下载器：
- 后台按已保存的搜索关键词轮询 M-Team，生成下载 token 并添加到 qBittorrent。
- 仅在 qB 成功添加后才标记“已看过”，避免漏下可重试的项。
- Web 界面支持任务预览、保存/启用关键词、查看下载记录、配置、账号密码等。

## 功能特性
- 已保存任务（每个任务一个关键词，支持启用/禁用）
- 预览搜索结果后再创建任务
- 一键手动触发轮询；后台按间隔自动轮询
- qBittorrent 连接性检查；严格校验添加返回值
- 删除下载记录时同步清理对应的“已看过”
- 登录保护：首次访问需创建密码，支持登录/退出/修改密码
- 前端使用 Vue 3，右上角 toast 通知；不使用浏览器原生 alert/confirm

## 本地运行

1) 安装依赖（建议 Python 3.10+）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) 启动服务

```bash
python main.py
```

首次打开 http://127.0.0.1:8000 会重定向到 /setup 创建管理密码，随后使用 /login 登录进入主界面。

## Docker 运行

镜像基于 `public.ecr.aws/docker/library/python:3.12-slim`（Amazon ECR Public 的官方镜像别名，规避部分环境下 docker.io 镜像源异常）。

构建：

```bash
docker build -t mteam:latest .
```

运行：

```bash
docker run --rm -p 8000:8000 \
	-e APP_SECRET=your_random_secret \
	-e HOST=0.0.0.0 -e PORT=8000 \
	-v mteam_data:/data \
	mteam:latest
```

或使用 compose：

```bash
docker compose up -d
```

提示：如果 qBittorrent 运行在宿主机上，容器内可使用 `http://host.docker.internal:8080` 访问 WebUI（macOS/Windows）。

## Web 使用说明

- 首次进入跳转到“初始化密码”页面（/setup），设置后到登录页（/login）。
- 主页：查看当前状态，管理“已保存任务”（启用、运行一次、删除）。
- 创建任务：输入关键词可先预览结果，再创建保存为任务。
- 设置：
	- 账户：修改密码、退出登录
	- 配置：M-Team API Key、qB WebUI 地址/账号/密码、轮询间隔（秒）、启用开关，以及“检查 qB 连接”
- 下载日志：展示后台添加到 qB 的记录，可删除记录（同时清理“已看过”）。

## 环境变量

- HOST：监听地址，默认 `0.0.0.0`
- PORT：监听端口，默认 `8000`
- DB_URL：数据库连接串
	- 本地默认：`sqlite:///app.db`
	- Docker 默认：`sqlite:////data/app.db`（持久化到挂载卷）
- APP_SECRET：会话签名密钥。默认 `change_me`，生产务必改为随机强密钥。
- 可选：`MTEAM_API_KEY` 可通过 Web 界面配置，通常无需环境变量。

## 后端接口（节选）

- 认证相关：`POST /api/setup`、`POST /api/login`、`POST /api/logout`、`POST /api/change_password`
- 配置：`GET/PUT /api/config`（服务器会隐藏/忽略带掩码的敏感字段）
- 预览：`POST /api/preview`
- 保存任务：`GET/POST/PUT/DELETE /api/saved_searches`，`POST /api/saved_searches/{id}/run`
- 下载记录：`GET /api/downloads`、`DELETE /api/downloads/{id}`（删除会同步清理“已看过”）
- 轮询：`POST /api/trigger` 立即执行一次；`GET /api/state` 查看状态
- qB 检查：`GET /api/qb/check`

## 数据与迁移

- 使用 SQLite 持久化（默认 `app.db` 或 Docker 中 `/data/app.db`）。
- 内置轻量迁移：自动为旧库添加管理员密码字段，保留历史数据。

## 故障排查

- 拉取基础镜像失败（docker.io 400/超时）：已改用 Amazon ECR Public 别名；也可在 Docker 设置中移除/更换 Registry Mirrors 后改回 `python:3.12-slim`。
- qB 未新增任务：使用“检查 qB 连接”确认可用；在设置中核对 WebUI 地址、账号和密码。
- M-Team 返回“key无效”：在设置页更新正确的 API Key。

## 目录结构（节选）

- `main.py` 后端入口，API 与轮询逻辑
- `templates/` 登录、初始化、首页（Vue）等页面
- `static/` 前端脚本与样式（`app.js`、`style.css`）
- `Dockerfile`、`docker-compose.yml`、`.dockerignore`
- `requirements.txt`


# M-Team 自动下载器 (FastAPI)

一个简单的全栈示例：
- 后端：FastAPI + SQLModel + SQLite，后台轮询 M-Team 搜索接口并将新结果推送到 qBittorrent。
- 前端：简单的 HTML/JS 页面，用于配置 API Key、qBittorrent WebUI、轮询间隔、关键词等，并查看任务记录。

## 运行

1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 启动服务

```bash
python main.py
```

浏览器打开 http://127.0.0.1:8000 即可。

## 配置说明
- API Key: M-Team 的 API Key。
- qBittorrent WebUI: 例如 `http://127.0.0.1:8080`。
- 用户名/密码: qBittorrent WebUI 登录信息。
- 关键词: 多个词用逗号分隔，例如 `earth, alien`。
- 轮询间隔: 单位秒，建议 ≥ 60，避免过于频繁请求。

## 注意
- 本项目只在本地保存必要的信息（SQLite `app.db`）。
- 任务仅记录提交状态，不做 qBittorrent 内部状态追踪；可在 WebUI 自行查看下载进度。
- 若 M-Team 返回结构变化，请在 `MTeamClient.search` 中适配解析逻辑。

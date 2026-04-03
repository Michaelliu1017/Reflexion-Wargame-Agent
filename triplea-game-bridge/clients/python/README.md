# TripleA-LLM

**TripleA-LLM** — play TripleA wargame with an LLM. Use OpenAI GPT to control one side (e.g. Japan): purchase, deploy, move, and attack via a Bridge HTTP API.

## 架构简述

```
TripleA Host (GUI)  ←── LAN ──→  Bridge (Java, :8081)  ←── HTTP ──→  GPT Driver (Python)
     建房间、选图                   接管日本、暴露 API                   调用 OpenAI + 发 act
```

- **Host**：TripleA 主程序，建游戏房间，选地图（如 Pacific 40）。
- **Bridge**：连上 Host 并接管指定玩家（如日本），对外提供 `GET /state`、`GET /legal_actions`、`POST /act`。
- **GPT Driver**：本目录下的 Python 脚本，循环获取局面、调用 GPT 决策、向 Bridge 发送动作。

## 环境要求

| 环境 | 要求 |
|------|------|
| **Java** | JDK 17+（用于编译和运行 TripleA、Bridge） |
| **Python** | 3.10+（推荐 3.11/3.12） |
| **OpenAI** | 有效 API Key（如 gpt-4o-mini） |

## 部署步骤（需按顺序）

### 1. 克隆仓库并进入项目根目录

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

### 2. 启动 Host（第一个终端）

- 运行 TripleA GUI：
  ```bash
  ./gradlew :game-app:game-headed:run
  ```
- 在 GUI 中：**创建游戏** → 选择地图（如 **Pacific 40**）→ 设置玩家（日本设为可由网络控制）→ 记下 **端口**（默认如 3300）→ 进入**等待玩家**界面，**先不要点「开始游戏」**。

### 3. 启动 Bridge（第二个终端）

```bash
cd 你的仓库名   # 项目根目录
./gradlew :game-app:game-bridge:run --args="--host 127.0.0.1 --port 3300 --name Bot_Bridge --take Japanese"
```

- `--port` 必须与 Host 建房间时显示的端口一致。
- `--take` 必须是**该地图里日本玩家的准确名字**（常见为 `Japanese` 或 `Japan`，以地图为准）。
- 看到日志中出现 **"Bridge connected and took player"** 和 **"Bridge HTTP server listening on port 8081"** 后，回到 Host 点击 **「开始游戏」**。

### 4. 启动 GPT Driver（第三个终端）

```bash
cd 你的仓库名/clients/python
export OPENAI_API_KEY=sk-你的OpenAI密钥
./run_auto.sh --rules-file rules_zh.txt
```

- 首次运行 `run_auto.sh` 会自动创建 `.venv` 并安装 `openai`。
- 若 Bridge 不在本机或端口不是 8081：
  ```bash
  ./run_auto.sh --base-url http://Bridge机器IP:8081 --rules-file rules_zh.txt
  ```
- 全自动会一直运行，按 **Ctrl+C** 停止。

## 一键命令汇总（本地三终端）

| 终端 | 命令 |
|------|------|
| **1 - Host** | `./gradlew :game-app:game-headed:run`，然后在 GUI 建房间、选图、等玩家 |
| **2 - Bridge** | `./gradlew :game-app:game-bridge:run --args="--host 127.0.0.1 --port 3300 --name Bot_Bridge --take Japanese"` |
| **3 - GPT** | `cd clients/python && export OPENAI_API_KEY=sk-xxx && ./run_auto.sh --rules-file rules_zh.txt` |

## 常见问题

- **「当前不是日本回合」/ legal_actions 为空**：检查 Bridge 的 `--take` 是否与地图中日本玩家名完全一致（如 Pacific 40 多为 `Japanese`）。
- **GET /state 500 或超时**：确认 Host 已点击「开始游戏」，且 Bridge 已连上；可查看 Bridge 终端日志。
- **上下文超长 400**：脚本已做上下文裁剪；若仍报错，可改 `chatgpt_driver.py` 中 `KEEP_LAST_USER_ROUNDS` 为 1。
- 更多排查见：[排查「不是日本回合」.md](排查「不是日本回合」.md)、[项目实现与部署说明.md](项目实现与部署说明.md)。

## 文档索引

| 文档 | 说明 |
|------|------|
| [README_AUTO.md](README_AUTO.md) | 单次/全自动运行方式说明 |
| [项目实现与部署说明.md](项目实现与部署说明.md) | 实现原理、接口说明、跨机部署 |
| [排查「不是日本回合」.md](排查「不是日本回合」.md) | 回合识别与断线排查 |

## License

与 TripleA 主项目一致（GPL-3.0）。

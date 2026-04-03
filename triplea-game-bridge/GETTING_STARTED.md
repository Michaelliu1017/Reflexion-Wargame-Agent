# TripleA-LLM：基于 gpt-4o-mini 的 TripleA Wargame 游戏
This project enables GPT-4o-mini to control a faction in the TripleA wargame.
The system connects the TripleA game engine with an LLM through a bridge service so that GPT can make decisions such as purchasing units, placing units, moving, and attacking.

In the current setup, GPT controls Japan while other nations are controlled by built-in bots.

## 项目部署 Project Deployment


### 1.1 环境准备 Prepare the Enviroment

| Requirement | Description |
|------|------|
| **Java JDK 17+** | 用于跑 TripleA Host 和 Bridge。 |
| **Python 3.10+** | 用于跑 GPT 驱动脚本。 |
| **OpenAI API Key** | 在 [OpenAI](https://platform.openai.com) 获取，用于调用 gpt-4o-mini。 |

### 1.2 下载项目 Clone the Repository

```bash
git clone https://github.com/Michaelliu1017/triplea-llm
cd triplea-llm
```

### 1.3 启动顺序 Startup Order

#### 打开三个终端页面 
#### You will run Host, Bridge, and GPT Controller in three separate terminals.
#### 终端 1：启动 Host
#### Terminal 1: Start the Host

```bash
cd triplea-llm
./gradlew :game-app:game-headed:run
```

- 等待 GUI 打开后：
- When the GUI opens:
  - 点击 **Play** → **Host a networked game**。
  - Click **Play** → **Host a networked game**。
  - 选地图 : WW2 Pacific 2nd Edition。
  - Select Map: WW2 Pacific 2nd Edition。
  - **玩家设置**：**日本** 设为 **Human** 并取消选择前边的checkbox，**其他所有玩家**设为 **AI**（Bot）或 **Human**。
  - **Player Setting** Set Japan to Human and uncheck the checkbox in front of it and Set all other players to AI (Bot) or Human
  - 记下 **端口** 选择 3301，**不要设密码**。
  - Choose port 3301 and Do not set a password
  - 进入 **等待玩家** 界面，**先不要点「开始游戏」**。
  - Enter the Waiting for Players screen

#### 终端 2：启动 Bridge
#### Terminal 2: Start the Bridge

```bash
cd triplea-llm
./gradlew :game-app:game-bridge:run --args="--host 127.0.0.1 --port 3301 --name Bot_Yamamoto_Isoroku --take Japanese"
```

- `--port 3301` 要和 Host 里显示的端口一致。
- `--take Japanese` 要和**地图里日本方的名字**一致。
- 看到 Bot加入游戏后即可开始，点击 **「开始游戏」**。

#### 终端 3：启动 GPT 驱动
#### Terminal 3: Start the GPT Controller

```bash
cd triplea-llm/clients/python
export OPENAI_API_KEY=sk-你的OpenAI密钥
./run_auto.sh --rules-file rules_zh.txt
```

- 第一次运行会自动建 `.venv` 并装 `openai`。
- On first run: A .venv virtual environment will be created automatically and The openai package will be installed
- 之后 GPT 会持续替日本做决策（购买、部署、移动、进攻等），直到你按 **Ctrl+C** 或对局结束。
- After that, GPT will continuously control Japan, making decisions. The match continues until you press Ctrl+C
- 复制你的密钥：https://platform.openai.com/settings/organization/api-keys
- Create or copy your API key from: https://platform.openai.com/settings/organization/api-keys

至此：**GPT 实现控制日本 vs 其他国家 Bot**，在本地完成对战。
At this point: GPT controls Japan vs other AI nations locally in TripleA.

### 1.4 一键命令汇总  Quick Command Summary

假设仓库已克隆到 `triplea-llm`，在**项目根**执行：

| 终端 | 命令 |
|------|------|
| **1 - Host** | `./gradlew :game-app:game-headed:run` → 在 GUI 建房间,其他玩家选 AI,日本设为Human, 等待连接 |
| **2 - Bridge** | `./gradlew :game-app:game-bridge:run --args="--host 127.0.0.1 --port 3301 --name Bot_Bridge --take Japanese"` |
| **3 - GPT** | `cd clients/python && export OPENAI_API_KEY=sk-你的key && ./run_auto.sh --rules-file rules_zh.txt` |

### 1.5 常见问题 Troubleshooting

- **Bridge 连不上 / 没有「开始游戏」**：确认 Host 已到等待玩家界面，端口、无密码，再启动 Bridge。
- **Bridge cannot connect / no "Start Game" button**; Ensure: Host is already in the Waiting for Players screen,Correct port,No password,Start Bridge after Host
- **一直提示「当前不是日本回合」**：注意你的启动顺序，先启动host，再连接bridge。
- **Not Japan's turn" appears repeatedly on terminal**; Check your startup order: Start Host, then start Bridge, and finally start GPT Controller
- **GET /state 报错或超时**：确认 Host 已点「开始游戏」，且 Bridge 终端里已打印 "Bridge connected"。
- **GET /state timeout or error**; Ensure: The Host GUI has clicked "Start Game" The Bridge terminal shows
- 更多见：`clients/python/排查「不是日本回合」.md`、`clients/python/项目实现与部署说明.md`。

---



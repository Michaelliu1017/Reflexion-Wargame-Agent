# 本地 ChatGPT 全自动对战

不打开浏览器、不需要每次点授权，用本机 Python + OpenAI API 让 ChatGPT 一直帮日本下指令，直到你按 Ctrl+C。

## 前置条件

1. **TripleA host + bridge 已跑起来**
   - 终端 1：`./gradlew :game-app:game-headed:run`，在 GUI 里开新局，选地图（如 Pacific 40），日本 = Human，端口 3301，无密码。
   - 终端 2：`./gradlew :game-app:game-bridge:run --args="--host 127.0.0.1 --port 3301 --name Bot_Michael --take Japanese"`  
   - 确认 bridge 连上并接管日本（GUI 里能看到 Bot_Michael / 日本被占）。

2. **本机已安装 Python 3，并安装 openai（推荐用虚拟环境）**
   - 若系统提示 `externally-managed-environment`，请在项目下用虚拟环境：
     ```bash
     cd .../triplea/clients/python
     python3 -m venv .venv
     source .venv/bin/activate   # Windows: .venv\Scripts\activate
     pip install openai
     ```
   - 之后每次运行前先 `source .venv/bin/activate`，再用 `python chatgpt_driver.py --auto`。
   - 或直接使用脚本（会自动用 .venv）：
     ```bash
     export OPENAI_API_KEY=sk-你的key
     ./run_auto.sh
     ```

3. **设置 OpenAI API Key**
   ```bash
   export OPENAI_API_KEY=sk-你的key
   ```
   （Windows 用 `set OPENAI_API_KEY=sk-...` 或在系统环境变量里设。）

## 运行方式

进入客户端目录：

```bash
cd /Users/michaelliu/Documents/swe/cursor_test/triplea/clients/python
```

### 单次执行（只跑一轮决策）

```bash
python chatgpt_driver.py
```

模型会拉取 state / legal_actions，然后执行若干次 BUY_UNITS / PLACE_UNITS / END_TURN，用完后脚本退出。

### 全自动对战（一直跑，直到你 Ctrl+C）

```bash
python chatgpt_driver.py --auto
```

- 每轮模型做完当前回合后，脚本会自动再发一条「继续」；
- 下一轮会再次 get_state / get_legal_actions 并继续操作日本；
- 直到你按 **Ctrl+C** 或对局断开，脚本才会退出。

### bridge 不在本机时

```bash
python chatgpt_driver.py --auto --base-url http://其他机器IP:8081
```

## 输出说明

- `[GPT]` 开头：模型用中文说的总结或说明；
- `[Tool] get_game_state -> ok=...`：调用 get_state 的结果；
- `[Tool] do_action -> ok=True`：执行 BUY_UNITS / PLACE_UNITS / END_TURN 成功；
- `[Tool] do_action -> ok=False error=...`：执行失败，模型会根据错误决定下一步（例如改 END_TURN）。

按上述步骤即可在本地实现「ChatGPT 全自动对战」，无需浏览器授权。

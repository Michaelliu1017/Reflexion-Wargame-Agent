# 用 ChatGPT 控制 TripleA（日本）

Bridge 提供 HTTP API：`GET /state`、`GET /legal_actions`、`POST /act`。用 Python 包一层后，有两种常见方式对接 ChatGPT。

---

## 方式一：OpenAI API + Function Calling（推荐）

在本地跑一个 Python 服务，把「读状态」和「执行动作」做成 **tools**，让 ChatGPT 决定每一步动作。

### 1. 安装依赖

```bash
pip install openai requests
```

### 2. 定义 tools（给 OpenAI 的 schema）

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_game_state",
            "description": "Get current TripleA game state: round, phase, Japanese PUs, territories, units, purchase/place options.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_legal_actions",
            "description": "Get list of legal action types for current turn, e.g. BUY_UNITS, PLACE_UNITS, END_TURN.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "do_action",
            "description": "Execute one action. For BUY_UNITS pass units dict e.g. {\"infantry\": 3}. For PLACE_UNITS pass placements list. For END_TURN pass nothing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["BUY_UNITS", "PLACE_UNITS", "END_TURN"]},
                    "units": {"type": "object", "description": "For BUY_UNITS: {\"unitType\": count}"},
                    "placements": {"type": "array", "items": {"type": "object", "properties": {"territory": {}, "unitType": {}, "count": {}}}}
                },
                "required": ["action_type"]
            }
        }
    }
]
```

### 3. 实现 tool 调用（调 bridge）

```python
from triplea_bridge_client import TripleABridgeClient

client = TripleABridgeClient("http://localhost:8081")

def run_tool(name, arguments):
    if name == "get_game_state":
        return client.get_state()
    if name == "get_legal_actions":
        return client.get_legal_actions()
    if name == "do_action":
        at = arguments.get("action_type")
        if at == "END_TURN":
            return client.act_end_turn()
        if at == "BUY_UNITS":
            return client.act_buy(arguments.get("units") or {})
        if at == "PLACE_UNITS":
            return client.act_place(arguments.get("placements") or [])
    return {"error": "unknown tool"}
```

### 4. 对话循环（简化）

```python
from openai import OpenAI

openai_client = OpenAI()  # 需要 OPENAI_API_KEY

messages = [
    {"role": "system", "content": "You control the Japanese player in a TripleA game. Use get_game_state and get_legal_actions to see the situation, then do_action to BUY_UNITS, PLACE_UNITS, or END_TURN. Reply briefly in Chinese."}
]

while True:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    msg = response.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:
        print(msg.content)
        break
    for tc in msg.tool_calls:
        result = run_tool(tc.function.name, json.loads(tc.function.arguments))
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False)
        })
```

把上面拼成一个脚本（例如 `chatgpt_driver.py`），运行后 ChatGPT 会反复调 get_state / get_legal_actions / do_action，直到它选择 END_TURN 或你打断。

---

## 方式二：Custom GPT + 外部 API（需公网或 ngrok）

1. 在 OpenAI 网页创建 **Custom GPT**，在 Instructions 里说明：
   - “你控制 TripleA 里的日本。用户会给你当前游戏状态和合法动作，你回复一个 JSON：`{\"action_type\": \"BUY_UNITS\"|\"PLACE_UNITS\"|\"END_TURN\", \"units\": {...}, \"placements\": [...]}`。”
2. 在 Custom GPT 的 **Actions** 里配置你的 API：
   - 若 bridge 在本地，用 **ngrok** 暴露：`ngrok http 8081`，得到 `https://xxx.ngrok.io`。
   - Schema 里定义三个 operation：`GET /state`、`GET /legal_actions`、`POST /act`（body 为上面的 JSON）。
3. 你或另一个小脚本定期拉 `/state` 和 `/legal_actions`，把结果贴给 Custom GPT，再把 GPT 返回的 JSON 发给 `POST /act`。

---

## 小结

| 方式 | 优点 | 缺点 |
|------|------|------|
| **OpenAI API + Function Calling** | 全自动、可本地跑、逻辑清晰 | 需要写 Python、用 API Key |
| **Custom GPT + 公网 API** | 在网页里和 ChatGPT 对话即可 | 需 ngrok 或服务器、配置 Actions |

建议先用 **方式一** 在本地跑通一个 `chatgpt_driver.py`，再按需做成 Custom GPT。

# 如何教 GPT 游戏规则

本地驱动（`chatgpt_driver.py`）里，有两种方式让 GPT 按 TripleA 规则来玩。

---

## 方式一：用规则文件（推荐）

把你想让模型遵守的规则写在一个**文本文件**里（UTF-8），用 `--rules-file` 传入，脚本会把这些内容塞进系统提示，模型就会按这些规则做决策。

### 1. 新建规则文件

在 `clients/python/` 下建一个文件，例如 `rules_zh.txt` 或 `rules.txt`，用中文或英文写规则摘要。不必写完整规则书，写**和决策相关的部分**即可。

### 2. 建议包含的内容

- **回合阶段顺序**  
  例如：政治 → 购买 → 战斗移动 → 战斗 → 非战斗移动 → 部署新单位 → 收税 → 下一玩家。  
  这样模型才知道什么时候该 BUY、什么时候该 PLACE、什么时候只能 END_TURN。

- **购买阶段**  
  - PUs 怎么花、哪些单位能买、大致成本（如步兵 3、战斗机 10）。  
  - 说明：`get_game_state` 里的 `purchaseOptions` 会给出当前可买单位与成本，模型应据此决定 `units`，不要瞎编单位名。

- **部署阶段**  
  - 新买的单位必须在本回合部署；只能放在己方领土或符合规则的海域。  
  - 说明：`placeOptions` 会给出可放置的地块，模型应只在这些地块里选，且 `territory` 名字必须和 state 里一致（如 `"Japan"`、`"6 Sea Zone"`）。

- **日本常用策略（可选）**  
  例如：优先守本土、占中国/东南亚产 PU、航母与舰载机配合等。写几条即可，避免模型乱打。

### 3. 运行时带上规则文件

```bash
# 全自动 + 规则文件
./run_auto.sh --rules-file rules_zh.txt

# 或先激活 venv 再跑
source .venv/bin/activate
python chatgpt_driver.py --auto --rules-file rules_zh.txt
```

规则文件路径可以是相对路径（相对当前工作目录）或绝对路径。

---

## 方式二：直接改脚本里的 SYSTEM_PROMPT

不想单独维护文件的话，可以**直接改代码**里的系统提示。

1. 打开 `chatgpt_driver.py`，找到 `SYSTEM_PROMPT = """..."""` 这一段。
2. 在第一个 `"""` 后面、现有说明之前，加一段规则摘要，例如：

```text
回合阶段大致顺序：政治 → 购买 → 战斗移动 → 战斗 → 非战斗移动 → 部署 → 收税。
购买阶段：根据 japan.pus 和 purchaseOptions 决定买什么，单位名和 cost 以 state 为准。
部署阶段：根据 placeOptions 决定放哪，territory 名称必须与 state 中完全一致。
...
```

3. 存盘后照常运行 `./run_auto.sh` 或 `python chatgpt_driver.py --auto`。

这样不用传 `--rules-file`，规则会一直跟着脚本走；缺点是不如单独文件好维护和版本管理。

---

## 规则文件示例（可复制后按需删改）

在 `clients/python/` 下建 `rules_zh.txt`，例如：

```text
【TripleA 日本方规则摘要】

1. 回合阶段顺序（按 stepName/phaseName 判断）：
   政治(japanesePolitics) → 购买(japanesePurchase) → 战斗移动 → 战斗 → 非战斗移动 → 部署(japanesePlace) → 收税等 → 下一玩家。
   只有购买阶段能 BUY_UNITS，只有部署阶段能 PLACE_UNITS；其他阶段若 legal_actions 里有 END_TURN 就只发 END_TURN。

2. 购买阶段：
   - 用 get_game_state 看 japan.pus 和 purchaseOptions（每项有 unitType、cost、maxAffordable）。
   - do_action(BUY_UNITS, units={"infantry": 3}) 表示买 3 个步兵；单位类型必须与 purchaseOptions 里一致，总花费不能超过 pus。

3. 部署阶段：
   - 用 placeOptions 看哪些领土可放、容量等；placements 里的 territory 必须与 state 里地名完全一致（如 "Japan"、"6 Sea Zone"）。
   - 只能部署本回合购买或尚未放置的单位，数量不能超过持有数。

4. 日本策略建议：
   - 优先保证本土和关键产 PU 省份；航母与战斗机配合控制海域；中国战场按 PU 和战线权衡推进。
```

保存后运行：

```bash
./run_auto.sh --rules-file rules_zh.txt
```

---

## 小结

| 方式           | 优点                 | 缺点           |
|----------------|----------------------|----------------|
| `--rules-file` | 规则与代码分离，易改 | 需多传一个参数 |
| 改 SYSTEM_PROMPT | 一次改完永久生效     | 规则和代码混在一起 |

推荐：用 **`--rules-file`** 维护一个 `rules_zh.txt`，把官方规则书里和「阶段顺序、购买、部署、日本策略」相关的部分摘进去，需要时再补充几句即可。

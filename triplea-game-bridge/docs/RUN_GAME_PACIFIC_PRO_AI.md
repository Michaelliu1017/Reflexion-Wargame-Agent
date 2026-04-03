# 运行游戏：加载 Pacific 2nd 地图并全部设为 PRO_AI

本文档基于项目 README 与源码整理，说明如何运行 TripleA、加载 **Pacific 2nd version** 地图，并将**全部国家设为 PRO_AI（Hard AI）** 后开始对局。

---

## 所有运行游戏的指令（总览）

### 环境要求

- **JDK 21**（见 [Developer Setup](development/README.md)）
- 可选首次：`./gradlew downloadAssets`（下载资源，避免启动后缺图）

---

### 1. GUI 客户端（有界面）

| 用途 | 指令 |
|------|------|
| 正常启动游戏 | `./gradlew :game-app:game-headed:run` |
| 指定游戏 XML 并进入本地设置 | `./gradlew :game-app:game-headed:run --args="triplea.game=<游戏XML绝对路径>"` |
| 指定游戏并直接打开「本地游戏」界面 | `./gradlew :game-app:game-headed:run --args="triplea.game=<游戏XML绝对路径> triplea.start=local"` |
| 启动后直接进大厅 | `./gradlew :game-app:game-headed:run --args="triplea.start=lobby"` |
| 以服务器模式启动（开房） | 需设 JVM 参数，见下方「GUI 可用的系统属性」 |

---

### 2. Headless（无界面 Bot）

| 用途 | 指令 |
|------|------|
| 本地跑 Bot（连本地 Lobby） | 先创建目录 `mkdir -p $HOME/triplea/downloadedMaps`，再执行：<br>`MAPS_FOLDER=$HOME/triplea/downloadedMaps ./gradlew :game-app:game-headless:run` |
| 说明 | **MAPS_FOLDER 必须是已存在的目录**，不能写占位符 `/path/to/downloadedMaps`，否则会报错退出。 |

Headless 的 Lobby URI、端口、Bot 名等可在 `game-app/game-headless/build.gradle` 的 `run { ... }` 里改（如 `triplea.lobby.uri`、`triplea.port`、`triplea.name`）。

---

### 3. Docker 方式跑 Headless Bot

```bash
cd game-app/game-headless/
./clean-run-docker.sh
```

需配合本地 Lobby（如 `docker compose up` 在 lobby-server 目录）。

---

### 4. 生产环境跑 Bot（Linux + Docker）

```bash
BOT_NAME=你的Bot名
sudo ufw allow 4000
docker pull ghcr.io/triplea-game/bot:latest
MAPS_FOLDER=/home/$USER/triplea/downloadedMaps
docker run \
  --env BOT_NAME=$BOT_NAME \
  --env LOBBY_URI=https://prod.triplea-game.org \
  -v $MAPS_FOLDER:/downloadedMaps \
  -p 4000:4000 \
  ghcr.io/triplea-game/bot:latest
```

---

### 5. GUI 可用的系统属性（通过 `--args` 传入）

`--args` 里用空格分隔多个 `name=value`，会设置系统属性。支持的 name：

| 属性 | 含义 | 示例值 |
|------|------|--------|
| `triplea.game` | 游戏 XML 或存档文件路径 | `/path/to/ww2pac40_2nd_edition.xml` |
| `triplea.start` | 启动后直接进入的界面 | `local` / `lobby` / `pbf` / `pbem` |
| `triplea.server` | 是否以服务器模式启动 | `true` |
| `triplea.client` | 是否以客户端模式连接 | `true` |
| `triplea.port` | 服务器端口（服务器模式） | `3300` |
| `triplea.name` | 玩家/主机名 | 任意字符串 |

示例：

```bash
./gradlew :game-app:game-headed:run --args="triplea.game=/path/to/game.xml triplea.start=local"
```

---

### 6. 其他常用 Gradle 指令（与“运行游戏”相关）

| 用途 | 指令 |
|------|------|
| 全量构建与检查 | `./verify` |
| 格式化代码 | `./gradlew spotlessApply` |
| 运行所有测试 | `./gradlew test` |
| 运行某子项目测试 | `./gradlew :game-app:game-core:test` |
| 运行单个测试类 | `./gradlew :game-app:game-core:test --tests games.strategy.triplea.UnitUtilsTest` |
| 强制重新跑任务 | `./gradlew --rerun-tasks :game-app:game-core:test` |

---

## 一、环境与编译

- **JDK**：需要 **JDK 21**（见 [README](../README.md) 与 [build-overview](development/build-overview-and-development.md)）。
- **首次构建**（可选，用于下载资源）：
  ```bash
  ./gradlew downloadAssets
  ```

---

## 二、运行游戏的方式

### 1. GUI 客户端（推荐用于“选地图 + Local Game + 全 PRO_AI”）

```bash
./gradlew :game-app:game-headed:run
```

启动后是主界面（选择 Play Online / Start Local Game 等）。

### 2. Headless（无界面，用于 bot / 自动化）

需要指定地图目录，例如：

```bash
MAPS_FOLDER=/path/to/downloadedMaps ./gradlew :game-app:game-headless:run
```

详见 [game-headless/README.md](../game-app/game-headless/README.md)。  
**说明**：干净仓库下 headless 默认不会自动加载 Pacific 2nd，需配合地图路径和（若后续实现）游戏 XML 等配置使用。

---

## 三、Pacific 2nd version 地图

- **规则集名称**：**World War II Pacific 1940 2nd Edition**（见 [game-rule-sets.md](development/game-rule-sets.md)）。
- **游戏 XML**：在官方地图仓库中为  
  `world_war_ii_pacific` → `map/games/ww2pac40_2nd_edition.xml`。

本仓库**不包含**地图文件，地图通常来自：

1. **游戏内下载**：GUI 启动后通过 Map Download / 地图管理 下载 “World War II Pacific 1940 2nd Edition”。
2. **克隆 triplea-maps**：从 [triplea-maps](https://github.com/triplea-maps) 克隆 `world_war_ii_pacific` 等仓库，放到 TripleA 使用的**地图目录**（例如用户目录下的 `triplea/downloadedMaps` 或通过 `MAPS_FOLDER` 指定的目录），使游戏能扫描到该地图。

---

## 四、GUI 下：加载 Pacific 2nd + 全部 PRO_AI 并运行

与“和 GUI 一致”的流程一致：**选地图 → Local Game → 全部国家设为 PRO_AI → 开始游戏**。

### 步骤 1：启动 GUI

```bash
./gradlew :game-app:game-headed:run
```

### 步骤 2：进入本地游戏设置

- 在主界面点击 **Play**。
- 选择 **Start Local Game**（开始本地游戏）。

### 步骤 3：选择 Pacific 2nd 地图

- 在游戏选择区域（左侧或上方）选择 **World War II Pacific 1940 2nd Edition**。  
- 若列表中没有，请先通过“地图下载”或把 `world_war_ii_pacific` 放到地图目录后再启动。

### 步骤 4：全部国家设为 PRO_AI（Hard AI）

- 在 **Select Local Players and AI's**（选择本地玩家与 AI）区域：
  - 每个国家有一列 **Type**（类型），将每个都选为 **Hard (AI)**，即引擎中的 **PRO_AI**；或
  - 使用 **Set All To** 下拉框，选择 **Hard (AI)**，一次性把所有国家设为 PRO_AI。

### 步骤 5：开始对局

- 点击 **Play** 开始游戏。  
- 对局中可通过菜单 **Save Game** 保存为 `.tsvg` 等格式。

---

## 五、通过命令行预加载地图并打开本地设置（可选）

若已知道游戏 XML 的**绝对路径**，可在启动时传入系统属性，让 GUI 直接加载该地图并打开本地设置界面，然后再在界面中把全部国家设为 Hard (AI) 并点 Play。

```bash
./gradlew :game-app:game-headed:run --args="triplea.game=/绝对路径/map/games/ww2pac40_2nd_edition.xml triplea.start=local"
```

例如地图目录为 `/Users/you/triplea/downloadedMaps/world_war_ii_pacific` 时：

```bash
./gradlew :game-app:game-headed:run --args="triplea.game=/Users/you/triplea/downloadedMaps/world_war_ii_pacific/map/games/ww2pac40_2nd_edition.xml triplea.start=local"
```

- **triplea.game**：游戏 XML 的完整路径。  
- **triplea.start=local**：启动后直接进入“本地游戏”设置界面（见 `HeadedGameRunner` 中的 `TRIPLEA_START`、`TRIPLEA_START_LOCAL`）。  

**注意**：当前代码**没有**通过命令行参数“默认全部 PRO_AI”的入口，仍需在本地设置界面中手动将“Set All To: Hard (AI)”或逐国选择 Hard (AI)，再点击 Play。

---

## 六、参考源码位置

| 内容           | 位置 |
|----------------|------|
| GUI 启动与参数 | `game-app/game-headed/.../HeadedGameRunner.java`（main、initializeDesktopIntegrations、showMainFrame、startGameDirectly） |
| 本地游戏入口   | `HeadedServerSetupModel.showLocal()`，`MetaSetupPanel` 中 “Start Local Game” 按钮 |
| 玩家类型选择   | `SetupPanel`、`PlayerSelectorRow`；“Hard (AI)” 对应 `PlayerTypes.PRO_AI`，i18n：`startup.PlayerTypes.PLAYER_TYPE_AI_HARD_LABEL` |
| 规则集与 Pacific 2nd | `docs/development/game-rule-sets.md` |
| Headless 运行  | `game-app/game-headless/README.md`，`HeadlessGameRunner` |

---

## 七、简要检查清单

- [ ] 已安装 JDK 21，`./gradlew :game-app:game-headed:run` 能正常启动 GUI。  
- [ ] 已通过下载或克隆地图，使 **World War II Pacific 1940 2nd Edition** 出现在游戏列表中。  
- [ ] 已选择 **Start Local Game**，并在 **Select Local Players and AI's** 中将 **Set All To: Hard (AI)**（或逐国选 Hard (AI)）。  
- [ ] 点击 **Play** 开始对局；需要时可使用 **Save Game** 保存为 .tsvg。

若需要**无界面、指定回合数后自动保存 .tsvg** 的流程，需要在当前 headless 或本地游戏流程上增加相应逻辑（例如在干净代码上重新实现“本地游戏 + 指定回合后保存”），可在此基础上再扩展。

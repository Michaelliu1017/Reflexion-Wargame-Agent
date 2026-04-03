# 将项目打包并发布到 GitHub

本文说明如何把 **TripleA-LLM**（含 Game Bridge 与 Python 驱动）放到自己的 GitHub 仓库，并让其他人能按 README 部署。

## 一、在 GitHub 上创建仓库

1. 登录 [GitHub](https://github.com)，点击 **New repository**。
2. 填写 **Repository name**，建议：**`triplea-llm`**（与项目名一致）。
3. 选择 **Public**，可选 **Add a README**（若本地已有可不必勾选）。
4. 创建后记下仓库地址，例如：`https://github.com/你的用户名/triplea-llm.git`。

## 二、本地初始化为 Git 并推送到 GitHub

在**项目根目录**（含 `build.gradle`、`clients/`、`game-app/` 的目录）执行：

```bash
# 若尚未初始化
git init

# 添加所有文件（.gitignore 会排除 build、.gradle、.venv 等）
git add .
git status   # 确认要提交的内容

# 首次提交
git commit -m "Add TripleA with GPT Bridge and Python driver"

# 添加远程仓库（替换为你的用户名与仓库名）
git remote add origin https://github.com/你的用户名/triplea-llm.git

# 推送到 main（若默认分支是 master 则写 master）
git branch -M main
git push -u origin main
```

若仓库已存在且已有提交历史，可先拉再推：

```bash
git remote add origin https://github.com/你的用户名/triplea-llm.git
git pull origin main --rebase
git push -u origin main
```

## 三、建议不要提交的内容（.gitignore）

项目根目录下的 `.gitignore` 已包含常见忽略项，例如：

- `build/`、`.gradle/`、`out/`
- `.idea/`、`*.iml`
- `.venv/`（Python 虚拟环境建议忽略）

若 `clients/python/.venv` 未被忽略，可在项目根或 `clients/python/` 的 `.gitignore` 中增加一行：

```
clients/python/.venv/
```

**不要**把 `OPENAI_API_KEY` 或任何密钥写进代码或配置文件并提交。

## 四、README 与部署说明

- **仓库首页**：根目录 [README.md](../README.md) 已包含「GPT-Driven Japan」一节，并链接到 [clients/python/README.md](../clients/python/README.md)。
- **用户部署**：用户克隆后按 [clients/python/README.md](../clients/python/README.md) 的「部署步骤」执行即可（Host → Bridge → GPT Driver）。
- 更细的实现与跨机部署见 [clients/python/项目实现与部署说明.md](../clients/python/项目实现与部署说明.md)。

## 五、后续更新流程

```bash
git add .
git commit -m "描述本次修改"
git push origin main
```

用户侧更新：

```bash
git pull origin main
```

然后按需重新编译或重启 Bridge / Driver。

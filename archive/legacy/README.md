# Legacy archive（历史提交与文档）

`submission_v6.py` 至 `submission_v18.py` 以及部分与当前主版本无关的说明文档，集中放在此目录，便于将 **`kaggle/` 工作区默认收敛到 v19**（`submission_v19.py`），同时保留可复现的旧版与参考资料。

## 布局

| 目录 / 文件 | 说明 |
|-------------|------|
| `submissions/` | 历史 `submission_*.py` 与 `submission.py` |
| `docs/` | `AGENTS.md`（版本演进正文）、`ONBOARDING.md`、`CLAUDE.md`、规则与起步 txt 等 |
| `from_repo_root/` | 从仓库**外层**根目录迁入的备忘 txt（原与 v19 开发无强绑定） |
| `dist_standalone_v7/` | 历史随 v7 的独立打包目录示例 |

## 在脚本里加载旧版

不要手写死路径。请在 `kaggle/` 下使用 `submission_resolve.resolve_submission_path` 或 `load_submission_module`（会先查当前目录，再查 `archive/legacy/submissions/`）。

## 版本总览

仓库根目录 **[VERSIONS.md](../../../VERSIONS.md)**（相对本文件向上三级到仓库根）列出各版变更要点，并指向本目录下的 `docs/AGENTS.md` 详解。

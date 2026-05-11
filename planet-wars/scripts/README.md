# `planet-wars/scripts/`

| 脚本 | 作用 |
|------|------|
| [`gen_lisp_symbol_docs.py`](gen_lisp_symbol_docs.py) | 递归扫描 `planet-wars/**/*.lisp`（跳过 `src/usocket-0.4.1/test/`），提取顶层 **`defun` / `defmacro` / `defgeneric` / `defmethod` / `defclass`**，把 docstring（若有）写入 Markdown；输出 [`../docs/symbols/`](../docs/symbols/README.md)。 |

执行（在仓库任意目录）：

```bash
python3 planet-wars/scripts/gen_lisp_symbol_docs.py
```

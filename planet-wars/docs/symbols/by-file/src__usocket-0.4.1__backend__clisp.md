# `src/usocket-0.4.1/backend/clisp.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 21 | `DEFUN` | `get-host-name` | （未附带 docstring，见该行附近源码。） |
| 29 | `DEFUN` | `remap-maybe-for-win32` | （未附带 docstring，见该行附近源码。） |
| 37 | `DEFPARAMETER` | `+clisp-error-map+` | 初值摘要: `#+win32   (append (remap-maybe-for-win32 +unix-errno-condition-map+)           (remap-maybe-for-win32 +unix-errno-error-...` |
| 45 | `DEFUN` | `handle-condition` | （未附带 docstring，见该行附近源码。） |
| 58 | `DEFUN` | `socket-connect` | （未附带 docstring，见该行附近源码。） |
| 83 | `DEFUN` | `socket-listen` | （未附带 docstring，见该行附近源码。） |
| 99 | `DEFMETHOD` | `socket-accept` | （未附带 docstring，见该行附近源码。） |
| 111 | `DEFMETHOD` | `socket-close` | （未附带 docstring，见该行附近源码。） |
| 118 | `DEFMETHOD` | `socket-close` | （未附带 docstring，见该行附近源码。） |
| 123 | `DEFMETHOD` | `get-local-name` | （未附带 docstring，见该行附近源码。） |
| 129 | `DEFMETHOD` | `get-peer-name` | （未附带 docstring，见该行附近源码。） |
| 135 | `DEFMETHOD` | `get-local-address` | （未附带 docstring，见该行附近源码。） |
| 138 | `DEFMETHOD` | `get-peer-address` | （未附带 docstring，见该行附近源码。） |
| 141 | `DEFMETHOD` | `get-local-port` | （未附带 docstring，见该行附近源码。） |
| 144 | `DEFMETHOD` | `get-peer-port` | （未附带 docstring，见该行附近源码。） |
| 148 | `DEFUN` | `%setup-wait-list` | （未附带 docstring，见该行附近源码。） |
| 151 | `DEFUN` | `%add-waiter` | （未附带 docstring，见该行附近源码。） |
| 154 | `DEFUN` | `%remove-waiter` | （未附带 docstring，见该行附近源码。） |
| 158 | `DEFMETHOD` | `wait-for-input-internal` | （未附带 docstring，见该行附近源码。） |

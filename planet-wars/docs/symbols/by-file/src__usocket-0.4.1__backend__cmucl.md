# `src/usocket-0.4.1/backend/cmucl.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 9 | `DEFUN` | `remap-for-win32` | （未附带 docstring，见该行附近源码。） |
| 17 | `DEFPARAMETER` | `+cmucl-error-map+` | 初值摘要: `#+win32   (append (remap-for-win32 +unix-errno-condition-map+)           (remap-for-win32 +unix-errno-error-map+))   #-w...` |
| 25 | `DEFUN` | `cmucl-map-socket-error` | （未附带 docstring，见该行附近源码。） |
| 46 | `DEFUN` | `handle-condition` | （未附带 docstring，见该行附近源码。） |
| 53 | `DEFUN` | `socket-connect` | （未附带 docstring，见该行附近源码。） |
| 88 | `DEFUN` | `socket-listen` | （未附带 docstring，见该行附近源码。） |
| 105 | `DEFMETHOD` | `socket-accept` | （未附带 docstring，见该行附近源码。） |
| 117 | `DEFMETHOD` | `socket-close` | （未附带 docstring，见该行附近源码。） |
| 124 | `DEFMETHOD` | `socket-close` | （未附带 docstring，见该行附近源码。） |
| 131 | `DEFMETHOD` | `get-local-name` | （未附带 docstring，见该行附近源码。） |
| 137 | `DEFMETHOD` | `get-peer-name` | （未附带 docstring，见该行附近源码。） |
| 143 | `DEFMETHOD` | `get-local-address` | （未附带 docstring，见该行附近源码。） |
| 146 | `DEFMETHOD` | `get-peer-address` | （未附带 docstring，见该行附近源码。） |
| 149 | `DEFMETHOD` | `get-local-port` | （未附带 docstring，见该行附近源码。） |
| 152 | `DEFMETHOD` | `get-peer-port` | （未附带 docstring，见该行附近源码。） |
| 156 | `DEFUN` | `lookup-host-entry` | （未附带 docstring，见该行附近源码。） |
| 174 | `DEFUN` | `get-host-by-address` | （未附带 docstring，见该行附近源码。） |
| 179 | `DEFUN` | `get-hosts-by-name` | （未附带 docstring，见该行附近源码。） |
| 185 | `DEFUN` | `get-host-name` | （未附带 docstring，见该行附近源码。） |
| 188 | `DEFUN` | `%setup-wait-list` | （未附带 docstring，见该行附近源码。） |
| 191 | `DEFUN` | `%add-waiter` | （未附带 docstring，见该行附近源码。） |
| 194 | `DEFUN` | `%remove-waiter` | （未附带 docstring，见该行附近源码。） |
| 198 | `DEFUN` | `wait-for-input-internal` | （未附带 docstring，见该行附近源码。） |

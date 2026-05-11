# `src/usocket-0.4.1/usocket.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 8 | `DEFPARAMETER` | `*wildcard-host*` | 初值摘要: `#(0 0 0 0)   "Hostname to pass when all interfaces in the current system are to be bound."` |
| 11 | `DEFPARAMETER` | `*auto-port*` | 初值摘要: `0   "Port number to pass when an auto-assigned port number is wanted."` |
| 14 | `DEFCLASS` | `usocket` | Implementation specific socket object instance.' |
| 57 | `DEFCLASS` | `stream-usocket` | Stream instance associated with the socket. |
| 74 | `DEFCLASS` | `stream-server-usocket` | Default element type for streams created by `socket-accept'. |
| 85 | `DEFUN` | `usocket-p` | （未附带 docstring，见该行附近源码。） |
| 88 | `DEFUN` | `stream-usocket-p` | （未附带 docstring，见该行附近源码。） |
| 91 | `DEFUN` | `stream-server-usocket-p` | （未附带 docstring，见该行附近源码。） |
| 94 | `DEFUN` | `make-socket` | （未附带 docstring，见该行附近源码。） |
| 100 | `DEFUN` | `make-stream-socket` | （未附带 docstring，见该行附近源码。） |
| 115 | `DEFUN` | `make-stream-server-socket` | （未附带 docstring，见该行附近源码。） |
| 129 | `DEFGENERIC` | `socket-accept` | （未附带 docstring，见该行附近源码。） |
| 136 | `DEFGENERIC` | `socket-close` | （未附带 docstring，见该行附近源码。） |
| 139 | `DEFGENERIC` | `get-local-address` | （未附带 docstring，见该行附近源码。） |
| 142 | `DEFGENERIC` | `get-peer-address` | （未附带 docstring，见该行附近源码。） |
| 146 | `DEFGENERIC` | `get-local-port` | （未附带 docstring，见该行附近源码。） |
| 152 | `DEFGENERIC` | `get-peer-port` | （未附带 docstring，见该行附近源码。） |
| 155 | `DEFGENERIC` | `get-local-name` | （未附带 docstring，见该行附近源码。） |
| 161 | `DEFGENERIC` | `get-peer-name` | （未附带 docstring，见该行附近源码。） |
| 166 | `DEFMACRO` | `with-connected-socket` | （未附带 docstring，见该行附近源码。） |
| 180 | `DEFMACRO` | `with-client-socket` | （未附带 docstring，见该行附近源码。） |
| 191 | `DEFMACRO` | `with-server-socket` | （未附带 docstring，见该行附近源码。） |
| 200 | `DEFMACRO` | `with-socket-listener` | （未附带 docstring，见该行附近源码。） |
| 208 | `DEFSTRUCT` | `wait-list` | 槽位: %wait, waiters, map |
| 220 | `DEFUN` | `make-wait-list` | （未附带 docstring，见该行附近源码。） |
| 228 | `DEFUN` | `add-waiter` | （未附带 docstring，见该行附近源码。） |
| 234 | `DEFUN` | `remove-waiter` | （未附带 docstring，见该行附近源码。） |
| 241 | `DEFUN` | `remove-all-waiters` | （未附带 docstring，见该行附近源码。） |
| 248 | `DEFUN` | `wait-for-input` | （未附带 docstring，见该行附近源码。） |
| 296 | `DEFUN` | `integer-to-octet-buffer` | （未附带 docstring，见该行附近源码。） |
| 304 | `DEFUN` | `octet-buffer-to-integer` | （未附带 docstring，见该行附近源码。） |
| 315 | `DEFMACRO` | `port-to-octet-buffer` | （未附带 docstring，见该行附近源码。） |
| 318 | `DEFMACRO` | `ip-to-octet-buffer` | （未附带 docstring，见该行附近源码。） |
| 321 | `DEFMACRO` | `port-from-octet-buffer` | （未附带 docstring，见该行附近源码。） |
| 324 | `DEFMACRO` | `ip-from-octet-buffer` | （未附带 docstring，见该行附近源码。） |
| 331 | `DEFUN` | `list-of-strings-to-integers` | （未附带 docstring，见该行附近源码。） |
| 339 | `DEFUN` | `hbo-to-dotted-quad` | （未附带 docstring，见该行附近源码。） |
| 347 | `DEFUN` | `hbo-to-vector-quad` | （未附带 docstring，见该行附近源码。） |
| 355 | `DEFUN` | `vector-quad-to-dotted-quad` | （未附带 docstring，见该行附近源码。） |
| 362 | `DEFUN` | `dotted-quad-to-vector-quad` | （未附带 docstring，见该行附近源码。） |
| 366 | `DEFGENERIC` | `host-byte-order` | （未附带 docstring，见该行附近源码。） |
| 367 | `DEFMETHOD` | `host-byte-order` | （未附带 docstring，见该行附近源码。） |
| 374 | `DEFMETHOD` | `host-byte-order` | （未附带 docstring，见该行附近源码。） |
| 380 | `DEFMETHOD` | `host-byte-order` | （未附带 docstring，见该行附近源码。） |
| 383 | `DEFUN` | `host-to-hostname` | （未附带 docstring，见该行附近源码。） |
| 392 | `DEFUN` | `ip=` | （未附带 docstring，见该行附近源码。） |
| 404 | `DEFUN` | `ip/=` | （未附带 docstring，见该行附近源码。） |
| 453 | `DEFUN` | `split-timeout` | （未附带 docstring，见该行附近源码。） |

# `src/usocket-0.4.1/backend/armedbear.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 26 | `DEFSTRUCT` | `java-object-proxy` | 槽位: value, class |
| 31 | `DEFVAR` | `*jm-get-return-type*` | 初值摘要: `(java:jmethod "java.lang.reflect.Method" "getReturnType")` |
| 34 | `DEFVAR` | `*jf-get-type*` | 初值摘要: `(java:jmethod "java.lang.reflect.Field" "getType")` |
| 37 | `DEFVAR` | `*jc-get-declaring-class*` | 初值摘要: `(java:jmethod "java.lang.reflect.Constructor" "getDeclaringClass")` |
| 40 | `DECLAIM` | `—` | 编译期声明: `declaim (inline make-return-type-proxy)` |
| 41 | `DEFUN` | `make-return-type-proxy` | （未附带 docstring，见该行附近源码。） |
| 48 | `DEFUN` | `make-field-type-proxy` | （未附带 docstring，见该行附近源码。） |
| 55 | `DEFUN` | `make-constructor-type-proxy` | （未附带 docstring，见该行附近源码。） |
| 62 | `DEFUN` | `jcoerce` | （未附带 docstring，见该行附近源码。） |
| 82 | `DEFUN` | `jtype-of` | （未附带 docstring，见该行附近源码。） |
| 91 | `DECLAIM` | `—` | 编译期声明: `declaim (inline jop-deref)` |
| 92 | `DEFUN` | `jop-deref` | （未附带 docstring，见该行附近源码。） |
| 97 | `DEFUN` | `java-value-and-class` | （未附带 docstring，见该行附近源码。） |
| 101 | `DEFUN` | `do-jmethod-call` | （未附带 docstring，见该行附近源码。） |
| 111 | `DEFUN` | `do-jstatic-call` | （未附带 docstring，见该行附近源码。） |
| 118 | `DEFUN` | `do-jnew-call` | （未附带 docstring，见该行附近源码。） |
| 124 | `DEFUN` | `do-jfield` | （未附带 docstring，见该行附近源码。） |
| 138 | `DEFMACRO` | `do-jstatic` | （未附带 docstring，见该行附近源码。） |
| 141 | `DEFMACRO` | `do-jmethod` | （未附带 docstring，见该行附近源码。） |
| 146 | `DEFMACRO` | `jstatic-call` | （未附带 docstring，见该行附近源码。） |
| 154 | `DEFMACRO` | `jmethod-call` | （未附带 docstring，见该行附近源码。） |
| 168 | `DEFUN` | `jequals` | （未附带 docstring，见该行附近源码。） |
| 172 | `DEFMACRO` | `jnew-call` | （未附带 docstring，见该行附近源码。） |
| 180 | `DEFUN` | `get-host-name` | （未附带 docstring，见该行附近源码。） |
| 185 | `DEFUN` | `handle-condition` | （未附带 docstring，见该行附近源码。） |
| 189 | `DEFUN` | `socket-connect` | （未附带 docstring，见该行附近源码。） |
| 220 | `DEFUN` | `socket-listen` | （未附带 docstring，见该行附近源码。） |
| 245 | `DEFMETHOD` | `socket-accept` | （未附带 docstring，见该行附近源码。） |
| 261 | `DEFMETHOD` | `socket-close` | （未附带 docstring，见该行附近源码。） |
| 270 | `DEFMETHOD` | `socket-close` | （未附带 docstring，见该行附近源码。） |
| 276 | `DEFMETHOD` | `get-local-address` | （未附带 docstring，见该行附近源码。） |
| 282 | `DEFMETHOD` | `get-peer-address` | （未附带 docstring，见该行附近源码。） |
| 288 | `DEFMETHOD` | `get-local-port` | （未附带 docstring，见该行附近源码。） |
| 292 | `DEFMETHOD` | `get-peer-port` | （未附带 docstring，见该行附近源码。） |
| 296 | `DEFMETHOD` | `get-local-name` | （未附带 docstring，见该行附近源码。） |
| 300 | `DEFMETHOD` | `get-peer-name` | （未附带 docstring，见该行附近源码。） |
| 345 | `DEFUN` | `op-read` | （未附带 docstring，见该行附近源码。） |
| 349 | `DEFUN` | `op-accept` | （未附带 docstring，见该行附近源码。） |
| 353 | `DEFUN` | `op-connect` | （未附带 docstring，见该行附近源码。） |
| 357 | `DEFUN` | `valid-ops` | （未附带 docstring，见该行附近源码。） |
| 360 | `DEFUN` | `channel-class` | （未附带 docstring，见该行附近源码。） |
| 369 | `DEFUN` | `socket-channel-class` | （未附带 docstring，见该行附近源码。） |
| 378 | `DEFUN` | `wait-for-input-internal` | （未附带 docstring，见该行附近源码。） |
| 441 | `DEFUN` | `%setup-wait-list` | （未附带 docstring，见该行附近源码。） |
| 445 | `DEFUN` | `%add-waiter` | （未附带 docstring，见该行附近源码。） |
| 449 | `DEFUN` | `%remove-waiter` | （未附带 docstring，见该行附近源码。） |

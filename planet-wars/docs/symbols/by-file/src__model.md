# `src/model.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 34 | `DECLAIM` | `—` | 编译期声明: `declaim (inline make-count-vector)` |
| 35 | `DEFUN` | `make-count-vector` | （未附带 docstring，见该行附近源码。） |
| 38 | `DECLAIM` | `—` | 编译期声明: `declaim (inline make-player-vector)` |
| 39 | `DEFUN` | `make-player-vector` | （未附带 docstring，见该行附近源码。） |
| 43 | `DEFCLASS` | `planet` | （未附带 docstring，见该行附近源码。） |
| 71 | `DEFCLASS` | `game` | （未附带 docstring，见该行附近源码。） |
| 80 | `DEFCLASS` | `order` | （未附带 docstring，见该行附近源码。） |
| 90 | `DEFMETHOD` | `print-object` | （未附带 docstring，见该行附近源码。） |
| 99 | `DEFMETHOD` | `print-object` | （未附带 docstring，见该行附近源码。） |
| 110 | `DEFUN` | `arrival-turn` | （未附带 docstring，见该行附近源码。） |
| 113 | `DEFUN` | `order=` | （未附带 docstring，见该行附近源码。） |
| 120 | `DEFUN` | `move=` | （未附带 docstring，见该行附近源码。） |
| 124 | `DEFUN` | `current-order-p` | （未附带 docstring，见该行附近源码。） |
| 132 | `DEFMACRO` | `do-neighbours` | （未附带 docstring，见该行附近源码。） |
| 142 | `DEFMACRO` | `do-neighbours/reverse` | （未附带 docstring，见该行附近源码。） |
| 152 | `DEFUN` | `planet-id` | （未附带 docstring，见该行附近源码。） |
| 157 | `DEFUN` | `turns-to-travel*` | （未附带 docstring，见该行附近源码。） |
| 161 | `DEFUN` | `turns-to-travel` | （未附带 docstring，见该行附近源码。） |
| 164 | `DEFUN` | `count-ships-for-battle` | （未附带 docstring，见该行附近源码。） |
| 172 | `DEFUN` | `resolve-battle` | （未附带 docstring，见该行附近源码。） |
| 192 | `DECLAIM` | `—` | 编译期声明: `declaim (inline player-multiplier)` |
| 193 | `DEFUN` | `player-multiplier` | （未附带 docstring，见该行附近源码。） |
| 199 | `DECLAIM` | `—` | 编译期声明: `declaim (inline opponent)` |
| 200 | `DEFUN` | `opponent` | （未附带 docstring，见该行附近源码。） |
| 205 | `DEFPARAMETER` | `*turn-adjustment*` | 初值摘要: `0` |
| 209 | `DEFUN` | `execute-order` | （未附带 docstring，见该行附近源码。） |
| 231 | `DEFUN` | `undo-order` | （未附带 docstring，见该行附近源码。） |
| 234 | `DEFUN` | `execute-orders` | （未附带 docstring，见该行附近源码。） |
| 237 | `DEFUN` | `undo-orders` | （未附带 docstring，见该行附近源码。） |
| 240 | `DEFCLASS` | `move-and-stuff` | （未附带 docstring，见该行附近源码。） |
| 244 | `DEFVAR` | `*moves*` | 初值摘要: `—` |
| 246 | `DEFUN` | `orders-since` | （未附带 docstring，见该行附近源码。） |
| 251 | `DEFUN` | `lookup-cached-stuff` | （未附带 docstring，见该行附近源码。） |
| 261 | `DEFUN` | `set-cached-stuff` | （未附带 docstring，见该行附近源码。） |
| 264 | `DEFMACRO` | `with-orders` | （未附带 docstring，见该行附近源码。） |
| 290 | `DEFUN` | `truncate-planet` | （未附带 docstring，见该行附近源码。） |
| 304 | `DEFUN` | `truncate-game` | （未附带 docstring，见该行附近源码。） |

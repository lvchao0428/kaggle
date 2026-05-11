# `src/player.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 3 | `DEFVAR` | `*depth*` | 初值摘要: `—` |
| 4 | `DEFVAR` | `*turn*` | 初值摘要: `—` |
| 5 | `DECLAIM` | `—` | 编译期声明: `declaim (type (unsigned-byte 16) *depth* *turn*)` |
| 11 | `DEFVAR` | `*best-move-so-far*` | 初值摘要: `—` |
| 13 | `DEFUN` | `register-evaluated-move` | （未附带 docstring，见该行附近源码。） |
| 20 | `DEFMACRO` | `with-best-move-on-timeout` | （未附带 docstring，见该行附近源码。） |
| 30 | `DEFMACRO` | `with-best-move-on-timeout` | （未附带 docstring，见该行附近源码。） |
| 46 | `DEFCLASS` | `future` | （未附带 docstring，见该行附近源码。） |
| 58 | `DEFMETHOD` | `print-object` | （未附带 docstring，见该行附近源码。） |
| 70 | `DEFUN` | `first-owner` | （未附带 docstring，见该行附近源码。） |
| 73 | `DEFUN` | `last-owner` | （未附带 docstring，见该行附近源码。） |
| 96 | `DEFUN` | `compute-future*` | （未附带 docstring，见该行附近源码。） |
| 167 | `DEFUN` | `compute-future` | （未附带 docstring，见该行附近源码。） |
| 191 | `DEFUN` | `cumulative-surplus` | （未附带 docstring，见该行附近源码。） |
| 241 | `DEFUN` | `uncumulate-surplus!` | （未附带 docstring，见该行附近源码。） |
| 255 | `DEFUN` | `uncumulate-surplus` | （未附带 docstring，见该行附近源码。） |
| 263 | `DEFUN` | `surplus` | （未附带 docstring，见该行附近源码。） |
| 273 | `DEFCLASS` | `full-attack-future` | （未附带 docstring，见该行附近源码。） |
| 290 | `DEFUN` | `compute-full-attack-future` | （未附带 docstring，见该行附近源码。） |
| 325 | `DEFUN` | `add-surplus-into-arrivals` | （未附带 docstring，见该行附近源码。） |
| 378 | `DEFUN` | `compute-full-attack-arrivals` | （未附带 docstring，见该行附近源码。） |
| 407 | `DEFUN` | `update-full-attack-arrivals` | （未附带 docstring，见该行附近源码。） |
| 452 | `DEFUN` | `make-full-attack-arrivals` | （未附带 docstring，见该行附近源码。） |
| 503 | `DEFUN` | `score` | （未附带 docstring，见该行附近源码。） |
| 544 | `DEFUN` | `evaluate/full-attack` | （未附带 docstring，见该行附近源码。） |
| 567 | `DEFUN` | `evaluate-planet` | （未附带 docstring，见该行附近源码。） |
| 591 | `DEFUN` | `first-non-neutral-turn` | （未附带 docstring，见该行附近源码。） |
| 595 | `DEFUN` | `candidate-min-turns-to-arrive` | （未附带 docstring，见该行附近源码。） |
| 610 | `DEFUN` | `evaluate-non-neutral-planet` | （未附带 docstring，见该行附近源码。） |
| 642 | `DEFUN` | `eval*` | （未附带 docstring，见该行附近源码。） |
| 681 | `DEFUN` | `compute-step-target` | （未附带 docstring，见该行附近源码。） |
| 704 | `DEFUN` | `arrivals-of-player` | （未附带 docstring，见该行附近源码。） |
| 711 | `DEFUN` | `find-first-ownership-change` | （未附带 docstring，见该行附近源码。） |
| 729 | `DEFUN` | `find-first-possible-takeover-opportunity` | （未附带 docstring，见该行附近源码。） |
| 771 | `DEFUN` | `maybe-take-over-and-defend` | （未附带 docstring，见该行附近源码。） |
| 832 | `DEFUN` | `find-neutral-steal` | （未附带 docstring，见该行附近源码。） |
| 843 | `DEFUN` | `find-step` | （未附带 docstring，见该行附近源码。） |
| 881 | `DEFUN` | `find-steps` | （未附带 docstring，见该行附近源码。） |
| 890 | `DEFUN` | `takeablep` | （未附带 docstring，见该行附近源码。） |
| 893 | `DEFUN` | `generate-candidate-steps` | （未附带 docstring，见该行附近源码。） |
| 961 | `DEFUN` | `planets-involved-in-move` | （未附带 docstring，见该行附近源码。） |
| 968 | `DEFUN` | `valid-move-p` | （未附带 docstring，见该行附近源码。） |
| 977 | `DEFUN` | `generate-moves-from-steps` | （未附带 docstring，见该行附近源码。） |
| 993 | `DEFUN` | `score-and-sort-moves` | （未附带 docstring，见该行附近源码。） |
| 1004 | `DEFUN` | `generate-and-score-moves` | （未附带 docstring，见该行附近源码。） |
| 1025 | `DEFUN` | `safety-margin` | （未附带 docstring，见该行附近源码。） |
| 1052 | `DEFUN` | `safety-margin!` | （未附带 docstring，见该行附近源码。） |
| 1069 | `DEFUN` | `n-turns-to-break-even` | （未附带 docstring，见该行附近源码。） |
| 1079 | `DEFUN` | `safe-to-invest-p` | （未附带 docstring，见该行附近源码。） |
| 1092 | `DEFUN` | `horizon` | （未附带 docstring，见该行附近源码。） |
| 1114 | `DEFCLASS` | `bocsimacko` | （未附带 docstring，见该行附近源码。） |
| 1121 | `DEFMACRO` | `with-game` | （未附带 docstring，见该行附近源码。） |
| 1150 | `DEFMETHOD` | `compute-orders` | （未附带 docstring，见该行附近源码。） |
| 1175 | `DEFMETHOD` | `compute-orders` | （未附带 docstring，见该行附近源码。） |

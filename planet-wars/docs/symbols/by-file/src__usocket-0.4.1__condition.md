# `src/usocket-0.4.1/condition.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 75 | `DEFMACRO` | `define-usocket-condition-classes` | （未附带 docstring，见该行附近源码。） |
| 165 | `DEFMACRO` | `with-mapped-conditions` | （未附带 docstring，见该行附近源码。） |
| 169 | `DEFPARAMETER` | `+unix-errno-condition-map+` | 初值摘要: ``(((11) . retry-condition) ;; EAGAIN     ((35) . retry-condition) ;; EDEADLCK     ((4) . interrupted-condition))` |
| 174 | `DEFPARAMETER` | `+unix-errno-error-map+` | 初值摘要: `;;### the first column is for non-(linux or srv4) systems   ;; the second for linux   ;; the third for srv4   ;;###FIXME...` |
| 204 | `DEFUN` | `map-errno-condition` | （未附带 docstring，见该行附近源码。） |
| 208 | `DEFUN` | `map-errno-error` | （未附带 docstring，见该行附近源码。） |
| 212 | `DEFPARAMETER` | `+unix-ns-error-map+` | 初值摘要: ``((1 . ns-host-not-found-error)     (2 . ns-try-again-condition)     (3 . ns-no-recovery-error))` |
| 219 | `DEFMACRO` | `unsupported` | （未附带 docstring，见该行附近源码。） |
| 225 | `DEFMACRO` | `unimplemented` | （未附带 docstring，见该行附近源码。） |

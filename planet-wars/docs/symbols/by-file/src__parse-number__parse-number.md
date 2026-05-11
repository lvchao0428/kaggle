# `src/parse-number/parse-number.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 53 | `DECLAIM` | `—` | 编译期声明: `declaim (inline parse-integer-and-places)` |
| 54 | `DEFUN` | `parse-integer-and-places` | （未附带 docstring，见该行附近源码。） |
| 68 | `DEFUN` | `parse-integers` | （未附带 docstring，见该行附近源码。） |
| 90 | `DECLAIM` | `—` | 编译期声明: `declaim (inline number-value places)` |
| 91 | `DEFUN` | `number-value` | （未附带 docstring，见该行附近源码。） |
| 92 | `DEFUN` | `places` | （未附带 docstring，见该行附近源码。） |
| 94 | `DECLAIM` | `—` | 编译期声明: `declaim (type cons *white-space-characters*)` |
| 95 | `DEFPARAMETER` | `*white-space-characters*` | 初值摘要: `'(#\Space #\Tab #\Return #\Linefeed)` |
| 98 | `DECLAIM` | `—` | 编译期声明: `declaim (inline white-space-p)` |
| 99 | `DEFUN` | `white-space-p` | （未附带 docstring，见该行附近源码。） |
| 113 | `DEFUN` | `parse-number` | （未附带 docstring，见该行附近源码。） |
| 157 | `DEFUN` | `parse-real-number` | （未附带 docstring，见该行附近源码。） |
| 203 | `DEFUN` | `parse-positive-real-number` | （未附带 docstring，见该行附近源码。） |
| 313 | `DEFPARAMETER` | `*test-values*` | 初值摘要: `'("1" "-1" "1034" "-364" "80/335" "3.5333" "2.4E4" "6.8d3" "#xFF" "#b-1000" "#o-101/75" "13.09s3" "35.66l5" "21.4f2" "#C...` |
| 316 | `DEFUN` | `run-tests` | （未附带 docstring，见该行附近源码。） |

# `src/timer.lisp`

由 `scripts/gen_lisp_symbol_docs.py` 自动生成。

| 行号 | 类型 | 符号 | 说明 |
|------|------|------|------|
| 15 | `DECLAIM` | `—` | 编译期声明: `declaim (inline heap-parent heap-left heap-right)` |
| 17 | `DEFUN` | `heap-parent` | （未附带 docstring，见该行附近源码。） |
| 20 | `DEFUN` | `heap-left` | （未附带 docstring，见该行附近源码。） |
| 23 | `DEFUN` | `heap-right` | （未附带 docstring，见该行附近源码。） |
| 26 | `DEFUN` | `heapify` | （未附带 docstring，见该行附近源码。） |
| 48 | `DEFUN` | `heap-insert` | （未附带 docstring，见该行附近源码。） |
| 62 | `DEFUN` | `heap-maximum` | （未附带 docstring，见该行附近源码。） |
| 66 | `DEFUN` | `heap-extract` | （未附带 docstring，见该行附近源码。） |
| 75 | `DEFUN` | `heap-extract-maximum` | （未附带 docstring，见该行附近源码。） |
| 80 | `DEFSTRUCT` | `priority-queue` | 槽位: contents, keyfun |
| 86 | `DEFUN` | `make-priority-queue` | （未附带 docstring，见该行附近源码。） |
| 94 | `DEFMETHOD` | `print-object` | （未附带 docstring，见该行附近源码。） |
| 99 | `DEFUN` | `priority-queue-maximum` | （未附带 docstring，见该行附近源码。） |
| 105 | `DEFUN` | `priority-queue-extract-maximum` | （未附带 docstring，见该行附近源码。） |
| 112 | `DEFUN` | `priority-queue-insert` | （未附带 docstring，见该行附近源码。） |
| 118 | `DEFUN` | `priority-queue-empty-p` | （未附带 docstring，见该行附近源码。） |
| 121 | `DEFUN` | `priority-queue-remove` | （未附带 docstring，见该行附近源码。） |
| 132 | `DEFSTRUCT` | `timer` | 槽位: name, function, expire-time, repeat-interval, thread, interrupt-function, cancel-function |
| 145 | `DEFMETHOD` | `print-object` | （未附带 docstring，见该行附近源码。） |
| 155 | `DEFUN` | `make-timer` | （未附带 docstring，见该行附近源码。） |
| 166 | `DEFUN` | `timer-name` | （未附带 docstring，见该行附近源码。） |
| 170 | `DEFUN` | `timer-scheduled-p` | （未附带 docstring，见该行附近源码。） |
| 182 | `DEFVAR` | `*scheduler-lock*` | 初值摘要: `(sb-thread:make-mutex :name "Scheduler lock")` |
| 184 | `DEFMACRO` | `with-scheduler-lock` | （未附带 docstring，见该行附近源码。） |
| 193 | `DEFPARAMETER` | `*schedule*` | 初值摘要: `(make-priority-queue :key #'%timer-expire-time)` |
| 195 | `DEFUN` | `peek-schedule` | （未附带 docstring，见该行附近源码。） |
| 198 | `DEFUN` | `time-left` | （未附带 docstring，见该行附近源码。） |
| 203 | `DEFUN` | `delta->real` | （未附带 docstring，见该行附近源码。） |
| 208 | `DEFUN` | `make-cancellable-interruptor` | （未附带 docstring，见该行附近源码。） |
| 232 | `DEFUN` | `%schedule-timer` | （未附带 docstring，见该行附近源码。） |
| 250 | `DEFUN` | `schedule-timer` | （未附带 docstring，见该行附近源码。） |
| 271 | `DEFUN` | `unschedule-timer` | （未附带 docstring，见该行附近源码。） |
| 288 | `DEFUN` | `list-all-timers` | （未附带 docstring，见该行附近源码。） |
| 295 | `DEFUN` | `reschedule-timer` | （未附带 docstring，见该行附近源码。） |
| 310 | `DEFUN` | `real-time->sec-and-usec` | （未附带 docstring，见该行附近源码。） |
| 323 | `DEFUN` | `set-system-timer` | （未附带 docstring，见该行附近源码。） |
| 335 | `DEFUN` | `run-timer` | （未附带 docstring，见该行附近源码。） |
| 353 | `DEFUN` | `run-expired-timers` | （未附带 docstring，见该行附近源码。） |
| 370 | `DEFUN` | `timeout-cerror` | （未附带 docstring，见该行附近源码。） |
| 373 | `DEFMACRO` | `sb-ext:with-timeout` | （未附带 docstring，见该行附近源码。） |

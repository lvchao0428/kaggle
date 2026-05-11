# `timer.lisp` — 逐符号中文说明（`src/timer.lisp`）

> 文件以 `(in-package "SB-IMPL")` 实现；**本质是较新 SBCL 中计时/超时逻辑的 backport**。与 [`player.lisp`](../../../src/player.lisp) 对齐：SBCL 用 `sb-ext:with-timeout`（约在 **L31–38**），Allegro 用 `mp:with-timeout`（约在 **L19–26**）。

自动符号表：[src__timer.md](../../by-file/src__timer.md)。

---

## 上下文与兼容性

源码头注释：**官方服务器_sbcl 1.0.11 上直接用 WITH-TIMEOUT 会触发「interrupt nesting depth exceeded」**；此处将 **1.0.40 时代**的实现搬回老环境。载入时 **`sb-ext:unlock-package`** 解开 `SB-IMPL`/`SB-EXT` 便于定义宏。

---

## 堆（二叉堆）与 `DECLAIM`

| 符号 | 行号 | 说明 |
|------|------|------|
| `(declaim (inline heap-parent heap-left heap-right))` | **L15** | 编译期内联三个下标辅助函数。 |
| `heap-parent` | **L17** | 堆父结点下标：`(ash (1- i) -1)`。 |
| `heap-left` / `heap-right` | **L20** / **L23** | 左右子结点下标。 |
| `heapify` | **L26** | 在向量 `heap` 上从 `start` 下沉，键 `key`，比较 `test`（默认 `>=`，即大根堆语义配合 extract-max）。 |
| `heap-insert` | **L48** | 尾部插入新元素后沿父链上浮；返回插入位置下标。 |
| `heap-maximum` | **L62** | 返回根（下标 0），空堆返回 `nil`。 |
| `heap-extract` | **L66** | 取出下标 `i` 处元素，用末尾填洞后 `heapify`。 |
| `heap-extract-maximum` | **L75** | 提取根。 |

---

## `priority-queue` 结构体

| 槽位 | 说明 |
|------|------|
| `contents` | 可调整大小的向量，物理上存堆。 |
| `keyfun` | 取优先键（定时器场景中按 `%timer-expire-time`）。 |

| 符号 | 行号 | 说明 |
|------|------|------|
| `%make-priority-queue`（`:constructor`） | **L80** | 内部构造器；前缀 `%pqueue-` **conc-name**。 |
| `make-priority-queue` | **L86** | 封装：创建 `contents` 初值向量 + 绑定 `keyfun`。 |
| `print-object`（`priority-queue`） | **L94** | 打印队列是否空及条目数。 |
| `priority-queue-maximum` | **L99** | 堆顶元素（不落出）。 |
| `priority-queue-extract-maximum` | **L105** | 弹出最大键元素。 |
| `priority-queue-insert` | **L112** | 插入；测试 `<=` 与最大堆配合。 |
| `priority-queue-empty-p` | **L118** | 长度为 0。 |
| `priority-queue-remove` | **L121** | 按 `test`（默认 `eq`）找条目并 `heap-extract` 该下标。 |

---

## `timer` 结构体

| 槽位 | 说明 |
|------|------|
| `name` | 调试用的名字（可选）。 |
| `function` | 计时到期要执行的 Lisp 函数。 |
| `expire-time` | 到期时刻（内部实时时钟单位，`get-internal-real-time` 同基底）。 |
| `repeat-interval` | 若为真，超时回调里会推进下一拍并重新入队（周期定时）。 |
| `thread` | 向哪条线程发 **interrupt**；`t` 表示新起线程跑 `function`；默认可绑当前线程。 |
| `interrupt-function` | 实际在线程上触发的函数（可由 `make-cancellable-interruptor` 包装成可取消）。 |
| `cancel-function` | 配套取消逻辑。 |

| 符号 | 行号 | 说明 |
|------|------|------|
| `print-object`（`timer`） | **L145** | 可读打印，尽量显示 `name`。 |
| `make-timer` | **L155** | 用户 API：创建 timer，设置 `function`、可选 `name`/`thread`。 |
| `timer-name` | **L166** | 访问器。 |
| `timer-scheduled-p` | **L170** | 是否已在 `*schedule*` 优先队列中。 |

---

## 调度器全局状态

| 符号 | 行号 | 说明 |
|------|------|------|
| `*scheduler-lock*` | **L182** | SBCL 互斥锁，保护调度队列与系统定时器状态，避免与信号处理交错。 |
| `with-scheduler-lock` | **L184** | 宏：在 `*scheduler-lock*` 下执行 body（注释：SIGALRM 处理器不能破坏临界区）。 |
| `*schedule*` | **L193** | **全局优先队列**，键为 `%timer-expire-time`，最早到期的 timer 在堆顶。 |
| `#+nil (defun under-scheduler-lock-p ...)` | **L189** | 死代码分支；生成器已跳过，不进入索引。 |

---

## 时间与系统定时器

| 符号 | 行号 | 说明 |
|------|------|------|
| `peek-schedule` | **L195** | 看堆顶 timer，不弹出。 |
| `time-left` | **L198** | 某 timer 距离到期还剩多少内部时间单位。 |
| `delta->real` | **L203** | 内部时间单位 → 秒（floor 到秒）。 |
| `make-cancellable-interruptor` | **L208** | 返回一对函数：一个执行原 `function`，另一个置「已取消」使之后调用无效。 |
| `%schedule-timer` | **L232** | 加锁把 timer 插入 `*schedule*` 并 `set-system-timer`。 |
| `schedule-timer` | **L250** | 计算 `expire-time` 后调用 `%schedule-timer`。 |
| `unschedule-timer` | **L271** | 从队列移除；可能需重设 `setitimer`。 |
| `list-all-timers` | **L288** | 调试用：列出队列中所有 timer。 |
| `reschedule-timer` | **L295** | 改重复定时间隔等并重新排队。 |
| `real-time->sec-and-usec` | **L310** | 把内部时间差拆成秒/微秒；**故意留 0.0001s 余量**防饿死。 |
| `set-system-timer` | **L323** | 调 `sb-unix:unix-setitimer` 设 **下一个**真实定时；无 timer 则清零。 |
| `run-timer` | **L335** | 执行到期逻辑：`thread` 为 `t` 时新线程；否则 `interrupt-thread` 打断目标线程。 |
| `run-expired-timers` | **L353** | **信号处理器路径**：加锁取堆顶，若时间未到则可能是误报 SIGALRM，仍 `set-system-timer` 后返回；否则弹出并 `run-timer`。 |
| `timeout-cerror` | **L370** | 包装 `sb-ext::timeout` 为可 `cerror` 继续的条件。 |
| `sb-ext:with-timeout` | **L373** | `expires>0` 时建 timer、调度、`unwind-protect` 里跑 body，结束取消 timer；**L394–405 的 `#+nil` 旧实现**已被跳过不索引。 |

---

## 与 `compute-orders` 的衔接（概念）

`player.lisp` 在 SBCL 下用 **`handler-case` + `sb-ext:with-timeout`** 包住整段思考；到期时返回已登记的 **best move**。本文件的 **SIGALRM / setitimer + 优先队列** 保证墙钟到期能异步打断思考线程，与「始终有合法着法可交」一致。

---

## 参考

- 自动生成表：[src__timer.md](../../by-file/src__timer.md)
- 模块综述：[CHAMPION_MODULES_zh.md](../CHAMPION_MODULES_zh.md)

# `planet-wars/src/proxy-bot/`

通过网络 **TCP** 与其它进程或远程测试服对战时使用的 **代理 Bot**：本地仍是 Lisp Bot，stdin/stdout 与代理之间按协议转发。

| 文件 | 作用 |
|------|------|
| [`package.lisp`](package.lisp) | 独立包名声明。 |
| [`proxy-bot.lisp`](proxy-bot.lisp) | 代理主逻辑：套接字连接、`usocket` 读写、协议格式转换。 |

CLI 封装见 [`../../bin/run-proxy-bot.sh`](../../bin/run-proxy-bot.sh)。

**符号表**：[package](../../docs/symbols/by-file/src__proxy-bot__package.md) · [proxy-bot](../../docs/symbols/by-file/src__proxy-bot__proxy-bot.md)

;;; =============================================================================
;;; 中文注释副本 — 对应 ../../src/package.lisp
;;; 仅供阅读；竞赛与 ASDF 载入仍用原版。仅增加注释行，未改代码。
;;; =============================================================================

;;; 定义 planet-wars 包：昵称 :pw :pwbot；USE 标准 CL、（SBCL 下）sb-bsd-sockets、pw-util。
;;; 仅导出 PLAY 与 START-SERVER-FOR-PROXY-BOT，供脚本与引擎侧入口调用。
(defpackage :planet-wars
  (:nicknames :pw :pwbot)
  (:use :cl #+sbcl :sb-bsd-sockets :pw-util)
  (:export #:play
           #:start-server-for-proxy-bot))

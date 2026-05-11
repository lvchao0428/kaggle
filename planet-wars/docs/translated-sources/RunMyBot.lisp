;;; =============================================================================
;;; 中文注释副本 — 对应 ../../RunMyBot.lisp
;;; 本地快速跑局：stdout 重定向到 stderr，避免引擎误读；末尾直接 (PW:PLAY)。
;;; =============================================================================

;;;; This file can also be loaded to test a bot without compiling and
;;;; saving an image or using proxy bot. Used by bin/run-bot.sh.

;;; Load the sytem, but make sure nothing is written to the orignal
;;; stdout as that's read by the engine.
(let ((*standard-output* *error-output*))
  (load (merge-pathnames "setup.lisp" *load-truename*))
  (require :planet-wars))

;;; 调用主循环：读 stdin / 写订单与 GO。
(pw:play)

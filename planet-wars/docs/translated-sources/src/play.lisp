;;; =============================================================================
;;; 中文注释副本 — 对应 ../../src/play.lisp
;;; 引擎主循环、泛型 COMPUTE-ORDERS、可选 TCP 服务（代理 Bot）。
;;; =============================================================================

(in-package :planet-wars)

;;; MyBot 导出的二进制入口：错误即退出前提下调用 PLAY。
(defun main ()
  (pw-util:with-reckless-exit
    (pw-util:with-errors-logged (:exit-on-error-p t)
      (play))))


;;; 由各 Bot 实现：读 INPUT，返回 ORDER 列表（可含未来回合挂单）。
(defgeneric compute-orders (bot input))

;;; 默认玩家为 BOCSIMACKO；循环每回合调用 COMPUTE-ORDERS，拆分 turn=0 与延后单，写(stdout) 并输出 GO。
(defun play (&key (player (make-instance 'bocsimacko))
             (input *standard-input*) (output *standard-output*))
  (pw-util:logmsg "~&~%* game started at ~A~%"
                  (pw-util:current-date-time-string))
  (loop while (peek-char nil input nil nil)
        for turn from 1 do
        (pw-util:logmsg "** turn ~A~%" turn)
        (let* ((orders (compute-orders player input))
               (orders-now (remove-if-not #'current-order-p orders))
               (orders-later (remove-if #'current-order-p orders)))
          (pw-util:logmsg "*** orders~%~S~%" orders-now)
          (when orders-later
            (pw-util:logmsg "*** orders later~%~S~%" orders-later))
          (write-orders orders-now output))
        (write-line "go" output)
        (force-output output)))

;;; 本机 41807 监听；每连接一线程跑 PLAY，流作 stdin/stdout。ONE-SHOT 为真则只受理一次。
(defun start-server-for-proxy-bot (&key (player-class 'bocsimacko) one-shot)
  (let ((socket (usocket:socket-listen #+allegro "localhost"
                                       #+sbcl #(127 0 0 1)
                                       41807 :reuse-address t)))
    (unwind-protect
         (loop do
               (pw-util:logmsg "Waiting for connection...~%")
               (let* ((client (usocket:socket-accept socket))
                      (stream (usocket:socket-stream client)))
                 (pw-util:logmsg "Got connection...~%")
                 (#+sb-thread
                  sb-thread:make-thread
                  #-sb-thread
                  funcall
                  (lambda ()
                    (unwind-protect
                         (pw-util:with-errors-logged ()
                           (play :player (make-instance player-class)
                                 :input stream :output stream))
                      (ignore-errors (usocket:socket-close client))))))
               until one-shot)
      (ignore-errors (usocket:socket-close socket)))))

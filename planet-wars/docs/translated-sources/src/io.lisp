;;; =============================================================================
;;; 中文注释副本 — 对应 ../../src/io.lisp
;;; 引擎文本协议：P/F 行读写，构造 GAME 与订单写出。
;;; =============================================================================

(in-package :planet-wars)

;;; 解析一行 P：坐标、属主、兵力、增长；四向桶长 1+*N-TURNS-TILL-HORIZON*（须已绑定）。
(defun parse-planet (line)
  (let ((tokens (split-sequence:split-sequence #\space line))
        (n (1+ *n-turns-till-horizon*)))
    (assert (string= "P" (elt tokens 0)))
    (assert (<= 0 (parse-number:parse-number (elt tokens 4))))
    (make-instance 'planet
                   :x (parse-number:parse-number (elt tokens 1))
                   :y (parse-number:parse-number (elt tokens 2))
                   :owner (parse-number:parse-number (elt tokens 3))
                   :n-ships (parse-number:parse-number (elt tokens 4))
                   :growth (parse-number:parse-number (elt tokens 5))
                   :arrivals-1 (make-count-vector n)
                   :arrivals-2 (make-count-vector n)
                   :departures-1 (make-count-vector n)
                   :departures-2 (make-count-vector n))))

(defun read-planet (stream)
  (parse-planet (read-line stream)))

(defun write-planet (planet stream)
  (format stream "P ~F ~F ~D ~D ~D~%" (x planet) (y planet)
          (owner planet) (n-ships planet) (growth planet)))

;;; 解析 F 行：写入目标星 ARRIVALS-* 桶或返回「超出地平线」的船数供 GAME 累计。
(defun parse-fleet (planets line)
  (let ((tokens (split-sequence:split-sequence #\space line)))
    (assert (string= "F" (elt tokens 0)))
    (let ((owner (parse-number:parse-number (elt tokens 1)))
          (n-ships (parse-number:parse-number (elt tokens 2)))
          (destination
           (aref planets (parse-number:parse-number (elt tokens 4))))
          (n-remaining-turns (parse-number:parse-number (elt tokens 6))))
      ;; FIXME: turn this off in final build
      (let* ((source
              (aref planets (parse-number:parse-number (elt tokens 3))))
             (n-turns-to-travel (parse-number:parse-number (elt tokens 5))))
        (unless (= n-turns-to-travel (turns-to-travel source destination))
          (warn "Planet ~A distance mismatch with fleet ~S (~S vs ~S)"
                source line (turns-to-travel source destination)
                n-turns-to-travel)))
      (assert (<= 0 n-ships))
      (cond ((< *n-turns-till-horizon* n-remaining-turns)
             (ecase owner
               ((1) (values n-ships 0))
               ((2) (values 0 n-ships))))
            (t
             (ecase owner
               ((1) (incf (aref (arrivals-1 destination)
                                n-remaining-turns)
                          n-ships))
               ((2) (incf (aref (arrivals-2 destination)
                                n-remaining-turns)
                          n-ships)))
             (values 0 0))))))

;;; When called with an element TURN-FN returns the turn while KEY
;;; returns the object.

;;; 将列表按 TURN-FN 分组（相邻同 turn 合并）。
(defun group-by-turn (list &key turn-fn key)
  (let ((groups ()))
    (dolist (e list)
      (let ((last-turn (caar groups))
            (turn (funcall turn-fn e))
            (e (funcall key e)))
        (if (and last-turn (= last-turn turn))
            (setf (cdr (first groups))
                  (cons e (cdr (first groups))))
            (push (cons turn (list e)) groups))))
    (nreverse groups)))

;;; Return what will be the NEIGHBOURS of PLANET.

;;; 按航程排序邻居并分组 → 即将赋给 PLANET:NEIGHBOURS 的格式。
(defun group-planets (planet planets)
  (group-by-turn (sort (map 'list (lambda (neighbour)
                                    (cons (turns-to-travel planet neighbour)
                                          neighbour))
                            planets)
                       #'< :key #'car)
                 :turn-fn #'car :key #'cdr))

;;; 两阶段读入：先 P 建星与邻接，再 F 至 go，返回 GAME。
(defun read-game (stream)
  (let ((planets ())
        (next-planet-id 0)
        (n-ships-beyond-1 0)
        (n-ships-beyond-2 0)
        line)
    (loop do (setq line (read-line stream nil nil))
          while (and line
                     (or (zerop (length line))
                         (and (char/= #\F (aref line 0))
                              (not (string= "go" line)))))
          do
          (pw-util:logmsg "~A~%" line)
          (when (and (plusp (length line))
                     (char= #\P (aref line 0)))
            (let ((planet (parse-planet line)))
              (setf (slot-value planet 'id) next-planet-id)
              (incf next-planet-id)
              (push planet planets))))
    (setq planets (coerce (nreverse planets) 'vector))
    (loop for planet across planets
          do (setf (slot-value planet 'turns-to-neighbours)
                   (map 'vector
                        (lambda (neighbour)
                          (turns-to-travel* planet neighbour))
                        planets)))
    (loop for planet across planets
          do (setf (slot-value planet 'neighbours)
                   (group-planets planet planets)))
    (loop while (and line
                     (or (zerop (length line))
                         (not (string= "go" line))))
          do
          (pw-util:logmsg "~A~%" line)
          (when (and (plusp (length line))
                     (char= #\F (aref line 0)))
            (multiple-value-bind (n1 n2)
                (parse-fleet planets line)
              (incf n-ships-beyond-1 n1)
              (incf n-ships-beyond-2 n2)))
          do (setq line (read-line stream nil nil)))
    (make-instance 'game :planets planets
                   :n-ships-beyond-2 n-ships-beyond-2
                   :n-ships-beyond-1 n-ships-beyond-1)))

;;; 单行订单：源 id、宿 id、船数。
(defun write-order (order stream)
  (format stream "~D ~D ~D~%" (planet-id (source order))
          (planet-id (destination order))
          (n-ships order)))

(defun write-orders (orders stream)
  (map nil (lambda (order)
             (write-order order stream))
       orders))

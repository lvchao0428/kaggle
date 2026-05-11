;;; =============================================================================
;;; 中文注释副本 — 对应 ../../src/model.lisp
;;; 数据结构、挂单 WITH-ORDERS、战斗与地平线截断。
;;; =============================================================================

(in-package :planet-wars)

;;; COMPILE-TIME：定义比赛规模常量；*N-TURNS-TILL-HORIZON* 等在 COMPUTE-ORDERS 早绑。
(eval-when (:compile-toplevel :load-toplevel :execute)
  ;; These three are only used for the calculation of the width of the
  ;; SHIP-COUNT type.
  (defparameter *max-n-initial-ships* 100)
  (defparameter *max-growth* 5)
  (defparameter *max-n-planets* 23)

  (defparameter *max-n-turns* 200)
  ;; Used quite pervasively, bind this early in COMPUTE-ORDERS.
  (defvar *n-turns-till-horizon*)
  (defvar *n-turns-left-in-game*)
  ;; Number of players including neutral.
  (defparameter *n-players* 3))

;;; There are 23 planets, max growth is 5, 200 turns, 100 initial
;;; ships. (+ 100 (* 23 5 200)) is 23100 that fits into SHIP-COUNT.
;;; 兵力整数类型宽度由最大船数上界推导。
(deftype ship-count ()
  (upgraded-array-element-type
   ;; Signed to allow for packing counts for two players encoded where
   ;; appropriate.
   '(signed-byte #.(1+ (integer-length
                        (1- (+ *max-n-initial-ships*
                               (* *max-n-planets*
                                  *max-growth*
                                  *max-n-turns*))))))))
(deftype ship-count-vector () '(simple-array ship-count (*)))
;;; 玩家枚举（含中立）。
(deftype player ()
  (upgraded-array-element-type
   '(unsigned-byte #.(integer-length (1- *n-players*)))))
(deftype player-vector () '(simple-array player (*)))

(declaim (inline make-count-vector))
;;; 分配 SHIP-COUNT 向量。
(defun make-count-vector (n)
  (make-array n :element-type 'ship-count :initial-element 0))

(declaim (inline make-player-vector))
;;; 分配 PLAYER 向量。
(defun make-player-vector (n)
  (make-array n :element-type 'player :initial-element 0))

;;; Planets are mostly inmmutable except for WITH-ORDER.
;;; PLANET：到港/出发按回合桶、邻接分组；除挂单外视同不可变。
(defclass planet ()
  ((id :initarg :id :reader id)
   (owner :initarg :owner :reader owner)
   (n-ships :initarg :n-ships :reader n-ships)
   (x :initarg :x :reader x)
   (y :initarg :y :reader y)
   (growth :initarg :growth :reader growth)
   ;; Incoming fleets that arrive to the same planet on the same turn
   ;; can be merged. All that matters is owner and total ship count.
   ;; That can be represented by a vector of ship counts indexed by
   ;; turn for each non-neutral player. These two vectors cannot be
   ;; merged into one by treating ships of player 1 as -1, because
   ;; when attacking neutral planets it is important to know what is
   ;; the total count for each.
   (arrivals-1 :type ship-count-vector :initarg :arrivals-1 :reader arrivals-1)
   (arrivals-2 :type ship-count-vector :initarg :arrivals-2 :reader arrivals-2)
   ;; For thinking about orders in the future one must keep track of
   ;; departures. These two could be merged into a signed
   ;; representation without loss of information.
   (departures-1 :type ship-count-vector :initarg :departures-1
                 :reader departures-1)
   (departures-2 :type ship-count-vector :initarg :departures-2
                 :reader departures-2)
   ;; A list of (TURNS-TO-TRAVEL &REST PLANETS) list in ascending
   ;; order of TURNS-TO-TRAVEL. Includes this planet at time 0.
   (neighbours :initform () :initarg :neighbours :reader neighbours)
   (turns-to-neighbours :reader turns-to-neighbours)))

(defclass game ()
  ((planets :initarg :planets :reader planets)
   ;; The number of ships player 1 has in fleets that will arrive
   ;; after the game has ended. These ships would be beyond the end of
   ;; ARRIVALS-1.
   (n-ships-beyond-1 :initarg :n-ships-beyond-1 :reader n-ships-beyond-1)
   (n-ships-beyond-2 :initarg :n-ships-beyond-2 :reader n-ships-beyond-2)
   (caches-and-moves :initform () :accessor caches-and-moves)))

;;; ORDER：可多拍调度，TURN=0 为引擎当场执行。
(defclass order ()
  ((source :initarg :source :reader source)
   (destination :initarg :destination :reader destination)
   (owner :initform 1 :initarg :owner :reader owner)
   (n-ships :initarg :n-ships :reader n-ships)
   ;; This is the turn (relative to the current turn, 0) on which to
   ;; execute this order. It is 0 for orders to be executed by the
   ;; engine.
   (turn :initform 0 :initarg :turn :reader turn)))

(defmethod print-object ((planet planet) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (planet stream :type t)
      (format stream "~S ~S ~S ~S ~S ~S"
              :id (ignore-errors (id planet))
              :owner (ignore-errors (owner planet))
              :n-ships (ignore-errors (n-ships planet)))))
  planet)

(defmethod print-object ((order order) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (order stream :type t)
      (format stream "~S of ~S from ~S to ~S on turn ~S"
              (ignore-errors (n-ships order))
              (ignore-errors (owner order))
              (ignore-errors (id (source order)))
              (ignore-errors (id (destination order)))
              (ignore-errors (turn order)))))
  order)

;;; 到账回合 = ORDER.TURN + 航程。
(defun arrival-turn (order)
  (+ (turn order) (turns-to-travel (source order) (destination order))))

;;; 单笔订单相等。
(defun order= (order1 order2)
  (and (eq (source order1) (source order2))
       (eq (destination order1) (destination order2))
       (= (owner order1) (owner order2))
       (= (n-ships order1) (n-ships order2))
       (= (turn order1) (turn order2))))

(defun move= (move1 move2)
  (and (= (length move1) (length move2))
       (every #'order= move1 move2)))

;;; 己方且相对回合为零。
(defun current-order-p (order)
  (and (owner order)
       (= 0 (turn order))))

;;; Iterate over the groups of neighbouring planets. Each iteration
;;; TURNS-TO-TRAVEL is bound to the number of turns it takes to travel
;;; from PLANET to any planet in the NEIGHBOURS list. TURNS-TO-TRAVEL
;;; is increasing.
;;; 按航程递增遍历邻居分组（跳过 0 航程自环）。
(defmacro do-neighbours (((turns-to-travel neighbours) planet) &body body)
  (alexandria:with-unique-names (e)
    (alexandria:once-only (planet)
      `(loop for ,e in (neighbours ,planet)
        do (destructuring-bind (,turns-to-travel &rest ,neighbours) ,e
             (declare (type fixnum ,turns-to-travel))
             (unless (= 0 ,turns-to-travel)
               ,@body))))))

;;; Like DO-NEIGHBOURS but in reverse order.
(defmacro do-neighbours/reverse (((turns-to-travel neighbours) planet)
                                 &body body)
  (alexandria:with-unique-names (e)
    (alexandria:once-only (planet)
      `(loop for ,e in (reverse (neighbours ,planet))
        do (destructuring-bind (,turns-to-travel &rest ,neighbours) ,e
             (declare (type fixnum ,turns-to-travel))
             (unless (= 0 ,turns-to-travel)
               ,@body))))))

(defun planet-id (obj)
  (if (typep obj 'planet)
      (id obj)
      obj))

;;; 欧式距离 CEILING。
(defun turns-to-travel* (planet1 planet2)
  (ceiling (sqrt (+ (expt (- (x planet1) (x planet2)) 2)
                    (expt (- (y planet1) (y planet2)) 2)))))

;;; 查预缓存向量。
(defun turns-to-travel (planet1 planet2)
  (aref (turns-to-neighbours planet1) (id planet2)))

(defun count-ships-for-battle (owner owner-n-ships fleets)
  (let ((counts (make-count-vector *n-players*)))
    (incf (aref counts owner) owner-n-ships)
    (dolist (fleet fleets)
      (incf (aref counts (owner fleet)) (n-ships fleet)))
    counts))

;;; Return the owner and the number of ships.
;;; Neutral/p1/p2 同时到达会战：幸存方与剩余船。
(defun resolve-battle (owner c0 c1 c2 turn)
  (declare (ignore turn)
           (type ship-count c0 c1 c2)
           (optimize speed))
  (let ((i0 0)
        (i1 1)
        (i2 2))
    (when (< c0 c1)
      (rotatef c0 c1)
      (rotatef i0 i1))
    (when (< c1 c2)
      (rotatef c1 c2)
      (rotatef i1 i2)
      (when (< c0 c1)
        (rotatef c0 c1)
        (rotatef i0 i1)))
    (if (> c0 c1)
        (values i0 (- c0 c1))
        (values owner 0))))

(declaim (inline player-multiplier))
;;; +1/-1/0。
(defun player-multiplier (player)
  (ecase player
    ((0) 0)
    ((1) 1)
    ((2) -1)))

(declaim (inline opponent))
(defun opponent (player)
  (ecase player
    ((1) 2)
    ((2) 1)))

;;; 到达桶索引微调。
(defparameter *turn-adjustment* 0)

;;; Execute ORDER by mutating the planets: change ARRIVALS-{1,2} and
;;; DEPARTURES-{1,2} as appropriate.
;;; 可变写 DEPARTURES/ARRIVALS；桶下溢报错 FUTURE-IMPOSSIBLE。
(defun execute-order (order &optional undo)
  (let* ((n-ships (* (if undo -1 1) (n-ships order)))
         (turn (turn order))
         (source (source order))
         (destination (destination order))
         (turns-to-travel (turns-to-travel source destination)))
    (ecase (owner order)
      ((1)
       (unless (<= 0 (incf (aref (departures-1 (source order)) turn) n-ships))
         (error 'future-impossible))
       (unless (<= 0 (incf (aref (arrivals-1 (destination order))
                                 (+ turn turns-to-travel *turn-adjustment*))
                           n-ships))
         (error 'future-impossible)))
      ((2)
       (unless (<= 0 (incf (aref (departures-2 (source order)) turn) n-ships))
         (error 'future-impossible))
       (unless (<= 0 (incf (aref (arrivals-2 (destination order))
                                 (+ turn turns-to-travel *turn-adjustment*))
                           n-ships))
         (error 'future-impossible))))))

(defun undo-order (order)
  (execute-order order t))

(defun execute-orders (orders)
  (map nil #'execute-order orders))

(defun undo-orders (orders)
  (map nil #'undo-order orders))

;;; 挂单一层：MOVE + 每星 CACHE。
(defclass move-and-stuff ()
  ((move :initarg :move :reader move)
   (cache :initarg :cache :accessor cache)))

;;; 动态栈链；WITH-ORDERS PUSH。
(defvar *moves*)

;;; 回溯到给定层为止积累的订单。
(defun orders-since (move-and-stuff)
  (loop for move-and-stuff* in *moves*
        until (eq move-and-stuff* move-and-stuff)
        append (move move-and-stuff*)))

;;; 在每层 CACHE 按键查找。
(defun lookup-cached-stuff (planet key)
  (let ((id (id planet)))
    (dolist (move-and-stuff *moves*)
      (let ((cache (aref (cache move-and-stuff) id)))
        (dolist (entry cache)
          (declare (optimize speed))
          (when (equal (car entry) key)
            (return-from lookup-cached-stuff
              (values (cdr entry) t (orders-since move-and-stuff)))))))))

(defun set-cached-stuff (planet key values)
  (push (cons key values) (aref (cache (first *moves*)) (id planet))))

;;; EXECUTE→PUSH→BODY→UNWIND 必 UNDO。
(defmacro with-orders ((orders &optional n-planets) &body body)
  (alexandria:once-only (orders)
    (alexandria:with-unique-names (move-and-stuff)
      ;; Without WITHOUT-INTERRUPTS a timeout could unwind before
      ;; EXECUTE-ORDER is complete and signal an error cancelling the
      ;; timeout.
      `(#+sbcl sb-sys:without-interrupts #-sbcl progn
         (unwind-protect
              (progn
                (execute-orders ,orders)
                (let* ((,move-and-stuff
                        (make-instance 'move-and-stuff
                                       :move ,orders
                                       :cache (make-array
                                               (or ,n-planets
                                                   (length (cache
                                                            (first *moves*))))
                                               :initial-element ())))
                       (*moves* (cons ,move-and-stuff
                                      (if (boundp '*moves*)
                                          *moves*
                                          nil))))
                  (#+sbcl sb-sys:with-local-interrupts #-sbcl progn
                    ,@body)))
           (undo-orders ,orders))))))

;;; 将四向量裁到地平线长度。
(defun truncate-planet (planet n)
  (let ((arrivals-1 (make-count-vector (1+ n)))
        (arrivals-2 (make-count-vector (1+ n)))
        (departures-1 (make-count-vector (1+ n)))
        (departures-2 (make-count-vector (1+ n))))
    (replace arrivals-1 (arrivals-1 planet))
    (replace arrivals-2 (arrivals-2 planet))
    (replace departures-1 (departures-1 planet))
    (replace departures-2 (departures-2 planet))
    (setf (slot-value planet 'arrivals-1) arrivals-1
          (slot-value planet 'arrivals-2) arrivals-2
          (slot-value planet 'departures-1) departures-1
          (slot-value planet 'departures-2) departures-2)))

(defun truncate-game (game n)
  (map nil  (lambda (planet)
              (truncate-planet planet n))
       (planets game)))

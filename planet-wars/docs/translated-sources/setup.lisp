;;; =============================================================================
;;; 中文注释副本 — 对应 ../../setup.lisp
;;; 在无 Quicklisp 环境下把本仓库所有 *.asd 所在目录登记进 asdf:*central-registry*。
;;; =============================================================================

;;;; Set up asdf locations.

(require :asdf)

(let* ((dir (pathname-directory *load-truename*))
       (asdf-files (directory
                    (merge-pathnames "**/*.asd"
                                     (make-pathname :directory dir)))))
  (setq asdf:*central-registry*
        (mapcar (lambda (directory)
                  (make-pathname :directory directory))
                (remove-duplicates (mapcar #'pathname-directory asdf-files)
                                   :test #'equal))))

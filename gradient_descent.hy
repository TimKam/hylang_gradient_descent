(import numpy)
(require [hy.contrib.loop [loop]])

;; our sigmoid function, which can also provide its derivative
(defn nonlin [x &optional (derive False)] (
        if derive
        (return (* x (- 1 x)))
        (return (/ 1 (+ 1 (.exp numpy (- x)))))))


;; training
(defn train [weights input-data output-data bias n_steps]
  (setv layer-0 (.append numpy input-data bias 1))
  (print layer-0)
  (loop [[i n_steps] [acc 1]]
    (global layer-2)
      (if (zero? i)
        [(return layer-2)]
        (recur (dec i)[  ;; do this `index` times
          (setv layer-1 (nonlin (.dot numpy layer-0 (get weights 0))))
          (setv layer-2 (nonlin (.dot numpy layer-1 (get weights 1))))

          ;; calculate errors
          (setv layer-2-error (- output-data layer-2))
          (setv layer-2-delta (* layer-2-error (nonlin layer-2 True)))
          (setv layer-1-error (.dot layer-2-delta (. (get weights 1) T)))
          (setv layer-1-delta (* layer-1-error (nonlin layer-1 True)))

          (if (= (% i 10000) 0)
              (print (+ "Error " (str (.mean numpy(.abs numpy layer-2-error))))))

          ;; update weights
          (assoc weights 0 (+ (get weights 0) (.T.dot layer-0 layer-1-delta)))
          (assoc weights 1 (+ (get weights 1) (.T.dot layer-1 layer-2-delta)))]))))


;; define input parameters
(setv input-data ;; training input
  (.array numpy
    [[0 0]
     [0 1]
     [1 0]
     [1 1]]))

(setv output-data ;; training output
  (.array numpy
    [[0]
     [1]
     [1]
     [0]]))

(setv bias
  (.array numpy
    [[1]
     [1]
     [1]
     [1]]))

(.random.seed numpy 1)

(setv weights {
                 0 (- (* 2 (.random.random numpy [3 4])) 1)
                 1 (- (* 2 (.random.random numpy [4 1])) 1)})

;; run traning function
(print
  (+ "Output after training \n\r"
     (str(train weights input-data output-data bias 60000))))
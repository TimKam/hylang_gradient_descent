(import numpy)
(require [hy.contrib.loop [loop]])

;; our sigmoid function
(defn nonlin [x &optional (derive False)] (
        if derive
        (return (* x (- 1 x)))
        (return (/ 1 (+ 1 (.exp numpy (- x)))))))


;; training
(defn train [synapses input-data index]
  (loop [[i index] [acc 1]]
    (global layer-2)
      (if (zero? i)
        [(return layer-2)]
        (recur (dec i)[  ;; do this `index` times
          (setv layer-0 input-data)
          (setv layer-1 (nonlin (.dot numpy layer-0 (get synapses 0))))
          (setv layer-2 (nonlin (.dot numpy layer-1 (get synapses 1))))
          (setv layer-2-error (- output-data layer-2))

          (if (= (% i 10000) 0)
              (print (+ "Error " (str (.mean numpy(.abs numpy layer-2-error))))))

          (setv layer-2-delta (* layer-2-error (nonlin layer-2 True)))
          (setv layer-1-error (.dot layer-2-delta (. (get synapses 1) T)))
          (setv layer-1-delta (* layer-1-error (nonlin layer-1 True)))
          ;; update weights
          (assoc synapses 0 (+ (get synapses 0) (.T.dot layer-0 layer-1-delta)))
          (assoc synapses 1 (+ (get synapses 1) (.T.dot layer-1 layer-2-delta)))]))))


;; define input data & synapses
(setv input-data
  (.array numpy ;; note: 3rd column contains bias term
    [[0 0 1]
     [0 1 1]
     [1 0 1]
     [1 1 1]]))

(setv output-data
  (.array numpy
    [[0]
     [1]
     [1]
     [0]]))

(.random.seed numpy 1)

(setv synapses {
                 0 (- (* 2 (.random.random numpy [3 4])) 1)
                 1 (- (* 2 (.random.random numpy [4 1])) 1)})

;; run traning function
(print (+ "Output after training \n\r" (str (train synapses input-data 60000))))
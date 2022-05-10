#lang slideshow
(define c (circle 10))
(define r (rectangle 10 20))
(hc-append 20 c r c)

(define (square n)
  ; A semi-colon starts a line comment.
  ; The expression below is the function body.
  (filled-rectangle n n)
  )

(square 10)

(define (four p)
  (define two-p (hc-append p p))
  (vc-append two-p two-p))

(four (circle 10))

(define (checker p1 p2)
  (let ([p12 (hc-append p1 p2)]
        [p21 (hc-append p2 p1)])
    (vc-append p12 p21)))

(checker (colorize (square 10) "red")
         (colorize (square 10) "black"))

(define (checkerboard p)
  (let* ([rp (colorize p "red")]
         [bp (colorize p "black")]
         [c (checker rp bp)]
         [c4 (four c)])
    (four c4)))

(checkerboard (square 10))


; 6. Functions are values
(define (series mk)
  (hc-append 4 (mk 4) (mk 10) (mk 20)))

(series circle)
(series (lambda (size) (checkerboard (square size))))

; 7. Lexical Scope
(define (rgb-series mk)
  (vc-append (series (lambda (sz) (colorize (mk sz) "red")))
             (series (lambda (sz) (colorize (mk sz) "green")))
             (series (lambda (sz) (colorize (mk sz) "blue")))
             ))

(rgb-series circle)

(define (rgb-marker mk)
  (lambda (sz)
    (vc-append (colorize (mk sz) "red")
               (colorize (mk sz) "green")
               (colorize (mk sz) "blue"))))

(series (rgb-marker circle))

; 8. Lists
(define l (list "red" "green" "blue"))
(define ll (list (circle 10) (square 10)))

(define (rainbow p)
  (map (lambda (color)
         (colorize p color))
       (list "red" "orange" "yellow" "green" "blue" "purple")))

(rainbow (square 10))

(apply vc-append (rainbow (square 10)))

; 9. Modules
(require pict/flash)
(filled-flash 40 30)

; 10. Macros
(require slideshow/code)
(code (circle 10))

(define-syntax pict+code
  (syntax-rules ()
    [(pict+code expr)
     (hc-append 10
                expr
                (code expr))]))

(pict+code (circle 10))

; 11. Objects
(require racket/class
         racket/gui/base)
(define f (new frame% [label "My Art"]
                      [width 300]
                      [height 300]
                      [alignment '(center center)]))

(send f show #t) ; #t is true, send calls the method show of f

(define (add-drawing p)
  (let ([drawer (make-pict-drawer p)])
    (new canvas% [parent f]
                 [style '(border)]
                 [paint-callback (lambda (self dc) (drawer dc 0 0))])))

(add-drawing (pict+code (circle 10)))
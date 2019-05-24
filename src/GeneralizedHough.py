import numpy as np
import cv2 as cv

img1 = cv.imread('../image/generalized_hough_demo_01.png')
cv.imshow("input", img1)
templ = cv.imread('../image/generalized_hough_demo_02.png')
cv.imshow('output', templ)

ballard = 1

if ballard == 1:
    alg = cv.GeneralizedHoughBallard();
else:
    alg = cv.GeneralizedHoughGuil();

alg.setTemplate(templ);

k = cv.waitKey(0)
if k == 27:
# wait for ESC key to exit
    cv2.destroyAllWindows()


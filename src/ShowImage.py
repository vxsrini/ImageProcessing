import numpy as np
import cv2 as cv
import math as mt

print(cv.__version__)
print(cv.__file__)

color_img1 = cv.imread('../image/Eyes.png')
cv.imshow("Switch", color_img1)
templ = cv.imread('../image/Eyes.png', 0)
cv.imshow("Free Port Template",templ)
k = cv.waitKey(0)
if k == 27:
# wait for ESC key to exit
    cv.destroyAllWindows()

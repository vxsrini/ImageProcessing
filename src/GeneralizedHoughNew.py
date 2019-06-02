import numpy as np
import cv2 as cv
import math as mt

print(cv.__version__)
print(cv.__file__)

#color_img1 = cv.imread('../image/generalized_hough_demo_01.png')
color_img1 = cv.imread('../image/Face.png')
cv.imshow("Switch", color_img1)
img1 = cv.cvtColor(color_img1, cv.COLOR_BGR2GRAY)
#templ = cv.imread('../image/generalized_hough_demo_02.png', 0)
templ = cv.imread('../image/Eyes.png', 0)
cv.imshow("Free Port Template",templ)
ballard = 0

if ballard == 1:
    alg = cv.createGeneralizedHoughBallard();
else:
    alg = cv.createGeneralizedHoughGuil();

alg.setTemplate(templ);
positions, pts = alg.detect(img1)

print(positions)
#print(pts)
#[[144. 267.   1.   0.]]


for opos in positions:
	for pos in opos:
		print(pos)
		topCornerX = int(pos[0] - 10)
		topCornerY = int(pos[1] - 10)
		bottomCornerX = int(pos[0] + 10)
		bottomCornerY = int(pos[1] + 10)
		print(topCornerY)
		#cv.rectangle(color_img1, (topCornerX, topCornerY), (bottomCornerX, bottomCornerY), (0,255,0), 2)
		cv.circle(color_img1, (pos[0], pos[1]), 10, (0,255,0), 2)
		#cv.rectangle(color_img1, (10, 10), (20, 20), (0, 255, 0), 2)
	
cv.imshow("Final", color_img1)
k = cv.waitKey(0)
if k == 27:
# wait for ESC key to exit
    cv.destroyAllWindows()

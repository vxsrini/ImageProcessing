import numpy as np
import cv2 as cv
import math as mt

print(cv.__version__)
print(cv.__file__)

color_img1 = cv.imread('../image/Panel.png')
#color_img1 = cv.imread('../image/Face.jpg')
img1 = cv.cvtColor(color_img1, cv.COLOR_BGR2GRAY)
color_templ = cv.imread('../image/PanelPort_1.png')
#color_templ = cv.imread('../image/Eyes.jpg')
templ = cv.cvtColor(color_templ, cv.COLOR_BGR2GRAY)

ballard = 1
if ballard == 1:
    alg = cv.createGeneralizedHoughBallard();
else:
    alg = cv.createGeneralizedHoughGuil();

alg.setTemplate(templ);
alg.setVotesThreshold(10)
alg.setMinDist(80)
positions, pts = alg.detect(img1)

print("Positions = " + str(positions))
print("Votes = " + str(pts))


for opos in positions:
	for pos in opos:
		print(str(int(pos[0])) + "|" + str(int(pos[1])))
		topCornerX = int(pos[0] - 10)
		topCornerY = int(pos[1] - 10)
		bottomCornerX = int(pos[0] + 10)
		bottomCornerY = int(pos[1] + 10)
		#print(topCornerY)
		#cv.rectangle(color_img1, (topCornerX, topCornerY), (bottomCornerX, bottomCornerY), (0,255,0), 2)
		cv.circle(color_img1, (pos[0], pos[1]), 10, (0,255,0), 2)
		#cv.rectangle(color_img1, (10, 10), (20, 20), (0, 255, 0), 2)
	
cv.namedWindow('Final',cv.WINDOW_NORMAL)
cv.resizeWindow('Final', 1500,900)
cv.imshow("Final", color_img1)
cv.imwrite('../image/final.png',color_img1)
k = cv.waitKey(0)
if k == 27:
# wait for ESC key to exit
    cv.destroyAllWindows()

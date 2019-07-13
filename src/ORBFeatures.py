import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../image/EthernetSwitch.png',0)          # queryImage
color_img1 = cv2.imread('../image/EthernetSwitch.png')          # queryImage
img2 = cv2.imread('../image/FreePort_2.png',0) # trainImage

orb=cv2.ORB_create()
# find the keypoints with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
n_img1 = img1
# draw only keypoints location,not size and orientation
n_img1 = cv2.drawKeypoints(img1,kp1, n_img1, color=(0,255,0), flags=0)

kp2, des2 = orb.detectAndCompute(img2,None)
n_img2 = img2
# draw only keypoints location,not size and orientation
n_img2 = cv2.drawKeypoints(img2,kp2, n_img2, color=(0,255,0), flags=0)

bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches=bf.match(des1,des2)
matches=sorted(matches, key= lambda x:x.distance)

# Initialize lists
list_kp1 = []
list_kp2 = []

# For each match...
for mat in matches:

	# Get the matching keypoints for each of the images
	img1_idx = mat.queryIdx
	img2_idx = mat.trainIdx

	# x - columns
	# y - rows
	# Get the coordinates
	(x1,y1) = kp1[img1_idx].pt
	(x2,y2) = kp2[img2_idx].pt

	# Append to each list
	list_kp1.append((x1, y1))
	list_kp2.append((x2, y2))

for pos in list_kp1:
	print(str(int(pos[0])) + "|" + str(int(pos[1])))
	cv2.circle(color_img1, (int(pos[0]), int(pos[1])), 2, (0,255,0), 2)

#cv.namedWindow('Final',cv.WINDOW_NORMAL)
#cv.resizeWindow('Final', 1500,900)
cv2.imshow("Final", color_img1)
k = cv2.waitKey(0)
if k == 27:
# wait for ESC key to exit
    cv2.destroyAllWindows()


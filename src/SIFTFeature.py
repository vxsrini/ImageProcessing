import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../image/EthernetSwitch.png',0)          # queryImage
color_img1 = cv2.imread('../image/EthernetSwitch.png')          # queryImage
img2 = cv2.imread('../image/FreePort.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print("Des1")
print(des1)
print ("Des2")
print(des2)

'''
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)



img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
'''

#bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf=cv2.BFMatcher()
matches=bf.match(des1,des2)
matches=sorted(matches, key= lambda x:x.distance)
#matches = bf.knnMatch(des1,des2, k=2)

#print(matches)
#good = []
#for m,n in matches:
#    if m.distance < 0.75*n.distance:
#        good.append([m])

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



print("Matches .... ")
print(len(matches))
print("Des2 ...")
print(len(des2))
for pos in list_kp1:
        print(str(int(pos[0])) + "|" + str(int(pos[1])))
        cv2.circle(color_img1, (int(pos[0]), int(pos[1])), 1, (0,255,0), 2)

#cv.namedWindow('Final',cv.WINDOW_NORMAL)
#cv.resizeWindow('Final', 1500,900)
cv2.imshow("Final", color_img1)
k = cv2.waitKey(0)
if k == 27:
# wait for ESC key to exit
    cv2.destroyAllWindows()

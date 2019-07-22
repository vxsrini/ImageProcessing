import cv2
import numpy as np
from matplotlib import pyplot as plt

f = open("/tmp/op.txt", "w")

################
# Find Contours of Panel
###################


origPanelImg = cv2.imread('../image/Panel.png')
panel_img = cv2.imread('../image/Panel.png', 0)
# panel_img = cv2.medianBlur(panel_img, 5)
# th_panel_img = cv2.adaptiveThreshold(
#     panel_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
panel_img = cv2.cvtColor(panel_img, cv2.COLOR_BGR2RGB)
canny = cv2.Canny(panel_img,100,200)


plt.imshow(canny , 'gray')
plt.title("Panel before Contour")
plt.xticks([]) ,plt.yticks([])
plt.show()

pnl_contours, pnl_hierarchy = cv2.findContours(
    canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



# a = []
# for i in pnl_contours:
#     approx = cv2.approxPolyDP(i, 0.0025*cv2.arcLength(i, True), True)
#     a.append(approx)

cv2.drawContours(origPanelImg, pnl_contours, -1, (0, 0, 255), 1)


plt.imshow(origPanelImg)
plt.title("Panel after contour")
plt.xticks([]) ,plt.yticks([])
plt.show()


# ################
# # Calculate Contours of Panel Port
# ###################
origPanelPort_img = cv2.imread('../image/FreePort_1.png')
panelPort_img = cv2.imread('../image/FreePort_1.png', 0)
blur = cv2.GaussianBlur(panelPort_img, (5, 5), 0)
ret3, th_panelPort_img = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th_panelPort_img = cv2.adaptiveThreshold(
    th_panelPort_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

'''
plt.imshow(th_panelPort_img , 'gray')
plt.title("Panel Port Before Contour")
plt.xticks([]) ,plt.yticks([])
plt.show()
'''

port_contours, port_hierarchy = cv2.findContours(
    th_panelPort_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# b = []
# for j in port_contours:
#     approx = cv2.approxPolyDP(j, 0.01*cv2.arcLength(j, True), True)
#     b.append(approx)

# print(b)
cv2.drawContours(origPanelPort_img, port_contours, -1, (0, 255, 255), 1)

plt.imshow(origPanelPort_img)
plt.title("Panel Port after contour")
plt.xticks([]), plt.yticks([])
plt.show()



###################
#match the contours
##################
for cnt in pnl_contours:
    ret = cv2.matchShapes(port_contours[1], cnt, 1, 0.0)
    f.write(str(ret) + "\n")
    # if (ret < 1):
    cv2.drawContours(origPanelImg, cnt, -1 , (255 ,0 ,255) , 1)



plt.imshow(origPanelImg)
plt.title("Matching Contour")
plt.xticks([]) ,plt.yticks([])
plt.show()


f.close()

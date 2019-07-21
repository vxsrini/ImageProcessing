import cv2
import numpy as np
from matplotlib import pyplot as plt

panel_img = cv2.imread('../image/Panel.png' ,0)
panel_img = cv2.medianBlur(panel_img ,5)
#panel_img = cv2.GaussianBlur(panel_img ,(5 ,5) ,0)

th_panel_img = cv2.adaptiveThreshold(panel_img ,255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY ,11 ,2)
'''
plt.imshow(th_panel_img , 'gray')
plt.title("Panel before Contour")
plt.xticks([]) ,plt.yticks([])
plt.show()
'''
contours , hierarchy = cv2.findContours(th_panel_img ,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(th_panel_img , contours , -1 , (0 ,255 ,0) , 3)

'''
plt.imshow(th_panel_img , 'gray')
plt.title("Panel after contour")
plt.xticks([]) ,plt.yticks([])
plt.show()

'''

panelPort_img = cv2.imread('../image/FreePort_1.png' , 0)
blur = cv2.GaussianBlur(panelPort_img ,(5 ,5) ,0)
ret3 ,th_panelPort_img = cv2.threshold(blur ,0 ,255 ,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th_panelPort_img = cv2.adaptiveThreshold(th_panelPort_img  ,255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY ,11 ,2)

plt.imshow(th_panelPort_img , 'gray')
plt.title("Panel Port Before Contour")
plt.xticks([]) ,plt.yticks([])
plt.show()

contours , hierarchy = cv2.findContours(th_panelPort_img ,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(th_panelPort_img , contours , 1 , (0 ,255 ,0) , 3)

plt.imshow(th_panelPort_img , 'gray')
plt.title("Panel Port after contour")
plt.xticks([]) ,plt.yticks([])
plt.show()

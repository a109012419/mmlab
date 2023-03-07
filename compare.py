from cv2 import cv2
from matplotlib import pyplot as plt

img_path1 = 'r50.png'
img1 = cv2.imread(img_path1)


img_path2 = 'r101.png'
img2 = cv2.imread(img_path2)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()
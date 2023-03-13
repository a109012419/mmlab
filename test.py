import math

import matplotlib.pyplot as plt
import rasterio
from cv2 import cv2

data = rasterio.open('data/loveDA/ann_dir/train/0.png')
label = data.read()[0]

img = cv2.imread('data/loveDA/img_dir/train/0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

data = [label, img,img]
lens = len(data)
rows = math.ceil(lens / 2)
fig, axes = plt.subplots(rows, 2, sharex=True, sharey=True)

if rows == 2:
    count = 0
    for i in range(rows):
        for j in [0, 1]:
            if count + 1 > lens: break
            axes[i, j].imshow(data[count])
            count += 1
else:
    axes[0].imshow(data[0])
    axes[1].imshow(data[1])

plt.show()

import matplotlib.pyplot as plt
import mmcv
from cv2 import cv2

from mmseg.apis import init_segmentor, inference_segmentor

config = 'config_self/psp/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint = 'config_self/psp/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# config = 'configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py'
# checkpoint = 'config/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth'


device = 'cuda'

model = init_segmentor(config, checkpoint, device=device)
print(type(model))
# img_path = 'demo/demo.png'
# img = cv2.imread(img_path)

# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
# ax1.imshow(img1)
# ax2.imshow(img2)
# plt.show()

# result = inference_segmentor(model, img)
#
# seg = result[0]
#
# plt.imsave('r50.png',seg)
# print('Done!')

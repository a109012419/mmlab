import matplotlib.pyplot as plt
import mmcv
import torch
from cv2 import cv2
import torchinfo
from mmcv.parallel import collate, scatter
from torchvision.transforms import ToTensor

from config_self.rxd_show import imgs_display
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose

config = 'config_self/psp/loveDA/pspnet_r101-d8_512x512_80k_loveda.py'
checkpoint = 'config_self/psp/loveDA/pspnet_r101-d8_512x512_80k_loveda_20211104_153212-1c06c6a8.pth'

# config = 'configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py'
# checkpoint = 'config/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth'


device = 'cuda'

model = init_segmentor(config, device=device)

img_path = 'data/loveDA/img_dir/train/6.png'
image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# result = inference_segmentor(model, img)

# data loading————————————————————————————
cfg = model.cfg
device = next(model.parameters()).device  # model device
# build the data pipeline

test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)
# prepare data
data = []
imgs = image if isinstance(image, list) else [image]
for img in imgs:
    img_data = dict(img=img)
    img_data = test_pipeline(img_data)
    data.append(img_data)
data = collate(data, samples_per_gpu=len(imgs))
if next(model.parameters()).is_cuda:
    # scatter to specified GPU
    data = scatter(data, [device])[0]
else:
    data['img_metas'] = [i.data[0] for i in data['img_metas']]
# ——————————————————————————————

print(data['img'][0].shape)
# feat = model.extract_feat(data['img'][0])
# print(feat[3].shape)
feats = model.encode_decode(data['img'][0], data['img_metas'])
feats = feats.cpu().detach().numpy()[0]

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imgs_display(feats[0], feats[1], feats[2], cv2.resize(image, (512, 512)))

# model————>test
# with torch.no_grad():
#     result = model(return_loss=False, rescale=True, **data)

# res = result[0]
# print(res.shape)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# imgs_display(res,image)

import torchinfo
import torchvision
from torch import nn

from mmseg.apis import init_segmentor

config = 'configs/pspnet/pspnet_r101-d8_512x512_80k_loveda.py'
checkpoint = 'config_self/psp/loveDA/pspnet_r101-d8_512x512_80k_loveda_20211104_153212-1c06c6a8.pth'

device = 'cuda'


model = init_segmentor(config,device=device)


# torchinfo.summary(model,(1, 3,512,512))
# print('___'*10)
# decode_head
# print(isinstance(model,nn.Module))

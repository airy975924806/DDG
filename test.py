# —*——coding:utf-8——*—
# Author：airy
# Date: 2022/10/17 8:21
# Name: test
import numpy as np
import torch
from PIL import Image


def create_mask( ths, ref_fft):
    # null mask

    mask = torch.ones((ref_fft.shape), dtype=torch.bool)
    _, _, h, w = ref_fft.shape
    b_h = np.floor((h * ths) / 2.0).astype(int)
    b_w = np.floor((w * ths) / 2.0).astype(int)

    mask[:, :, 0:b_h, 0:b_w] = 0  # top left
    mask[:, :, 0:b_h, w - b_w:w] = 0  # top right
    mask[:, :, h - b_h:h, 0:b_w] = 0  # bottom left
    mask[:, :, h - b_h:h, w - b_w:w] = 0  # bottom right

    return mask

def fre_remove_random(img, attn):
    # im_fft = torch.fft.fft2(img) #im: input image b, 3, h, w
    im_fft = torch.rfft(img, 2, onesided=False, normalized=True)
    # img_fft = im_fft[...,0]+im_fft[...,1]

    # im_fft = torch.fft.fft(im.clone(), 2) #im: input image b, 3, h, w
    print(im_fft.shape)
    b, c, im_h, im_w,_ = im_fft.shape

    # decomposition
    ths = list(np.arange(0., 1 + 1. / 64, 1. / 64))
    each_fft = []
    # for i in range(len(ths)-1):
    for i in range(0, 64):
        t1 = ths[i]
        t2 = ths[i + 1]

        mask1 = create_mask(t1, im_fft)
        mask2 = create_mask(t2, im_fft)
        mask = torch.logical_xor(mask1, mask2)
        band_fft = im_fft * mask.to(im_fft.device)
        band_fft = torch.unsqueeze(band_fft, 1)
        each_fft.append(band_fft)

    fft_all = torch.cat(each_fft, dim=1)  # b, 64, 3, w, h, 2

    # recover
    fft_sum = torch.sum(fft_all * attn, 1)

    reverse_img = torch.irfft(fft_sum).real
    # reverse_img = reverse_img.squeeze(0)
    return reverse_img


if __name__ == '__main__':
    input = torch.randn(4,3,64,64)
    # img = Image.open('1.png').convert('RGB')
    # img = torch.from_numpy(np.array(input).transpose(2,0,1)/255)
    im_fft = torch.rfft(input,2,onesided=False,normalized=True)
    print(im_fft.shape)
    re_img = torch.stack((im_fft[...,0],im_fft[...,1]),-1)
    print(re_img.shape)
    im_ff = torch.complex(im_fft[...,0],im_fft[...,1])
    print(im_ff.shape)
    new_fft = torch.stack((im_ff.real,im_ff.imag),-1)
    print(new_fft.shape)
    # print(im_fft[...,0])
    # print(im_fft[...,1])
    # print(im_ff[...,0][0])
    # img_re = torch.irfft(im_ff,2,onesided=False,normalized=True)
    # print(img==img_re)
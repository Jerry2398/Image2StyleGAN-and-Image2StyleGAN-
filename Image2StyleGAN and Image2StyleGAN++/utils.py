from PIL import Image
from torchvision import transforms
from math import log10
import torch


def get_device(use_cuda=True):
    if use_cuda:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def image_reader(img_path):
    with open(img_path,"rb") as f:
        image=Image.open(f)
        image=image.convert("RGB")
    transform = transforms.Compose([
        transforms.CenterCrop((1024, 1024)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def loss_function(syn_img, img, img_p, MSE_loss, upsample, perceptual):
    syn_img_p = upsample(syn_img)
    syn0, syn1, syn2, syn3 = perceptual(syn_img_p)
    r0, r1, r2, r3 = perceptual(img_p)
    mse = MSE_loss(syn_img, img)

    per_loss = 0
    per_loss += MSE_loss(syn0,r0)
    per_loss += MSE_loss(syn1,r1)
    per_loss += MSE_loss(syn2,r2)
    per_loss += MSE_loss(syn3,r3)

    return mse, per_loss


def W_loss(syn_img, img, MSE_loss, upsample, perceptual, lamb_p, lamb_mse, M_p = None , M_m = None):
    '''
    For W_l loss
    '''
    # adding mask on image
    if M_m is not None:
        mse = MSE_loss(M_m * syn_img, M_m * img)
    else:
        mse = MSE_loss(syn_img, img)

    if M_p is not None:
        syn_img_p = upsample(M_p * syn_img)
        img_p = upsample(M_p * img)
    else:
        syn_img_p = upsample(syn_img)
        img_p = upsample(img)

    syn0, syn1, syn2, syn3 = perceptual(syn_img_p)
    r0, r1, r2, r3 = perceptual(img_p)

    per_loss = 0
    per_loss += MSE_loss(syn0, r0)
    per_loss += MSE_loss(syn1, r1)
    per_loss += MSE_loss(syn2, r2)
    per_loss += MSE_loss(syn3, r3)

    loss = lamb_p * per_loss + lamb_mse * mse

    return loss


def Mkn_loss(syn_image, image1, image2, MSE_loss, lamd_mse1, lamb_mse2, M=None):
    '''
        For noise optimization loss
    '''
    if M is not None:
        syn_image1 = M * syn_image
        syn_image2 = (1-M) * syn_image
        image1 = M * image1
        image2 = (1-M) * image2
    else:
        syn_image1 = syn_image
        syn_image2 = syn_image
        image1 = image1
        image2 = image2

    mse = lamd_mse1 * MSE_loss(syn_image1, image1) + lamb_mse2 * MSE_loss(syn_image2, image2)
    return mse


def Mst_loss(syn_image, image, upsample, lamb_s, MSE_loss, style, M_s = None):
    '''
        For style loss
    '''
    if M_s is not None:
        syn_img_p = upsample(M_s * syn_image)
        img_p = upsample(M_s * image)
    else:
        syn_img_p = upsample(syn_image)
        img_p = upsample(image)

    syn_style = style(syn_img_p)
    img_style = style(img_p)

    loss = lamb_s * MSE_loss(syn_style, img_style)
    return loss


def PSNR(mse, flag = 0):
    if flag == 0:
        psnr = 10 * log10(1 / mse.item())
    return psnr
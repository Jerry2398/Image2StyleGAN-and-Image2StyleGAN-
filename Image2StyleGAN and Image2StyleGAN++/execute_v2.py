from image2stylegan_v2 import G_mapping, G_synthesis
from VGG16 import VGG16_perceptual, VGG16_style
from utils import image_reader, get_device, W_loss, Mkn_loss, Mst_loss
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--use_cuda", action='store', default=True, type=bool)
parser.add_argument("--model_dir", action='store', default="pretrain_stylegan", type=str)
parser.add_argument("--model_name", action='store', default="karras2019stylegan-ffhq-1024x1024.pt", type=str)
parser.add_argument("--images_dir", action='store', default="images/image2stylegan_v2", type=str)
parser.add_argument("--lr", action='store', default=0.01, type=float)
parser.add_argument("--w_epochs", action='store', default=1500, type=int)
parser.add_argument("--n_epochs", action='store', default=1500, type=int)
args = parser.parse_args()

device = get_device(args.use_cuda)
g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),
                                   ('g_synthesis', G_synthesis(resolution=1024))
                                   ]))

# Load the pre-trained model
g_all.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name), map_location=device))
g_all.eval()
g_all.to(device)
g_mapping, g_synthesis = g_all[0], g_all[1]

print("success")


def image_reconstruction(image):
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)

    perceptual = VGG16_perceptual().to(device)
    MSE_loss = nn.MSELoss(reduction="mean")

    # since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
    w = torch.zeros((1, 18, 512), requires_grad=True, device=device)

    # noise 初始化，noise是直接加在feature map上的
    noise_list = []
    for i in range(0, 9):
        noise_list.append(torch.randn((1, 1, pow(2, i + 2), pow(2, i + 2)), requires_grad=True, device=device))
        noise_list.append(torch.randn((1, 1, pow(2, i + 2), pow(2, i + 2)), requires_grad=True, device=device))

    # Optimizer to change latent code in each backward step
    w_opt = optim.Adam({w}, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    n_opt = optim.Adam(noise_list, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    loss_ = []
    loss_psnr = []

    # 优化w
    for e in range(args.w_epochs):
        w_opt.zero_grad()
        syn_img = g_synthesis(w, noise_list)
        syn_img = (syn_img + 1.0) / 2.0
        loss = W_loss(syn_img, image, img_p, MSE_loss, upsample, perceptual, 1e-5, 1e-5)
        loss.backward()
        w_opt.step()
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {}".format(e + 1, loss.item()))
            save_image(syn_img.clamp(0, 1), "save_images/image2stylegan_v2/image_reconstruct/reconstruct_{}.png".format(e + 1))

    # 优化noise
    for e in range(args.w_epochs, args.w_epochs + args.n_epochs):
        n_opt.zero_grad()
        syn_img = g_synthesis(w, noise_list)
        syn_img = (syn_img + 1.0) / 2.0
        loss = Mkn_loss(syn_img, image1, image1, MSE_loss, 1e-5, 0)
        loss.backward()
        n_opt.step()
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {}".format(e + 1, loss.item()))
            save_image(syn_img.clamp(0, 1), "save_images/image2stylegan_v2/image_reconstruct/reconstruct_{}.png".format(e + 1))

    plt.plot(loss_, label='Loss = MSELoss + Perceptual')
    plt.plot(loss_psnr, label='PSNR')
    plt.legend()
    return w, noise_list


def image_crossover(image1, image2):
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')

    # 计算mask
    half_blur_shape = np.array(image1.size())
    half_blur_shape[-1] = int(half_blur_shape[-1]/2)
    half_blur_shape = tuple(half_blur_shape)
    print(half_blur_shape)
    M_blur_left = torch.ones(size=half_blur_shape)
    M_blur_right = torch.zeros(size=half_blur_shape)
    M_blur = torch.cat([M_blur_left, M_blur_right], dim=-1)
    M_blur = M_blur.to(device)
    print(M_blur.shape)

    # perception loss 与 MSE loss
    perceptual = VGG16_perceptual().to(device)
    MSE_loss = nn.MSELoss(reduction="mean")

    # since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
    w = torch.zeros((1, 18, 512), requires_grad=True, device=device)

    # we add noise on feature map for each layer
    noise_list = []
    for i in range(0, 9):
        noise_list.append(torch.randn((1, 1, pow(2, i + 2), pow(2, i + 2)), requires_grad=True, device=device))
        noise_list.append(torch.randn((1, 1, pow(2, i + 2), pow(2, i + 2)), requires_grad=True, device=device))

    # Optimizer to change latent code in each backward step
    w_opt = optim.Adam({w}, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    n_opt = optim.Adam(noise_list, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    loss_ = []
    loss_psnr = []
    # w optimization
    for e in range(args.w_epochs):
        w_opt.zero_grad()
        syn_img = g_synthesis(w, noise_list)
        syn_img = (syn_img + 1.0) / 2.0
        loss = W_loss(syn_img, image1, MSE_loss, upsample, perceptual, 1e-5, 1e-5, M_blur, M_blur) + W_loss(syn_img, image2, MSE_loss, upsample, perceptual, 1e-5, 1e-5, 1- M_blur, 1 - M_blur)
        loss.backward()
        w_opt.step()
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {}".format(e + 1, loss.item()))
            save_image(syn_img.clamp(0, 1), "save_images/image2stylegan_v2/crossover/crossover_{}.png".format(e + 1))

    # noise optimization
    for e in range(args.w_epochs, args.w_epochs + args.n_epochs):
        n_opt.zero_grad()
        syn_img = g_synthesis(w, noise_list)
        syn_img = (syn_img + 1.0) / 2.0
        loss = Mkn_loss(syn_img, image1, image2, MSE_loss, 1e-5, 1e-5, M_blur)
        loss.backward()
        n_opt.step()
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {}".format(e + 1, loss.item()))
            save_image(syn_img.clamp(0, 1), "save_images/image2stylegan_v2/crossover/crossover_{}.png".format(e + 1))

    plt.plot(loss_, label='Loss = MSELoss + Perceptual')
    plt.plot(loss_psnr, label='PSNR')
    plt.legend()
    return w, noise_list


def local_style_transfer(image1, image2, mask):
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')

    M_blur = mask.to(device)
    print(M_blur.shape)

    perceptual = VGG16_perceptual().to(device)
    style = VGG16_style().to(device)
    MSE_loss = nn.MSELoss(reduction="mean")

    # since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
    w = torch.zeros((1, 18, 512), requires_grad=True, device=device)
    noise_list = []
    for i in range(0, 9):
        noise_list.append(torch.randn((1, 1, pow(2, i + 2), pow(2, i + 2)), requires_grad=True, device=device))
        noise_list.append(torch.randn((1, 1, pow(2, i + 2), pow(2, i + 2)), requires_grad=True, device=device))

    # Optimizer to change latent code in each backward step
    w_opt = optim.Adam({w}, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    n_opt = optim.Adam(noise_list, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    loss_ = []
    loss_psnr = []
    for e in range(args.w_epochs):
        w_opt.zero_grad()
        syn_img = g_synthesis(w, noise_list)
        syn_img = (syn_img + 1.0) / 2.0
        loss = W_loss(syn_img, image1, MSE_loss, upsample, perceptual, 1e-5, 1e-5, M_blur, M_blur) + Mst_loss(syn_img, image2, upsample, 5e-7, MSE_loss, style, (1-M_blur))
        loss.backward()
        w_opt.step()
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {}".format(e + 1, loss.item()))
            save_image(syn_img.clamp(0, 1), "save_images/image2stylegan_v2/local_style_transfer/local_style_transfer{}.png".format(e + 1))

    for e in range(args.w_epochs, args.w_epochs + args.n_epochs):
        n_opt.zero_grad()
        syn_img = g_synthesis(w, noise_list)
        syn_img = (syn_img + 1.0) / 2.0
        loss = Mkn_loss(syn_img, image1, syn_img, MSE_loss, 1e-5, 1e-5, M_blur)
        loss.backward()
        n_opt.step()
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {}".format(e + 1, loss.item()))
            save_image(syn_img.clamp(0, 1), "save_images/image2stylegan_v2/local_style_transfer/local_style_transfer{}.png".format(e + 1))

    plt.plot(loss_, label='Loss = MSELoss + Perceptual')
    plt.plot(loss_psnr, label='PSNR')
    plt.legend()
    return w, noise_list


img_path = os.path.join(args.images_dir, "0.png")
image1 = image_reader(img_path)
image1 = image1.to(device)
print(image1.shape)

img_path = os.path.join(args.images_dir, "4.png")
image2 = image_reader(img_path)
image2 = image2.to(device)
print(image2.shape)

reconstruct_w, reconstruct_noise = image_reconstruction(image1)
crossover_w, crossover_noise = image_crossover(image1, image2)
print("finished")


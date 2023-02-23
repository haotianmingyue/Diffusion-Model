# 开发者 haotian
# 开发时间: 2023/2/22 18:41
import os.path

import torch
import numpy as np
import torch.optim as optim
import torchvision

from nets.diffusion import GaussianDiffusion, generate_linear_schedule, generate_cosine_schedule
from nets.UNet import UNet
from  utils.utils import get_lr_scheduler
from torch.utils.data import DataLoader
from utils.stroke_dataset import stroke_Dataset

import matplotlib.pyplot as plt


# def p_sam


if __name__ == '__main__':

    Cuda = False

    diffusion_model_path = ''

    channel = 32

    schedule = 'linear'
    num_timsteps = 1000
    schedule_low = 1e-4
    schedule_high = 0.02

    input_shape = (256, 256)
    img_channel = 1

    Init_Epoch = 0
    Epoch = 200
    batch_size = 1

    Init_lr = 2e-4
    Min_lr = Init_lr * 0.01

    optimizer_type = 'adam'
    momentum = 0.9
    weight_decay = 0

    lr_decay_type = 'cos'

    # 每多少轮保存一次
    save_period = 25
    save_dir = 'logs'

    num_workers = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # betas
    if schedule == 'cosine':
        betas = generate_cosine_schedule(num_timsteps)
    elif schedule == 'linear':
        betas = generate_linear_schedule(
            num_timsteps,
            schedule_low * 1000 / num_timsteps,
            schedule_high * 1000 / num_timsteps
        )

    diffusion_model = GaussianDiffusion(UNet(img_channel, channel), input_shape, img_channel, betas=betas)

    # 将训练好的模型导入
    if diffusion_model_path != '':
        model_dict = diffusion_model.state_dict()
        pretrained_dict = torch.load(diffusion_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        diffusion_model.load_state_dict(model_dict)


    if Cuda:
        diffusion_model_train = diffusion_model.train().cuda()
    else:
        diffusion_model_train = diffusion_model.train()

    if True:
        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'adamw': optim.AdamW(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                                 weight_decay=weight_decay),
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        # epoch_step = num_train // batch_size
        # if epoch_step == 0:
        #     raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # minist_train = torchvision.datasets.FashionMNIST(root='E:/BaiduNetdiskDownload/FashionMnist', train=True,
        #                                                  download=True, transform=torchvision.transforms.ToTensor())
        # one_minist_data = list()
        # for data in minist_train:
        #     if data[1] == 1:
        #         one_minist_data.append(data)
        #
        # # train_dataset = ''
        #
        # gen = DataLoader(one_minist_data, shuffle=True, batch_size=batch_size, drop_last=True)

        root = 'E:/PythonPPP/pythonTest/test'
        # root = "E:/PythonPPP/pythonTest/Image/HenShan"
        img = stroke_Dataset(root)
        trainDataLoader = DataLoader(img, batch_size=batch_size,
                                     shuffle=False, num_workers=1)

        # 开始训练
        for epoch in range(Init_Epoch, Epoch):
            total_loss = 0
            for idx, images in enumerate(trainDataLoader):
                if Cuda:
                    images = images[0].float().cuda()
                else:
                    images = images[0].float()

                optimizer.zero_grad()
                difussion_loss = torch.mean(diffusion_model_train(images))
                difussion_loss.backward()
                optimizer.step()
                total_loss += difussion_loss.item()

            print(f"epoch{epoch}  loss", total_loss)

            if (epoch + 1) % save_period == 0:
                torch.save(diffusion_model_train.state_dict(), os.path.join(save_dir, f'Diffusion_Epoch{epoch}-loss{total_loss}.pth'))
                fig, ax = plt.subplots()
                # x = torch.randn((1, 1, 256, 256), dtype=torch.float)
                y = diffusion_model.sample(1, 'cuda', use_ema=None)
                ax.imshow(y[0][0])
                plt.show()
            total_loss = 0




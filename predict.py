# 开发者 haotian
# 开发时间: 2023/2/22 20:41

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

# from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
#                   generate_linear_schedule)
from DDPM.nets.UNet import UNet
from DDPM.nets.diffusion import GaussianDiffusion, generate_linear_schedule, generate_cosine_schedule
# from utils.utils import postprocess_output


class Diffusion(object):
    _defaults = {
        # -----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        # -----------------------------------------------#
        "model_path": './DDPM/model_data/Diffusion_Epoch474-loss0.007903145626187325.pth',
        # -----------------------------------------------#
        #   卷积通道数的设置
        # -----------------------------------------------#
        "channel": 32,
        # -----------------------------------------------#
        #   输入图像大小的设置
        # -----------------------------------------------#
        "input_shape": (256, 256),
        # -----------------------------------------------#
        #   betas相关参数
        # -----------------------------------------------#
        "schedule": "linear",
        "num_timesteps": 1000,
        "schedule_low": 1e-4,
        "schedule_high": 0.02,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": False,
    }

    # ---------------------------------------------------#
    #   初始化Diffusion
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.generate()

        # show_config(**self._defaults)

    def generate(self):
        # ----------------------------------------#
        #   创建Diffusion模型
        # ----------------------------------------#
        if self.schedule == "cosine":
            betas = generate_cosine_schedule(self.num_timesteps)
        else:
            betas = generate_linear_schedule(
                self.num_timesteps,
                self.schedule_low * 1000 / self.num_timesteps,
                self.schedule_high * 1000 / self.num_timesteps,
            )

        self.net = GaussianDiffusion(UNet(1, self.channel), self.input_shape, 1, betas=betas)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        self.betas = betas
        alphas_prod = torch.cumprod(1-betas, 0)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    def p_sample(self, model, x, t, betas, one_minus_alphas_bar_sqrt):
        """从x[T]采样t时刻的重构值"""
        '''
            x: 当前输入图像
            t: 当前时刻
            betas: 固定的数组值
            one_minus_alphas_bar_sqrt: 固定的数组值
        '''
        t = torch.tensor([t])  # t时刻

        coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

        # 得到预测噪音 eps_theta
        eps_theta = model(x, t)

        # 得到均值
        mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

        z = torch.randn_like(x)
        sigma_t = betas[t].sqrt()
        # 得到sample的分布
        sample = mean + sigma_t * z

        return (sample)

    # 从xt恢复x0
    def p_sample_loop(self, model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
        """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
        cur_x = torch.randn(shape)  # 相当于随机输入一个标准正态分布的图像
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            cur_x = self.p_sample(self.net, cur_x, i, self.betas, self.one_minus_alphas_bar_sqrt)
            x_seq.append(cur_x)
        return x_seq


    # ---------------------------------------------------#
    #   Diffusion1x1的图片
    # ---------------------------------------------------#
    def generate_1x1_image(self, save_path):
        with torch.no_grad():
            randn_in = torch.randn((1, 1)).cuda() if self.cuda else torch.randn((1, 1))

            test_images = self.net.sample(1, device=None, use_ema=False)

            print(test_images.shape)
            # test_images = postprocess_output(test_images[0].cpu().data.numpy().transpose(1, 2, 0))

            # Image.fromarray(np.uint8(test_images)).save(save_path)


if __name__ == '__main__':
    ddpm = Diffusion()

    ddpm.generate_1x1_image('./DDPM/results')
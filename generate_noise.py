# 开发者 haotian
# 开发时间: 2023/2/23 9:54

import torch
import matplotlib .pyplot as plt

from DDPM.nets.UNet import UNet
from DDPM.nets.diffusion import GaussianDiffusion, generate_linear_schedule, generate_cosine_schedule

Cuda = True
diffusion_model_path = './DDPM/model_data/Diffusion_Epoch499-loss0.016628040000796318.pth'
channel = 32
schedule = 'linear'
num_timesteps = 1000
schedule_low = 1e-4
schedule_high = 0.02
input_shape = (256, 256)
img_channel = 1

if schedule == "cosine":
    betas = generate_cosine_schedule(num_timesteps)
else:
    betas = generate_linear_schedule(
        num_timesteps,
        schedule_low * 1000 / num_timesteps,
        schedule_high * 1000 / num_timesteps,
    )

net = GaussianDiffusion(UNet(1, channel), input_shape, 1, betas=betas).cuda()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.load_state_dict(torch.load(diffusion_model_path))
net = net.eval()

fig, ax = plt.subplots()
# x = torch.randn((1, 1, 256, 256), dtype=torch.float)

y = net.sample(1, 'cuda', use_ema=None)

print(y.shape)
ax.imshow(y[0][0])

plt.show()


print(y)
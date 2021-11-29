import torch.optim as optim
import torchvision.utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import tqdm.auto
from random import seed
import time
import json

from models import *
import dataset
import sampling

torch.cuda.empty_cache()

# %%

# from google.colab import drive
# drive.mount('/content/drive')
# root_dir = "/content/drive/MyDrive/dissertation data/"
root_dir = "./"

# %%

import logging
logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed(1)
# print(device)


# %%

if __name__=='__main__':
    args = {
        'base_channels': 32,
        'channel_mults': (1, 2, 4, 8),
        'T': 500,
        'beta_schedule': 'cosine',
        'loss-type': 'l2',
        'loss_weight': 'prop-t',  # uniform or prop-t
        'sample_distance': 15,
        'random_slice': False,
        'img_size':(32,32),
        'EPOCHS':750,
        'Batch_Size':1,
        'lr':1e-4,
        'weight_decay':0.0,
    }
    training_dataset = dataset.MRIDataset(root_dir=f'{root_dir}Train/', img_size=args['img_size'],random_slice=args['random_slice'])
    testing_dataset = dataset.MRIDataset(root_dir=f'{root_dir}Test/', img_size=args['img_size'],random_slice=args['random_slice'])
    # testing_dataset = MRIDataset(root_dir='/content/drive/MyDrive/dissertation data/Anomalous/',transform=transform)
    dataset_loader = dataset.cycle(torch.utils.data.DataLoader(training_dataset,
                                                       batch_size=args['Batch_Size'], shuffle=True,
                                                       num_workers=2))

    new = next(dataset_loader)

    # new["image"] = torch.cat((new["image"][:],new["image"][:],new["image"][:]),dim=1)
    print(new["image"].shape)
    plt.rcParams['figure.dpi'] = 100
    plt.grid(False)
    plt.imshow(
        torchvision.utils.make_grid(((new["image"]+1)*127.5).clamp(0,255).to(torch.uint8)).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0),
        cmap='gray')
    # plt.show()
    plt.pause(0.0001)

    # %%

    for i in ['./model/',"./diffusion-videos/",'./diffusion-training-images/']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    # %%

    unet = UNet(args['base_channels'], channel_mults=args['channel_mults'])


    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusion(unet.to(device), args['img_size'], betas, loss_weight=args['loss_weight'],loss_type=args['loss-type'])


    # diffusion.load_state_dict(torch.load(f'{root_dir}diff-params-24-11-100epochs',map_location=device))

    diffusion = diffusion.to(device)

    optimiser = optim.AdamW(diffusion.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    #%%

    def training_outputs(x):
        out = []

        if epoch > 0 and epoch % 100 == 0:
            fig, ax = plt.subplots()
            if epoch % 200 == 0:
                out = diffusion.forward_backward(x, True, args['sample_distance'])
                imgs = [[ax.imshow(torchvision.utils.make_grid(((x+1)*127.5).clamp(0,255).to(torch.uint8), nrow=rowSize).cpu().data.permute(0, 2,
                                                                                                              1).contiguous().permute(
                    2, 1, 0), animated=True)] for x in out]
                # out = out.permute(0,2,3,1)
                ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,
                                                repeat_delay=1000)

                ani.save(
                    f'{root_dir}diffusion-videos/sample-{time.gmtime().tm_mday}-{time.gmtime().tm_mon}-{epoch}epochs.mp4')
                out = torch.cat((x[:rowSize, ...].cpu(), out[-1][:rowSize, ...].cpu()))
                plt.title(
                    f'x,forward_backward with video: {root_dir}diffusion-videos/sample-{time.gmtime().tm_mday}-{time.gmtime().tm_mon}-{args["EPOCHS"]}epochs.mp4')
            else:
                out = torch.cat((x[:rowSize, ...].cpu(),
                                 diffusion.forward_backward(x, False, args['sample_distance'])[:rowSize, ...].cpu()))
                plt.title(f'x,forward_backward-{epoch}epoch')

        elif epoch % 25 == 0:
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(x_t, t)
            out = torch.cat((x[:rowSize, ...].cpu(), temp["sample"][:rowSize, ...].cpu(),
                             temp["pred_x_0"][:rowSize, ...].cpu()))
            plt.title(f'real,sample,prediction x_0-{epoch}epoch')
        else:
            out = torch.cat((x[:rowSize, ...].cpu(), noisy[:rowSize, ...].cpu(), est[:rowSize, ...].cpu(),
                             (est - noisy).square().cpu()[:rowSize, ...]))
            plt.title(f'real,noisy,noise prediction,mse-{epoch}epoch')

        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(torchvision.utils.make_grid(((out+1)*127.5).clamp(0,255).to(torch.uint8), nrow=rowSize).cpu().data.permute(0, 2,
                                                                                                1).contiguous().permute(
            2, 1, 0),
            cmap='gray')

        plt.savefig(f'./diffusion-training-images/{epoch}-epoch')

    # %%

    losses = []
    tqdm_epoch = tqdm.trange(args['EPOCHS'])
    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []
        for i in range(100 // args['Batch_Size']):
            data = next(dataset_loader)
            x = data["image"]
            x = x.to(device)
            loss, noisy, est = diffusion(x)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            mean_loss.append(loss.data.cpu())
            tqdm_epoch.set_postfix({"imgs trained": (1 + i) * args['Batch_Size'] + epoch * 100, "loss": loss.data.cpu() ,'last epoch mean loss': losses[-1] if len(losses)>0 else 0})
            if epoch % 5 == 0 and i == 0:
                rowSize = min(8,args['Batch_Size'])
                training_outputs(x)

        losses.append(np.mean(mean_loss))

    #%%

    try:
        os.makedirs(f'./model/diff-params-{time.gmtime().tm_mday}-{time.gmtime().tm_mon}-{args["EPOCHS"]}epochs')
    except OSError:
        pass

    torch.save(diffusion.state_dict(),f'{root_dir}model/diff-params-{time.gmtime().tm_mday}-{time.gmtime().tm_mon}-{args["EPOCHS"]}epochs/params')


    with open(f'./model/diff-params-{time.gmtime().tm_mday}-{time.gmtime().tm_mon}-{args["EPOCHS"]}epochs/args.json','w') as file:
        json.dump(args,file)

    # %%
    sampling.testing(testing_dataset,diffusion, args=args,device=device)

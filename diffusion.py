import time
import json
import os
import sys
from random import seed

from torch import optim
import torch
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import tqdm.auto

from models import UNet, GaussianDiffusion, get_beta_schedule
import dataset


torch.cuda.empty_cache()

# from google.colab import drive
# drive.mount('/content/drive')
# ROOT_DIR = "/content/drive/MyDrive/dissertation data/"
ROOT_DIR = "./"


def output_img(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img)[:row_size,...], nrow=row_size).cpu().data.permute(0, 2,1).contiguous().permute(2, 1, 0)


def train(training_dataset_loader, testing_dataset_loader, args):
    unet = UNet(args['img_size'][0],args['base_channels'], channel_mults=args['channel_mults'])

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusion(unet.to(device), args['img_size'], betas, loss_weight=args['loss_weight'],
                                  loss_type=args['loss-type'])

    diffusion = diffusion.to(device)

    optimiser = optim.AdamW(diffusion.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    losses = []
    tqdm_epoch = tqdm.trange(args['EPOCHS'])
    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []
        for i in range(100 // args['Batch_Size']):
            data = next(training_dataset_loader)
            x = data["image"]
            x = x.to(device)
            loss, noisy, est = diffusion(x)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            mean_loss.append(loss.data.cpu())
            print(f"imgs trained: {(1 + i) * args['Batch_Size'] + epoch * 100}, loss: {loss.data.cpu():.2f} ,"
                  f"'last epoch mean loss': {losses[-1] if len(losses)>0 else 0:.4f}\r")
            # tqdm_epoch.set_postfix({"imgs trained": (1 + i) * args['Batch_Size'] + epoch * 100, "loss": loss.data.cpu() ,'last epoch mean loss': losses[-1] if len(losses)>0 else 0})
            if epoch % 5 == 0 and i == 0:
                row_size = min(8,args['Batch_Size'])
                training_outputs(diffusion, x, est, noisy, epoch, row_size, save_imgs=args['save_imgs'],
                                 save_vids=args['save_vids'])

        losses.append(np.mean(mean_loss))

    save(diffusion=diffusion,args=args)

    testing(testing_dataset_loader,diffusion, args=args,device=device)


def testing(testing_dataset_loader, diffusion, args, device=torch.device('cpu'), use_ddim=False):

    plt.rcParams['figure.dpi'] = 200
    # for i in [args['sample_distance'], args['T'] / 4, None]:
    for i in [*range(1,args['sample_distance']),args['T']]:
        data = next(testing_dataset_loader)
        x = data["image"]
        x = x.to(device)
        row_size = min(5,args['Batch_Size'])

        fig, ax = plt.subplots()
        out = diffusion.forward_backward(x, see_whole_sequence=True, use_ddim=use_ddim, t_distance=i)
        imgs = [[ax.imshow(output_img(x,row_size), animated=True)] for x in out]
        ani = animation.ArtistAnimation(fig, imgs, interval=200, blit=True,
                                        repeat_delay=1000)

        ani.save(f'{ROOT_DIR}diffusion-videos/test-set-DAY={time.gmtime().tm_mday}-MONTH={time.gmtime().tm_mon}'
                 f't={i}-ARGS={args["arg_num"]}.mp4')
        plt.title(
            f'x,forward_backward with video: {ROOT_DIR}diffusion-videos/test-set-{time.gmtime().tm_mday}-{time.gmtime().tm_mon}-{i}-{args["EPOCHS"]}epochs.mp4')
        print("saved")


def save(diffusion,args):
    try:
        os.makedirs(f'./model/diff-params-DAY={time.gmtime().tm_mday}-MONTH={time.gmtime().tm_mon}'
                     f'ARGS={args["arg_num"]}')
    except OSError:
        pass

    torch.save(diffusion.state_dict(),f'{ROOT_DIR}model/diff-params-DAY={time.gmtime().tm_mday}-MONTH={time.gmtime().tm_mon}'
                     f'ARGS={args["arg_num"]}/params')

    with open(f'./model/diff-params-DAY={time.gmtime().tm_mday}-MONTH={time.gmtime().tm_mon}'
                     f'ARGS={args["arg_num"]}/args.json','w') as file:
        json.dump(args,file)


def init_datasets(args):
    training_dataset = dataset.MRIDataset(ROOT_DIR=f'{ROOT_DIR}Train/', img_size=args['img_size'],random_slice=args['random_slice'])
    testing_dataset = dataset.MRIDataset(ROOT_DIR=f'{ROOT_DIR}Test/', img_size=args['img_size'],random_slice=args['random_slice'])
    # testing_dataset = MRIDataset(ROOT_DIR='/content/drive/MyDrive/dissertation data/Anomalous/',transform=transform)
    return training_dataset,testing_dataset


def init_dataset_loader(mri_dataset,args):
    dataset_loader = dataset.cycle(torch.utils.data.DataLoader(mri_dataset,
                                                       batch_size=args['Batch_Size'], shuffle=True,
                                                       num_workers=5))

    new = next(dataset_loader)

    # convert to rgb:
    # new["image"] = torch.cat((new["image"][:],new["image"][:],new["image"][:]),dim=1)
    print(new["image"].shape)
    plt.rcParams['figure.dpi'] = 100
    plt.grid(False)
    plt.imshow(output_img(new["image"]),cmap='gray')
    # plt.show()
    plt.pause(0.0001)
    return dataset_loader


def training_outputs(diffusion, x, est,noisy, epoch, row_size, save_imgs=False, save_vids=False):
    if save_imgs:
        if epoch % 25 == 0:
            # for a given t, output x_0, & prediction of x_(t-1), and x_0
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(x_t, t)
            out = torch.cat((x[:row_size, ...].cpu(), temp["sample"][:row_size, ...].cpu(),
                             temp["pred_x_0"][:row_size, ...].cpu()))
            plt.title(f'real,sample,prediction x_0-{epoch}epoch')
        else:
            # for a given t, output x_0, x_t, & prediction of noise in x_t & MSE
            out = torch.cat((x[:row_size, ...].cpu(), noisy[:row_size, ...].cpu(), est[:row_size, ...].cpu(),
                             (est - noisy).square().cpu()[:row_size, ...]))
            plt.title(f'real,noisy,noise prediction,mse-{epoch}epoch')

        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(output_img(out,row_size),cmap='gray')

        plt.savefig(f'./diffusion-training-images/ARGS={args["arg_num"]}-EPOCH={epoch}.png')

    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 100 == 0:
            plt.rcParams['figure.dpi'] = 200
            out = diffusion.forward_backward(x, True, args['sample_distance'])
            imgs = [[ax.imshow(output_img(x,row_size),animated=True)] for x in out]
            ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True,
                                            repeat_delay=1000)

            ani.save(f'{ROOT_DIR}diffusion-videos/sample-DAY={time.gmtime().tm_mday}-MONTH={time.gmtime().tm_mon}'
                     f'ARGS={args["arg_num"]}-EPOCH={epoch}.mp4')

    plt.close('all')


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)

    for i in ['./model/',"./diffusion-videos/",'./diffusion-training-images/']:
        try:
            os.makedirs(i)
        except OSError:
            pass


    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        files = os.listdir(f'{ROOT_DIR}test_args')

    for file in files:
        if file[-5:]==".json":
            with open(f'{ROOT_DIR}test_args/{file}', 'r') as f:
                args = json.load(f)
            args['arg_num'] = file[-6]
            training_dataset, testing_dataset = init_datasets(args)
            training_dataset_loader = init_dataset_loader(training_dataset, args)
            testing_dataset_loader = init_dataset_loader(testing_dataset, args)
            # load, pass args
            train(training_dataset_loader,testing_dataset_loader,args)

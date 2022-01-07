import random

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import os
import matplotlib.animation as animation
import dataset
import torch
import sys
import time
from diffusion import init_dataset_loader, output_img
from models import UNet, GaussianDiffusion, get_beta_schedule

np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))

def detect(image,diffusion):

    #
    pass

def heatmap(real:torch.Tensor,recon:torch.Tensor):
    diff = real - recon

    pass

if __name__ == "__main__":

    if len(sys.argv[1:]) > 0:
        params = sys.argv[1:]
    else:
        params = os.listdir("./model")
    if ".DS_Store" in params:
        params.remove(".DS_Store")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = './Cancerous Dataset/EdinburghDataset/Anomalous/raw'


    for param in params:
        param = "diff-params-ARGS=4-DAY=17-MONTH=12"
        output = torch.load(f'./model/{param}/params-final.pt', map_location=device)
        if "args" in output:
            args = output["args"]
        else:
            args = {
                  "img_size": [256,256],
                  "Batch_Size": 1,
                  "EPOCHS": 10000,
                  "T": 1000,
                  "base_channels": 128,
                  "beta_schedule": "linear",
                  "channel_mults": "",
                  "loss-type": "l2",
                  "loss_weight": "none",
                  "lr": 0.0001,
                  "random_slice": False,
                  "sample_distance": 40,
                  "weight_decay": 0.0,
                  "save_imgs": False,
                  "save_vids": True,
                  "train_start": True,
                  "arg_num":4
                }


        unet = UNet(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

        betas = get_beta_schedule(args['T'], args['beta_schedule'])

        diff = GaussianDiffusion(unet.to(device), args['img_size'], betas, loss_weight=args['loss_weight'],
                                      loss_type=args[
                                          'loss-type'])

        diff.load_state_dict(output['model_state_dict'])

        AnoDataset = dataset.AnomalousMRIDataset(ROOT_DIR=f'{dataset_path}', img_size=args['img_size'],
                                              slice_selection="random")
        loader = init_dataset_loader(AnoDataset,args)
        try:
            os.makedirs(f'./diffusion-videos/Anomalous')
        except OSError:
            pass

        FILES = 0
        for epoch in range(1):
            for i in range(len(AnoDataset)//5):
                new = next(loader)
                print(new["filenames"])
                output = diff.forward_backward(new["image"][0],see_whole_sequence=True,t_distance=30)
                fig, ax = plt.subplots()

                plt.rcParams['figure.dpi'] = 200

                imgs = [[ax.imshow(output_img(x, 1), animated=True)] for x in output]
                ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True,
                                                repeat_delay=1000)

                ani.save(
                    f'./diffusion-videos/Anomalous/{new["filenames"][0][-9:-4]}-ARGS={args["arg_num"]}-'
                    f'{random.randint(0,100000)}.mp4')

                plt.close('all')


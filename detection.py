import os
import random
import sys

import matplotlib.animation as animation
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset
from diffusion_training import init_dataset_loader, output_img
from models import GaussianDiffusion, get_beta_schedule, UNet

np.set_printoptions(
        edgeitems=30, linewidth=100000,
        formatter=dict(float=lambda x: "%.3g" % x)
        )


def detect(image, diffusion):
    pass


def heatmap(real: torch.Tensor, recon: torch.Tensor):
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
    dataset_path = './CancerousDataset/EdinburghDataset/Anomalous/raw'

    for param in params:
        try:
            # if checkpointed version
            # output = torch.load(f'./model/{param}/checkpoint/diff_epoch=0.pt', map_location=device)
            output = torch.load(f'./model/{param}/params-final.pt', map_location=device)
        except FileNotFoundError:
            continue
        if "args" in output:
            args = output["args"]
        else:
            args = {
                "img_size":        [256, 256],
                "Batch_Size":      1,
                "EPOCHS":          10000,
                "T":               1000,
                "base_channels":   128,
                "beta_schedule":   "linear",
                "channel_mults":   "",
                "loss-type":       "l2",
                "loss_weight":     "none",
                "lr":              0.0001,
                "random_slice":    False,
                "sample_distance": 40,
                "weight_decay":    0.0,
                "save_imgs":       False,
                "save_vids":       True,
                "train_start":     True,
                "arg_num":         4
                }

        print(f"args{args['arg_num']}")
        unet = UNet(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

        betas = get_beta_schedule(args['T'], args['beta_schedule'])

        diff = GaussianDiffusion(
                args['img_size'], betas, loss_weight=args['loss_weight'],
                loss_type=args['loss-type']
                )

        unet.load_state_dict(output["ema"])
        unet.to(device)

        AnoDataset = dataset.AnomalousMRIDataset(
                ROOT_DIR=f'{dataset_path}', img_size=args['img_size'],
                slice_selection="random"
                )
        loader = init_dataset_loader(AnoDataset, args)

        try:
            os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous')
        except OSError:
            pass

        for epoch in range(1):
            for i in range(len(AnoDataset)):
                new = next(loader)
                img = new["image"][0].to(device)
                timestep = random.randint(30, args["sample_distance"] * 2)
                output = diff.forward_backward(unet, img, see_whole_sequence="whole", t_distance=timestep)
                fig, ax = plt.subplots()

                plt.rcParams['figure.dpi'] = 200

                imgs = [[ax.imshow(output_img(x, 1), animated=True)] for x in output]
                ani = animation.ArtistAnimation(
                        fig, imgs, interval=100, blit=True,
                        repeat_delay=1000
                        )

                try:
                    os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}')
                except OSError:
                    pass

                temp = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}')

                output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/t={timestep}-attemp' \
                              f't={len(temp) + 1}.mp4'
                if "slices" in new:
                    output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/' \
                                  f'{new["filenames"][0][-9:-4]}/slices={new["slices"].tolist()}-t={timestep}-attemp' \
                                  f't={len(temp) + 1}.mp4'
                ani.save(output_name)

                plt.close('all')

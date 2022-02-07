import json
import os
import random
import sys

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from matplotlib import animation

import dataset
from diffusion_training import defaultdict_from_json, init_dataset_loader, output_img
# from models import GaussianDiffusion, get_beta_schedule, UNet
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule, mean_flat
from UNet import UNetModel



def heatmap(real: torch.Tensor, recon: torch.Tensor, filename):
    mse = mean_flat((recon - real).square())
    mse = mse.cpu().numpy()
    plt.imsave(filename, mse, cmap="YlOrRd")



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
            try:
                with open(f'./test_args/args{param[17:]}.json', 'r') as f:
                    args = json.load(f)
                args['arg_num'] = param[17:]
                args = defaultdict_from_json(args)
            except FileNotFoundError:
                print(f"args{param[17:]} doesn't exist for {param}")
                raise

        print(f"args{args['arg_num']}")
        unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

        betas = get_beta_schedule(args['T'], args['beta_schedule'])

        diff = GaussianDiffusionModel(
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

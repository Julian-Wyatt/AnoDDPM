import json
import os
import random
import sys

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import dataset
import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel

DATASET_PATH = './DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(param, use_checkpoint):
    """
    loads the most recent (non-corrupted) checkpoint or the final model
    :param param: args number
    :param use_checkpoint: checkpointed or final model
    :return:
    """
    if not use_checkpoint:
        return torch.load(f'./model/diff-params-ARGS={param}/params-final.pt', map_location=device)
    else:
        checkpoints = os.listdir(f'./model/diff-params-ARGS={param}/checkpoint')
        checkpoints.sort(reverse=True)
        for i in checkpoints:
            try:
                file_dir = f"./model/diff-params-ARGS={param}/checkpoint/{i}"
                loaded_model = torch.load(file_dir, map_location=device)
                break
            except RuntimeError:
                continue
        return loaded_model


def load_parameters():
    """
    Loads the trained parameters for the detection model
    :return:
    """
    if len(sys.argv[1:]) > 0:
        params = sys.argv[1:]
    else:
        params = os.listdir("./model")
    if ".DS_Store" in params:
        params.remove(".DS_Store")

    if params[0] == "CHECKPOINT":
        use_checkpoint = True
        params = params[1:]
    else:
        use_checkpoint = False

    print(params)
    for param in params:
        if param.isnumeric():
            output = load_checkpoint(param, use_checkpoint)
        elif param[:4] == "args" and param[-5:] == ".json":
            output = load_checkpoint(param[4:-5], use_checkpoint)
        elif param[:4] == "args":
            output = load_checkpoint(param[4:], use_checkpoint)
        else:
            raise ValueError(f"Unsupported input {param}")

        if "args" in output:
            args = output["args"]
        else:
            try:
                with open(f'./test_args/args{param[17:]}.json', 'r') as f:
                    args = json.load(f)
                args['arg_num'] = param[17:]
                args = defaultdict_from_json(args)
            except FileNotFoundError:
                raise ValueError(f"args{param[17:]} doesn't exist for {param}")

        if "noise_fn" not in args:
            args["noise_fn"] = "gauss"

        return args, output


def anomalous_validation():
    """
    Iterates over 4 anomalous slices for each Volume, returning diffused video for it,
    the heatmap of that & detection method (A&B) or C
    :return:
    """
    args, output = load_parameters()
    print(f"args{args['arg_num']}")
    unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"]
            )

    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    AnoDataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=True
            )
    loader = dataset.init_dataset_loader(AnoDataset, args)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass

    # make folder for each anomalous volume
    for i in AnoDataset.slices.keys():
        try:
            os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{i}')
        except OSError:
            pass

    dice_data = []
    for i in range(len(AnoDataset)):
        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], AnoDataset)
        img_mask = img_mask.to(device)

        for slice_number in range(4):
            try:
                os.makedirs(
                        f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                        f'{new["slices"][slice_number].numpy()[0]}'
                        )
            except OSError:
                pass
            if args["noise_fn"] == "gauss":
                timestep = random.randint(int(args["sample_distance"] * 0.3), int(args["sample_distance"] * 0.8))
            else:
                timestep = random.randint(int(args["sample_distance"] * 0.1), int(args["sample_distance"] * 0.6))

            output = diff.forward_backward(
                    unet, img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                    see_whole_sequence="whole",
                    t_distance=timestep
                    )

            fig, ax = plt.subplots()
            plt.axis('off')
            imgs = [[ax.imshow(gridify_output(x, 5), animated=True)] for x in output]
            ani = animation.ArtistAnimation(
                    fig, imgs, interval=50, blit=True,
                    repeat_delay=1000
                    )
            temp = os.listdir(
                    f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                    f'{new["slices"][slice_number].numpy()[0]}'
                    )

            output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/' \
                          f'{new["filenames"][0][-9:-4]}/{new["slices"][slice_number].numpy()[0]}/t={timestep}-attemp' \
                          f't={len(temp) + 1}'
            ani.save(output_name + ".mp4")

            dice_data.append(
                    evaluation.heatmap(
                            img[slice_number, ...].reshape(1, 1, *args["img_size"]), output[-1].to(device),
                            img_mask[new["slices"][slice_number].numpy()[0], ...].reshape(1, 1, *args["img_size"]),
                            output_name + ".png"
                            )
                    )

            plt.close('all')

            if args["noise_fn"] == "gauss":
                dice = diff.detection_B(
                        unet, img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice_number].numpy()[0]),
                        img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"]), "gauss",
                        total_avg=3
                        )
                dice_data.append(dice)
            elif args["noise_fn"] == "simplex":
                dice = diff.detection_B(
                        unet, img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice_number].numpy()[0]),
                        img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"]), "octave",
                        total_avg=3
                        )
                dice_data.append(dice)
            elif args["noise_fn"] == "simplex_randParam":
                diff.detection_A(
                        unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice].numpy()[0]),
                        img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"])
                        )
                dice = diff.detection_B(
                        unet, img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice_number].numpy()[0]),
                        img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"]), "octave",
                        total_avg=3
                        )
                dice_data.append(dice)

    print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")




if __name__ == "__main__":
    anomalous_validation()

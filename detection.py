import json
import os
import random
import sys

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation

import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel

DATASET_PATH = './CancerousDataset/EdinburghDataset/Anomalous-T1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def heatmap(real: torch.Tensor, recon: torch.Tensor, mask, filename):
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = mse > 0
    mse_threshold = (mse_threshold.float() * 2) - 1
    output = torch.cat((real, recon, mse, mse_threshold, mask))
    # plt.imshow(gridify_output(mse, 3)[..., 0], cmap="YlOrRd")
    # plt.colorbar()
    plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
    plt.axis('off')
    plt.savefig(filename)
    plt.clf()



def load_parameters():
    if len(sys.argv[1:]) > 0:
        params = sys.argv[1:]
    else:
        params = os.listdir("./model")
    if ".DS_Store" in params:
        params.remove(".DS_Store")
    print(params)
    for param in params:
        if param.isnumeric():
            output = torch.load(f'./model/diff-params-ARGS={param}/params-final.pt', map_location=device)
        elif param[:4] == "args" and param[-5:] == ".json":
            output = torch.load(f'./model/diff-params-ARGS={param[4:-5]}/params-final.pt', map_location=device)
        elif param[:4] == "args":
            output = torch.load(f'./model/diff-params-ARGS={param[4:]}/params-final.pt', map_location=device)
        else:
            # if checkpointed version
            # output = torch.load(f'./model/{param}/checkpoint/diff_epoch=0.pt', map_location=device)
            output = torch.load(f'./model/{param}/params-final.pt', map_location=device)

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

        if "noise_fn" not in args:
            args["noise_fn"] = "gauss"

        return args, output


def anomalous_validation():
    """
    Iterates over each anomalous slice, returning forward backward's for it,
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
            slice_selection="iterateKnown", resized=True
            )
    loader = dataset.init_dataset_loader(AnoDataset, args)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass

    for i in AnoDataset.slices.keys():
        try:
            os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{i}')
        except OSError:
            pass

    for i in range(len(AnoDataset)):
        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], AnoDataset)
        img_mask = img_mask.to(device)
        for slice in range(0, img.shape[0], 5):
            try:
                os.makedirs(
                        f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                        f'{new["slices"][slice].numpy()[0]}'
                        )
            except OSError:
                pass
            timestep = random.randint(args["sample_distance"] // 2, args["sample_distance"])

            output = diff.forward_backward(
                    unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                    see_whole_sequence="whole",
                    t_distance=timestep
                    )

            fig, ax = plt.subplots()
            plt.axis('off')
            imgs = [[ax.imshow(gridify_output(x, 5), animated=True)] for x in output]
            ani = animation.ArtistAnimation(
                    fig, imgs, interval=100, blit=True,
                    repeat_delay=1000
                    )
            temp = os.listdir(
                    f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                    f'{new["slices"][slice].numpy()[0]}'
                    )

            output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/' \
                          f'{new["filenames"][0][-9:-4]}/{new["slices"][slice].numpy()[0]}/t={timestep}-attemp' \
                          f't={len(temp) + 1}'
            ani.save(output_name + ".mp4")

            heatmap(
                    img[slice, ...].reshape(1, 1, *args["img_size"]), output[-1].to(device),
                    img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"]),
                    output_name + ".png"
                    )

            plt.close('all')

            if args["noise_fn"] == "gauss":
                diff.detection_B(
                        unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice].numpy()[0]),
                        img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"]), "gauss"
                        )
            elif args["noise_fn"] == "simplex":
                diff.detection_B(
                        unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice].numpy()[0]),
                        img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"]), "octave"
                        )
            elif args["noise_fn"] == "simplex_randParam":
                diff.detection_A(
                        unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice].numpy()[0]),
                        img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"])
                        )
                diff.detection_B(
                        unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice].numpy()[0]),
                        img_mask[new["slices"][slice].numpy()[0], ...].reshape(1, 1, *args["img_size"]), "octave"
                        )




if __name__ == "__main__":
    anomalous_validation()

    # print(f"args{args['arg_num']}")
    # unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])
    #
    # betas = get_beta_schedule(args['T'], args['beta_schedule'])
    #
    # diff = GaussianDiffusionModel(
    #         args['img_size'], betas, loss_weight=args['loss_weight'],
    #         loss_type=args['loss-type'], noise=args["noise_fn"]
    #         )
    #
    # unet.load_state_dict(output["ema"])
    # unet.to(device)
    #
    # AnoDataset = dataset.AnomalousMRIDataset(
    #         ROOT_DIR=f'{dataset_path}', img_size=args['img_size'],
    #         slice_selection="random", resized=True
    #         )
    # loader = init_dataset_loader(AnoDataset, args)
    #
    # try:
    #     os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous')
    # except OSError:
    #     pass
    #
    # for epoch in range(5):
    #     for i in range(len(AnoDataset)):
    #         new = next(loader)
    #         img = new["image"].to(device)
    #         timestep = random.randint(60, args["sample_distance"])
    #
    #         output = diff.forward_backward(unet, img, see_whole_sequence="whole", t_distance=timestep)
    #         fig, ax = plt.subplots()
    #
    #         plt.rcParams['figure.dpi'] = 200
    #
    #         imgs = [[ax.imshow(output_img(x, 1), animated=True)] for x in output]
    #         ani = animation.ArtistAnimation(
    #                 fig, imgs, interval=100, blit=True,
    #                 repeat_delay=1000
    #                 )
    #
    #         try:
    #             os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}')
    #         except OSError:
    #             pass
    #
    #         temp = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}')
    #
    #         output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/t={timestep}-attemp' \
    #                       f't={len(temp) + 1}.mp4'
    #         if "slices" in new:
    #             output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/' \
    #                           f'{new["filenames"][0][-9:-4]}/slices={new["slices"].tolist()}-t={timestep}-attemp' \
    #                           f't={len(temp) + 1}'
    #         ani.save(output_name + ".mp4")
    #
    #         heatmap(img, output[-1].to(device), output_name + ".png")
    #
    #         plt.close('all')

import dataset
from helpers import load_parameters


def output_denoise_sequence(sequence: list, filename, masks, predictions):
    """
    sequence is ideally [[t lots of images],[t lots of images],[t lots of images],[t lots of images]]
    or sequence is [t lots of images]
    :param sequence:
    :return:
    """

    if len(sequence) > 10:
        sequence = [sequence]

    relevant_elements_forward = np.linspace(0, len(sequence[0]) // 2, 6).astype(np.int32)
    relevant_elements_backward = (-1 * relevant_elements_forward[-2::-1]) - 1

    relevant_elements = np.append(relevant_elements_forward, relevant_elements_backward)

    output = torch.empty(13 * len(sequence), 1, 256, 256)

    for j, new_sequence in enumerate(sequence):

        for i, val in enumerate(relevant_elements):
            output[13 * j + i] = new_sequence[val]
        output[13 * (j + 1) - 1] = masks[j]
        output[13 * (j + 1) - 2] = predictions[j]

    fig, subplots = plt.subplots(
            len(sequence), 13, sharex=True, sharey=True, constrained_layout=False, figsize=(13, len(sequence)),
            gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=False
            )

    for brain in range(len(sequence)):
        for noise in range(13):
            # subplots[brain][noise].imshow(output[12 * brain + noise].reshape(256, 256, 1), cmap="gray")
            subplots[brain][noise].imshow(
                    output[13 * brain + noise].reshape(*output[0].shape[-2:]).cpu().numpy(),
                    cmap="gray"
                    )
            subplots[brain][noise].tick_params(
                    top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False
                    )
            # subplots[brain][noise].axis('off')
    for i in range(6):

        subplots[0][i].set_xlabel(f"$x_{{{relevant_elements[i]}}}$", fontsize=6)
        subplots[0][i].xaxis.set_label_position("top")

    for i in range(6, 11):
        subplots[0][i].set_xlabel(f"$x_{{{relevant_elements_forward[::-1][1:][i - 6]}}}$", fontsize=6)
        subplots[0][i].xaxis.set_label_position("top")

    subplots[0][-2].set_xlabel(f"Prediction", fontsize=6)
    subplots[0][-2].xaxis.set_label_position("top")
    subplots[0][-1].set_xlabel(f"Ground Truth", fontsize=6)
    subplots[0][-1].xaxis.set_label_position("top")

    plt.savefig(filename)


def output_masked_comparison(sequence, filename):
    """
    sequence is ideally [[x_0,recon,mse,threshold_mse,ground_truth]*4] where
    [x_0,recon,mse,threshold_mse,ground_truth] is a single (5,1,256,256) torch tensor
    or sequence is [x_0,recon,mse,threshold_mse,ground_truth]
    :param sequence:
    :return:
    """
    if type(sequence) == torch.tensor:
        sequence = [sequence]

    fig, subplots = plt.subplots(
            len(sequence), 6, sharex=True, sharey=True, constrained_layout=False, figsize=(6, len(sequence)),
            squeeze=False,
            gridspec_kw={'wspace': 0, 'hspace': 0}
            )
    plt.tick_params(
            top=False, bottom=False, left=False, right=False,
            labelleft=False, labelbottom=False
            )
    for i, brain in enumerate(sequence):
        for plot in range(brain.shape[0]):
            if plot == 3:
                subplots[i][plot].imshow(brain[plot].reshape(*brain.shape[-2:]).cpu().numpy(), cmap="hot")

            else:
                subplots[i][plot].imshow(brain[plot].reshape(*brain.shape[-2:]).cpu().numpy(), cmap="gray")
            subplots[i][plot].tick_params(
                    top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False
                    )

    for i, val in enumerate(["$x_0$", "$x_{250}$", "Reconstruction", "Square Error", "Prediction", "Ground Truth"]):
        subplots[0][i].set_xlabel(f"{val}", fontsize=6)
        subplots[0][i].xaxis.set_label_position("top")

    plt.savefig(filename)


def make_prediction(real, recon, mask, x_t):
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = mse > 0
    mse_threshold = (mse_threshold.float() * 2) - 1
    return torch.cat((real, x_t, recon, mse, mse_threshold, mask)), mse_threshold


def make_all_outputs():
    args, output = load_parameters(device)

    unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"]
            )

    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )

    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 1000

    if args["noise_fn"] == "gauss":
        t_distance = 500
    else:
        t_distance = 250

    for i in [f'./final-outputs/', f'./final-outputs/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    for i in range(20):

        predictions = []
        sequences = []
        masks = []
        mse_thresholds = []

        rows = np.random.choice([1, 2, 3, 4, 5, 8], p=[0.3, 0.2, 0.1, 0.2, 0.15, 0.05])
        print(f"epoch {i}, rows @ epoch: {rows}")
        for k in range(rows):
            new = next(loader)
            img = new["image"].to(device)
            img = img.reshape(img.shape[1], 1, *args["img_size"])
            img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
            img_mask = img_mask.to(device)

            slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
            output = diff.forward_backward(
                    unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                    see_whole_sequence="whole",
                    # t_distance=5, denoise_fn=args["noise_fn"]
                    t_distance=t_distance, denoise_fn=args["noise_fn"]
                    )

            output_images, mse_threshold = make_prediction(
                    img[slice, ...].reshape(1, 1, *args["img_size"]), output[-1].to(device),
                    img_mask[slice, ...].reshape(1, 1, *args["img_size"]), output[t_distance // 2].to(device)
                    )
            predictions.append(
                    output_images
                    )

            mse_thresholds.append(mse_threshold)
            masks.append(img_mask[slice, ...].reshape(1, 1, *args["img_size"]))
            sequences.append(output)

        temp = os.listdir(f"./final-outputs/ARGS={args['arg_num']}")
        output_masked_comparison(
                predictions, f'./final-outputs/ARGS={args["arg_num"]}/attempt'
                             f'={len(temp) + 1}-predictions.png'
                )
        output_denoise_sequence(
                sequences, f'./final-outputs/ARGS={args["arg_num"]}/attempt'
                           f'={len(temp) + 1}-sequence.png', masks, mse_thresholds
                )
        plt.close('all')


def make_varying_frequency_outputs():
    args, output = load_parameters(device)

    unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"]
            )

    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )

    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 1000

    for i in [f'./final-outputs/', f'./final-outputs/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    for i in range(22):

        print(f"epoch {i}")

        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        img_mask = img_mask.to(device)

        slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        output = diff.detection_A_fixedT(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]), args,
                img_mask[slice, ...].reshape(1, 1, *args["img_size"])
                )

        fig, subplots = plt.subplots(
                6, 6, sharex=True, sharey=True, constrained_layout=False, figsize=(6, 6),
                gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=False
                )

        tempplot = fig.add_subplot(111, frameon=False)

        for freq in range(6):
            for out_img in range(6):
                if out_img == 3:
                    subplots[freq][out_img].imshow(
                            output[6 * freq + out_img].reshape(*output.shape[-2:]).cpu().numpy(),
                            cmap="hot"
                            )
                else:
                    # subplots[brain][noise].imshow(output[12 * brain + noise].reshape(256, 256, 1), cmap="gray")
                    subplots[freq][out_img].imshow(
                            output[6 * freq + out_img].reshape(*output.shape[-2:]).cpu().numpy(),
                            cmap="gray"
                            )
                subplots[freq][out_img].tick_params(
                        top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False
                        )

        for i, val in enumerate(["$x_0$", "$x_{250}$", "Reconstruction", "Square Error", "Prediction", "Ground Truth"]):
            subplots[0][i].set_xlabel(f"{val}", fontsize=6)
            subplots[0][i].xaxis.set_label_position("top")

        for i in range(6):
            subplots[i][0].set_ylabel(f"$2^{i + 1}={2 ** (i + 1)}$", fontsize=6)
            subplots[i][0].yaxis.set_label_position("left")

        plt.tick_params(labelcolor='none', which='both', top=False, left=False, bottom=False, right=False)
        plt.ylabel("Starting Frequency\n", fontsize=6)

        temp = os.listdir(f"./final-outputs/ARGS={args['arg_num']}")

        plt.savefig(
                f'./final-outputs/ARGS={args["arg_num"]}/{new["filenames"][0][-9:-4]}-frequency-attempt'
                f'={len(temp) + 1}.png'
                )
        plt.close('all')


def gauss_varyingT_outputs():
    args, output = load_parameters(device)

    unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"]
            )

    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )

    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 1000

    for i in [f'./final-outputs/', f'./final-outputs/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    for i in range(20):


        print(f"epoch {i}")

        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        img_mask = img_mask.to(device)

        slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])

        output_250 = diff.forward_backward(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=250, denoise_fn=args["noise_fn"]
                )

        output_250_images, mse_threshold_250 = make_prediction(
                img[slice, ...].reshape(1, 1, *args["img_size"]), output_250[-1].to(device),
                img_mask[slice, ...].reshape(1, 1, *args["img_size"])
                )

        output_500 = diff.forward_backward(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=500, denoise_fn=args["noise_fn"]
                )

        output_500_images, mse_threshold_500 = make_prediction(
                img[slice, ...].reshape(1, 1, *args["img_size"]), output_500[-1].to(device),
                img_mask[slice, ...].reshape(1, 1, *args["img_size"])
                )

        output_750 = diff.forward_backward(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=750, denoise_fn=args["noise_fn"]
                )

        output_750_images, mse_threshold_750 = make_prediction(
                img[slice, ...].reshape(1, 1, *args["img_size"]), output_750[-1].to(device),
                img_mask[slice, ...].reshape(1, 1, *args["img_size"])
                )

        # x_0,x_t,\hat{x}_0,se,se_threshold,ground truth

        temp = os.listdir(f"./final-outputs/ARGS={args['arg_num']}")

        fig, subplots = plt.subplots(
                3, 6, sharex=True, sharey=True, constrained_layout=False, figsize=(6, 3),
                squeeze=False,
                gridspec_kw={'wspace': 0, 'hspace': 0}
                )
        tempplot = fig.add_subplot(111, frameon=False)

        for i in range(3):
            subplots[i][0].imshow(img[slice, ...].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")

        subplots[0][1].imshow(output_250[251 // 2].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[0][2].imshow(output_250[-1].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[0][3].imshow(output_250_images[2].reshape(*args["img_size"]).cpu().numpy(), cmap="hot")
        subplots[0][4].imshow(output_250_images[3].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[0][5].imshow(output_250_images[4].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")

        subplots[1][1].imshow(output_500[501 // 2].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[1][2].imshow(output_500[-1].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[1][3].imshow(output_500_images[2].reshape(*args["img_size"]).cpu().numpy(), cmap="hot")
        subplots[1][4].imshow(output_500_images[3].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[1][5].imshow(output_500_images[4].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")

        subplots[2][1].imshow(output_750[751 // 2].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[2][2].imshow(output_750[-1].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[2][3].imshow(output_750_images[2].reshape(*args["img_size"]).cpu().numpy(), cmap="hot")
        subplots[2][4].imshow(output_750_images[3].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[2][5].imshow(output_750_images[4].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")

        for i in range(3):
            for j in range(6):
                subplots[i][j].tick_params(
                        top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False
                        )

        for i, val in enumerate(["$x_0$", "$x_t$", "Reconstruction", "Square Error", "Prediction", "Ground Truth"]):
            subplots[0][i].set_xlabel(f"{val}", fontsize=6)
            subplots[0][i].xaxis.set_label_position("top")

        for i, val in enumerate([250, 500, 750]):
            subplots[i][0].set_ylabel(f"$x_{{{val}}}$", fontsize=6)
            subplots[i][0].yaxis.set_label_position("left")
        plt.tick_params(labelcolor='none', which='both', top=False, left=False, bottom=False, right=False)

        plt.savefig(
                f'./final-outputs/ARGS={args["arg_num"]}/{new["filenames"][0][-9:-4]}-Gauss-attempt'
                f'={len(temp) + 1}.png'
                )

        plt.close('all')



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import torch
    import numpy as np

    from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
    from UNet import UNetModel

    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # font_files = font_manager.findSystemFonts(fontpaths="./")
    # for font_file in font_files:
    #     font_manager.fontManager.addfont(font_file)
    # plt.rcParams["font.family"] = "Times New Roman"

    font_path = "./times new roman.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()

    DATASET_PATH = './DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    make_all_outputs()
    # make_varying_frequency_outputs()
    # gauss_varyingT_outputs()

"""
Output varying figures for paper
"""

import random

from matplotlib import animation
from torchvision import transforms

import dataset
from helpers import gridify_output, load_parameters


def make_prediction(real, recon, mask, x_t, threshold=0.5, error_fn="sq"):
    """
    Make generic prediction and output tensor with order (real, x_lambda, reconstruction, square error, square error
    threshold, ground truth mask)
    :param real: initial real image x_0
    :param recon: reconstruction when diffused to x_t
    :param mask: ground truth mask
    :param x_t: middle image when initial image x_0 is noised through t time steps
    :param threshold: value to take threshold
    :param error_fn: square or l1 error - future work could explore error functions in feature space
    :return:
    """
    if error_fn == "sq":
        mse = ((recon - real).square() * 2) - 1
    elif error_fn == "l1":
        mse = (recon - real)
    mse_threshold = mse > (threshold * 2) - 1
    mse_threshold = (mse_threshold.float() * 2) - 1

    return torch.cat((real, x_t, recon, mse, mse_threshold, mask)), mse_threshold


def output_denoise_sequence(sequence: list, filename, masks, predictions):
    """
    sequence is [[t lots of images],[t lots of images],[t lots of images],[t lots of images]]
    or sequence is [t lots of images] where t is the number of forwad images plus backward plus
    prediction and real mask
    :param sequence: sequence of images (see above)
    :param filename: output filename
    :param masks: list of ground truth masks
    :param predictions: list of predicted images
    :return:
    """

    if len(sequence) > 10:
        sequence = [sequence]

    # split the forward and backward elements for labelling
    relevant_elements_forward = np.linspace(0, len(sequence[0]) // 2, 6).astype(np.int32)
    relevant_elements_backward = (-1 * relevant_elements_forward[-2::-1]) - 1

    relevant_elements = np.append(relevant_elements_forward, relevant_elements_backward)

    # sequence[0].shape  # B,C,H,W
    # init empty figure
    output = torch.empty(13 * len(sequence), sequence[0][0].shape[1], 256, 256, )

    # push each subimage into figure
    for j, new_sequence in enumerate(sequence):

        for i, val in enumerate(relevant_elements):
            output[13 * j + i] = new_sequence[val]
        output[13 * (j + 1) - 2] = predictions[j]
        output[13 * (j + 1) - 1] = masks[j]

    output = output.permute(0, 2, 3, 1)
    fig, subplots = plt.subplots(
            len(sequence), 13, figsize=(13, len(sequence)),
            gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=False
            )
    # mpl implot images with relevant pixel value renormalisation
    for brain in range(len(sequence)):
        for noise in range(13):
            # plot diffusion sequence
            if output[0].shape[-1] == 1:
                # img = scale_img(output[13 * brain + noise])
                subplots[brain][noise].imshow(
                        output[13 * brain + noise].reshape(*output[0].shape[-3:-1]).cpu().numpy(), cmap="gray"
                        )
            else:
                # mask
                if noise < 12:
                    # img = scale_img(output[13 * brain + noise])
                    img = (output[13 * brain + noise] + 1) / 2
                else:
                    # prediction
                    img = output[13 * brain + noise]
                # img = scale_img(output[13 * brain + noise])
                subplots[brain][noise].imshow(
                        img.reshape(*output[0].shape[-3:]).cpu().numpy(),
                        )
            subplots[brain][noise].tick_params(
                    top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False
                    )

    # Add axis labels
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


def output_masked_comparison(sequence, filename, t_distance=250, ):
    """
    sequence is ideally [[x_0,recon,mse,threshold_mse,ground_truth]*4] where
    [x_0,recon,mse,threshold_mse,ground_truth] is a single (5,1,256,256) torch tensor
    or sequence is [x_0,recon,mse,threshold_mse,ground_truth]
    :param sequence:
    :return:
    """
    if type(sequence) == torch.tensor:
        sequence = [sequence]

    # make plots
    fig, subplots = plt.subplots(
            len(sequence), 6, constrained_layout=False, figsize=(6, len(sequence)),
            squeeze=False,
            gridspec_kw={'wspace': 0, 'hspace': 0}
            )
    plt.tick_params(
            top=False, bottom=False, left=False, right=False,
            labelleft=False, labelbottom=False
            )

    for i, brain in enumerate(sequence):
        for plot in range(brain.shape[0]):
            # print(plot, torch.max(brain[plot]), torch.min(brain[plot]))
            # BCHW
            if plot > 2:
                # pick colourmap
                if plot == 3:
                    cmap = "hot"
                else:
                    cmap = "gray"
                # convert square error to grayscale for rgb images
                if brain[plot].shape[-3] == 3:
                    square_error_gray = transforms.functional.rgb_to_grayscale(brain[plot] + 1)
                    # threshold prediction
                    if plot == 4:
                        square_error_gray = ((square_error_gray > 0.1).float() * 2) - 1

                    subplots[i][plot].imshow(
                            square_error_gray.permute(1, 2, 0).cpu().numpy(), cmap=cmap
                            )
                else:
                    subplots[i][plot].imshow((brain[plot] + 1).permute(1, 2, 0).cpu().numpy(), cmap=cmap)
            else:
                # img = scale_img(brain[plot])
                # renorm image
                img = (brain[plot] + 1) / 2

                subplots[i][plot].imshow(
                        img.permute(1, 2, 0).cpu().numpy(),
                        # cmap='gray'
                        )
            subplots[i][plot].tick_params(
                    top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False
                    )

    # add labels and save
    for i, val in enumerate(
            ["$x_0$", f"$x_{{{t_distance}}}$", "Reconstruction", "Square Error", "Prediction",
             "Ground Truth"]
            ):
        subplots[0][i].set_xlabel(f"{val}", fontsize=6)
        subplots[0][i].xaxis.set_label_position("top")

    plt.savefig(filename)


def make_videos():
    """
    generate videos for dataset based on input arguments
    :return: selection of videos for dataset of trained model
    """
    # load parameters
    args, output = load_parameters(device)
    in_channels = 1
    if args["dataset"].lower() == "leather":
        in_channels = 3

    # init model, betas and diffusion classes
    unet = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    print(args)
    # load checkpoint
    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    # load specific dataset - ie carpet / leather / MRI
    if args["dataset"].lower() == "carpet":
        d_set = dataset.DAGM("./DATASETS/CARPET/Class1", True)
    elif args["dataset"].lower() == "leather":
        d_set = dataset.MVTec(
                "./DATASETS/leather", anomalous=True, img_size=args["img_size"],
                rgb=True, include_good=False
                )
    else:
        d_set = dataset.AnomalousMRIDataset(
                ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
                slice_selection="iterateKnown_restricted", resized=False
                )

    loader = dataset.init_dataset_loader(d_set, args)
    plt.rcParams['figure.dpi'] = 100

    # make directories
    for i in [f'./final-outputs/', f'./final-outputs/ARGS={args["arg_num"]}']:
        if not os.path.exists(i):
            os.makedirs(i)

    # generate 20 videos
    for i in range(20):

        #  if using simplex noise - select random lambda parameters with weighted probabilities
        if str(args["arg_num"]) == '28':
            t_distance = np.random.choice([150, 200, 250], p=[0.2, 0.4, 0.4])
        else:
            # select different parameters
            t_distance = np.random.choice([250, 500, 750], p=[0.2, 0.4, 0.4])
        print(f"loop {i}")
        new = next(loader)
        img = new["image"].to(device)

        # if mri - select random slice
        if args["dataset"] != "carpet" and args["dataset"] != "leather":

            slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
            img = img.reshape(img.shape[1], 1, *args["img_size"])
            img = img[slice, ...].reshape(1, 1, *args["img_size"])

        # perform diffusion
        output = diff.forward_backward(
                unet, img,
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=t_distance, denoise_fn=args["noise_fn"]
                )
        # plot, animate and save diffusion process
        fig, ax = plt.subplots()
        plt.axis('off')
        imgs = [[ax.imshow(gridify_output(output[x], 1), animated=True)] for x in range(0, len(output), 2)]
        ani = animation.ArtistAnimation(
                fig, imgs, interval=25, blit=True,
                repeat_delay=1000
                )
        temp = os.listdir(
                f'./final-outputs/ARGS={args["arg_num"]}'
                )

        output_name = f'./final-outputs/ARGS={args["arg_num"]}/attempt={len(temp) + 1}-sequence.mp4'
        ani.save(output_name)


def make_ano_outputs():
    args, output = load_parameters(device)
    in_channels = 1
    if args["dataset"].lower() == "leather":
        in_channels = 3
    if args["channels"] != "":
        in_channels = args["channels"]

    unet = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )

    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    if args["dataset"].lower() == "carpet":
        d_set = dataset.DAGM("./DATASETS/CARPET/Class1", True)
    elif args["dataset"].lower() == "leather":
        d_set = dataset.MVTec(
                "./DATASETS/leather", anomalous=True, img_size=args["img_size"],
                rgb=True, include_good=False
                )
    else:
        d_set = dataset.AnomalousMRIDataset(
                ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
                slice_selection="iterateKnown_restricted", resized=False
                )

    loader = dataset.init_dataset_loader(d_set, args)
    plt.rcParams['figure.dpi'] = 1000

    for i in [f'./final-outputs/', f'./final-outputs/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    # t_distance = 200
    for i in range(30):

        predictions = []
        sequences = []
        masks = []
        mse_thresholds = []

        rows = np.random.choice([2, 3, 4], p=[0.2, 0.3, 0.5, ])
        t_distance = np.random.choice([50, 150, 200, 250], p=[0.25, 0.25, 0.25, 0.25])
        rows, t_distance = 1, 250
        # threshold = np.random.choice([0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4])
        threshold = 0.5
        print(f"epoch {i}, rows @ epoch: {rows}")
        for k in range(rows):
            new = next(loader)

            while new["filenames"][0][-9:-4] != "19423":
                new = next(loader)

            # while torch.sum(new["mask"]) < 1000:
            #     new = next(loader)

            img = new["image"].to(device)

            img_mask = new["mask"]
            img_mask = img_mask.to(device)

            if args["dataset"] != "carpet" and args["dataset"] != "leather":

                slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
                img = img.reshape(img.shape[1], 1, *args["img_size"])
                img = img[slice, ...].reshape(1, 1, *args["img_size"])
                img_mask = img_mask.reshape(img_mask.shape[1], 1, *args["img_size"])
                img_mask = img_mask[slice, ...].reshape(1, 1, *args["img_size"])

            output = diff.forward_backward(
                    unet, img,
                    see_whole_sequence="whole",
                    # t_distance=5, denoise_fn=args["noise_fn"]
                    t_distance=t_distance, denoise_fn=args["noise_fn"]
                    )

            if args["dataset"] == "leather":
                output_images, mse_threshold = make_prediction(
                        img, output[-1].to(device),
                        img_mask, output[t_distance // 2].to(device), threshold=threshold
                        )
            else:
                output_images, mse_threshold = make_prediction(
                        img, output[-1].to(device),
                        img_mask, output[t_distance // 2].to(device)
                        )

            predictions.append(
                    output_images
                    )

            mse_thresholds.append(mse_threshold)
            masks.append(img_mask)
            sequences.append(output)

        temp = os.listdir(f"./final-outputs/ARGS={args['arg_num']}")
        output_masked_comparison(
                predictions, f'./final-outputs/ARGS={args["arg_num"]}/attempt'
                             f'={len(temp) + 1}-{threshold}-predictions.png', t_distance
                )
        output_denoise_sequence(
                sequences, f'./final-outputs/ARGS={args["arg_num"]}/attempt'
                           f'={len(temp) + 1}-{threshold}-sequence.png', masks, mse_thresholds
                )
        plt.close('all')


def make_gauss_simplex_outputs(simplex_argNum="28", gauss_argNum="26"):
    """
    Output figure containing both gauss and simplex noise
    :param simplex_argNum: arg value of saved simplex checkpoint
    :param gauss_argNum: arg value of saved gauss checkpoint
    :return:
    """
    sys.argv[1] = simplex_argNum
    args_simplex, output_simplex = load_parameters(device)
    sys.argv[1] = gauss_argNum
    args_gauss, output_gauss = load_parameters(device)
    # consider different channel sizes

    in_channels = 1
    if args_simplex["dataset"].lower() == "leather":
        in_channels = 3
    if args_simplex["channels"] != "":
        in_channels = args_simplex["channels"]

    # init model and checkpoints

    unet_simplex = UNetModel(
            args_simplex['img_size'][0], args_simplex['base_channels'], channel_mults=args_simplex['channel_mults'],
            in_channels=in_channels
            )
    unet_gauss = UNetModel(
            args_gauss['img_size'][0], args_gauss['base_channels'], channel_mults=args_gauss['channel_mults'],
            in_channels=in_channels
            )

    betas = get_beta_schedule(args_simplex['T'], args_simplex['beta_schedule'])

    diff_simplex = GaussianDiffusionModel(
            args_simplex['img_size'], betas, loss_weight=args_simplex['loss_weight'],
            loss_type=args_simplex['loss-type'], noise=args_simplex["noise_fn"], img_channels=in_channels
            )
    diff_gauss = GaussianDiffusionModel(
            args_simplex['img_size'], betas, loss_weight=args_simplex['loss_weight'],
            loss_type=args_simplex['loss-type'], noise=args_gauss["noise_fn"], img_channels=in_channels
            )

    unet_simplex.load_state_dict(output_simplex["ema"])
    unet_gauss.load_state_dict(output_gauss["ema"])
    unet_simplex.eval()
    unet_gauss.eval()

    # init varying datasets
    if args_simplex["dataset"].lower() == "carpet":
        d_set = dataset.DAGM("./DATASETS/CARPET/Class1", True)
    elif args_simplex["dataset"].lower() == "leather":
        if in_channels == 3:
            d_set = dataset.MVTec(
                    "./DATASETS/leather", anomalous=True, img_size=args_simplex["img_size"],
                    rgb=True, include_good=False
                    )
        else:
            d_set = dataset.MVTec(
                    "./DATASETS/leather", anomalous=True, img_size=args_simplex["img_size"],
                    rgb=False, include_good=False
                    )
    else:
        d_set = dataset.AnomalousMRIDataset(
                ROOT_DIR=f'{DATASET_PATH}', img_size=args_simplex['img_size'],
                slice_selection="iterateKnown_restricted", resized=False
                )
    loader = dataset.init_dataset_loader(d_set, args_simplex)

    plt.rcParams['figure.dpi'] = 1000

    for i in [f'./final-outputs/', f'./final-outputs/gauss_simplex', f'./final-outputs/gauss_simplex/'
                                                                     f'{args_simplex["dataset"]}-{in_channels}']:
        if not os.path.exists(i):
            os.makedirs(i)
    # generate 20 figures
    for i in range(20):

        predictions = []
        unet_simplex.to(device)
        # select number of rows - ie 2 simplex and 2 gauss or 1 of each
        rows = random.randint(1, 2)
        # select lambda to directly compare lambda values
        t_distance = np.random.choice([150, 200, 250, 300], p=[0.25, 0.25, 0.25, 0.25])
        # select random threshold - this was found to be more of a hyperparameter for textured surfaces such as from
        # DAGM and MVTec
        threshold = np.random.choice([0.15, 0.2, 0.25])
        imgs = []

        for k in range(rows):
            new = next(loader)
            # As the MVTec takes a section of a larger image - omly use an image with more than 1000 pixels in ground
            # truth mask to guarantee an anomaly lies

            while torch.sum(new["mask"]) < 1000:
                new = next(loader)
            if args_simplex["dataset"] != "carpet" and args_simplex["dataset"] != "leather":
                slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
                new["image"] = new["image"].reshape(new["image"].shape[1], 1, *args_simplex["img_size"])
                new["image"] = new["image"][slice, ...].reshape(1, 1, *args_simplex["img_size"])
                new["mask"] = new["mask"].reshape(new["mask"].shape[1], 1, *args_simplex["img_size"])
                new["mask"] = new["mask"][slice, ...].reshape(1, 1, *args_simplex["img_size"])
            imgs.append(new)

        for new in imgs:
            img = new["image"].to(device)
            img_mask = new["mask"].to(device)

            output = diff_simplex.forward_backward(
                    unet_simplex, img,
                    see_whole_sequence="whole",
                    t_distance=t_distance, denoise_fn=args_simplex["noise_fn"]
                    )
            output_images, mse_threshold = make_prediction(
                    img, output[-1].to(device), img_mask, output[t_distance // 2].to(device), threshold=threshold,
                    error_fn="sq"
                    )
            predictions.append(
                    output_images
                    )

        unet_simplex.cpu()

        unet_gauss.to(device)
        for new in imgs:
            img = new["image"].to(device)
            img_mask = new["mask"].to(device)

            output = diff_gauss.forward_backward(
                    unet_gauss, img,
                    see_whole_sequence="whole",
                    t_distance=t_distance, denoise_fn=args_gauss["noise_fn"]
                    )

            output_images, mse_threshold = make_prediction(
                    img, output[-1].to(device),
                    img_mask, output[t_distance // 2].to(device), threshold=threshold, error_fn="sq"
                    )
            predictions.append(
                    output_images
                    )
        unet_gauss.cpu()
        temp = os.listdir(f"./final-outputs/gauss_simplex/{args_simplex['dataset']}-{in_channels}")
        output_masked_comparison(
                predictions,
                f"./final-outputs/gauss_simplex/{args_simplex['dataset']}-{in_channels}/{len(temp) + 1}test{i}"
                f"-{threshold}.png",
                t_distance
                )


def make_test_set_outputs(anomalous=False, t_distance=250):
    """
    Generate Fig 1 from paper - containing Gauss and simplex noise on healthy or anomalous test set
    :param anomalous: test on anomalous?
    :param t_distance: lambda value for diffusion process
    :return:
    """
    # load both simplex and Gauss checkpoints separately
    sys.argv[1] = "28"
    args_simplex, output_simplex = load_parameters(device)
    sys.argv[1] = "26"
    args_gauss, output_gauss = load_parameters(device)

    unet_simplex = UNetModel(
            args_simplex['img_size'][0], args_simplex['base_channels'], channel_mults=args_simplex['channel_mults']
            )
    unet_gauss = UNetModel(
            args_gauss['img_size'][0], args_gauss['base_channels'], channel_mults=args_gauss['channel_mults']
            )

    betas = get_beta_schedule(args_simplex['T'], args_simplex['beta_schedule'])

    diff_simplex = GaussianDiffusionModel(
            args_simplex['img_size'], betas, loss_weight=args_simplex['loss_weight'],
            loss_type=args_simplex['loss-type'], noise=args_simplex["noise_fn"]
            )
    diff_gauss = GaussianDiffusionModel(
            args_simplex['img_size'], betas, loss_weight=args_simplex['loss_weight'],
            loss_type=args_simplex['loss-type'], noise=args_gauss["noise_fn"]
            )

    unet_simplex.load_state_dict(output_simplex["ema"])
    unet_gauss.load_state_dict(output_gauss["ema"])
    unet_simplex.eval()
    unet_gauss.eval()
    if anomalous:
        ano_dataset = dataset.AnomalousMRIDataset(
                ROOT_DIR=f'{DATASET_PATH}', img_size=args_simplex['img_size'],
                slice_selection="iterateKnown_restricted", resized=False
                )

        loader = dataset.init_dataset_loader(ano_dataset, args_simplex)
    else:
        _, testing_dataset = dataset.init_datasets("./", args_simplex)
        loader = dataset.init_dataset_loader(testing_dataset, args_simplex)
    plt.rcParams['figure.dpi'] = 1000

    for i in [f'./final-outputs/', f'./final-outputs/ARGS={args_simplex["arg_num"]}']:
        if not os.path.exists(i):
            os.makedirs(i)

    for i in range(20):

        sequences = []
        unet_simplex.to(device)
        if anomalous:
            rows = 1
        else:
            rows = 2
        # rows = np.random.choice([1, 2], p=[0.4, 0.6])
        # select each volume for imgs - and diffuse each with simplex noise
        imgs = []
        for i in range(rows):
            new = next(loader)
            img = new["image"].to(device)
            img = img.reshape(img.shape[1], 1, *args_simplex["img_size"])
            if anomalous:
                slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
                img = img[slice, ...].reshape(1, 1, *args_simplex["img_size"])
            imgs.append(img)

        for k in range(rows):
            img = imgs[k]
            output = diff_simplex.forward_backward(
                    unet_simplex, img.reshape(1, 1, *args_simplex["img_size"]),
                    see_whole_sequence="whole",
                    t_distance=t_distance, denoise_fn=args_simplex["noise_fn"]
                    )

            sequences.append(output)
        unet_simplex.cpu()

        # diffuse each with gaussian noise
        unet_gauss.to(device)
        for k in range(rows):
            img = imgs[k]
            output = diff_gauss.forward_backward(
                    unet_gauss, img.reshape(1, 1, *args_simplex["img_size"]),
                    see_whole_sequence="whole",
                    t_distance=t_distance, denoise_fn=args_gauss["noise_fn"]
                    )

            sequences.append(output)
        unet_gauss.cpu()

        # following image generation - output images with similar format to function - output_denoise_sequence

        temp = os.listdir(f"./final-outputs/ARGS={args_simplex['arg_num']}")

        if len(sequences) > 10:
            sequences = [sequences]

        relevant_elements_forward = np.linspace(0, len(sequences[0]) // 2, 4).astype(np.int32)
        relevant_elements_backward = (-1 * relevant_elements_forward[-2::-1]) - 1

        relevant_elements = np.append(relevant_elements_forward, relevant_elements_backward)

        output = torch.empty(7 * len(sequences), 1, 256, 256)

        # insert each image into empty tensor
        for j, new_sequence in enumerate(sequences):

            for i, val in enumerate(relevant_elements):
                output[7 * j + i] = new_sequence[val]

        fig, subplots = plt.subplots(
                len(sequences), 7, figsize=(7, len(sequences)),
                gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=False
                )

        # plot each image
        for brain in range(len(sequences)):
            for noise in range(7):
                # subplots[brain][noise].imshow(output[12 * brain + noise].reshape(256, 256, 1), cmap="gray")
                subplots[brain][noise].imshow(
                        output[7 * brain + noise].reshape(*output[0].shape[-2:]).cpu().numpy(),
                        cmap="gray"
                        )
                subplots[brain][noise].tick_params(
                        top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False
                        )
                # subplots[brain][noise].axis('off')
        # set labels
        for i in range(4):

            subplots[0][i].set_xlabel(f"$x_{{{relevant_elements[i]}}}$", fontsize=6)
            subplots[0][i].xaxis.set_label_position("top")

        for i in range(4, 7):
            subplots[0][i].set_xlabel(f"$x_{{{relevant_elements_forward[::-1][1:][i - 4]}}}$", fontsize=6)
            subplots[0][i].xaxis.set_label_position("top")

        plt.savefig(
                f'./final-outputs/ARGS={args_simplex["arg_num"]}/test_set_mixed_attempt'
                f'={len(temp) + 1}-sequence.png'
                )
        plt.close('all')


def make_varying_frequency_outputs():
    """
    Figure for affect of simplex frequency on segmentation performance - Fig. 3 from AnoDDPM paper
    :return:
    """
    # init model and load checkpoint
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
        if not os.path.exists(i):
            os.makedirs(i)
    # generate 22 images - ie one figure for each volume in MRI dataset
    for i in range(22):

        print(f"epoch {i}")

        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = img_mask = new["mask"]
        img_mask = img_mask.to(device)

        slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        # diffuse over varying frequency values
        output = diff.detection_A_fixedT(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]), args,
                img_mask[slice, ...].reshape(1, 1, *args["img_size"])
                )

        fig, subplots = plt.subplots(
                6, 6, sharex=True, sharey=True, constrained_layout=False, figsize=(6, 6),
                gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=False
                )

        tempplot = fig.add_subplot(111, frameon=False)
        #  plot  images and set label

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
    """
    generate figure for Gaussian diffusion with lambda as 250,500,750
    :return:
    """

    # init model and load checkpoint
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
        if not os.path.exists("./final-outputs"):
            os.makedirs(i)

    # generate 20 figures
    for i in range(20):


        print(f"epoch {i}")

        # select image and slice
        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = new["mask"]
        img_mask = img_mask.to(device)

        slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])

        # perform diffusion with lambda as 250, 500 and 750

        output_250 = diff.forward_backward(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=250, denoise_fn=args["noise_fn"]
                )

        output_250_images, mse_threshold_250 = make_prediction(
                img[slice, ...].reshape(1, 1, *args["img_size"]), output_250[-1].to(device),
                img_mask[slice, ...].reshape(1, 1, *args["img_size"]), output_250[251 // 2].to(device)
                )

        output_500 = diff.forward_backward(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=500, denoise_fn=args["noise_fn"]
                )

        output_500_images, mse_threshold_500 = make_prediction(
                img[slice, ...].reshape(1, 1, *args["img_size"]), output_500[-1].to(device),
                img_mask[slice, ...].reshape(1, 1, *args["img_size"]), output_500[501 // 2].to(device)
                )

        output_750 = diff.forward_backward(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=750, denoise_fn=args["noise_fn"]
                )

        output_750_images, mse_threshold_750 = make_prediction(
                img[slice, ...].reshape(1, 1, *args["img_size"]), output_750[-1].to(device),
                img_mask[slice, ...].reshape(1, 1, *args["img_size"]), output_750[751 // 2].to(device)
                )

        # x_0,x_t,\hat{x}_0,se,se_threshold,ground truth

        # output figure manually
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
        subplots[0][3].imshow(output_250_images[3].reshape(*args["img_size"]).cpu().numpy(), cmap="hot")
        subplots[0][4].imshow(output_250_images[4].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[0][5].imshow(output_250_images[5].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")

        subplots[1][1].imshow(output_500[501 // 2].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[1][2].imshow(output_500[-1].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[1][3].imshow(output_500_images[3].reshape(*args["img_size"]).cpu().numpy(), cmap="hot")
        subplots[1][4].imshow(output_500_images[4].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[1][5].imshow(output_500_images[5].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")

        subplots[2][1].imshow(output_750[751 // 2].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[2][2].imshow(output_750[-1].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[2][3].imshow(output_750_images[3].reshape(*args["img_size"]).cpu().numpy(), cmap="hot")
        subplots[2][4].imshow(output_750_images[4].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
        subplots[2][5].imshow(output_750_images[5].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")

        # set labels and save
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


def make_gan_outputs():
    import Comparative_models.CE as CE
    import detection

    # Figure for context encoder model
    args, output = load_parameters(device)
    args["Batch_Size"] = 1

    netG = CE.Generator(start_size=args['img_size'][0], out_size=args['inpaint_size'], dropout=args["dropout"])

    # load params and dataset
    netG.load_state_dict(output["generator_state_dict"])
    netG.to(device)
    netG.eval()
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )

    loader = dataset.init_dataset_loader(ano_dataset, args)

    # init args
    overlapSize = args['overlap']
    input_cropped = torch.FloatTensor(args['Batch_Size'], 1, 256, 256)
    input_cropped = input_cropped.to(device)

    for i in [f'./final-outputs/', f'./final-outputs/ARGS={args["arg_num"]}']:
        if not os.path.exists("./final-outputs"):
            os.makedirs(i)

    # output 22 images
    for i in range(22):

        predictions = []

        # randomly select number of mri images per figure
        rows = np.random.choice([1, 2, 3, 4, 5, 8], p=[0.3, 0.3, 0.1, 0.1, 0.15, 0.05])
        print(f"epoch {i}, rows @ epoch: {rows}")
        for k in range(rows):
            print(k)
            new = next(loader)
            img = new["image"]
            img = img.reshape(img.shape[1], 1, *args["img_size"])
            img_mask = img_mask = new["mask"]

            slice = np.random.choice([0, 1, 2, 3], p=[0.15, 0.35, 0.35, 0.15])

            img_mask = img_mask[slice, ...].reshape(1, 1, *args["img_size"]).to(device)
            img = img[slice, ...].reshape(1, 1, *args["img_size"]).to(device)

            # x_cpu = new["image"]
            # x = x_cpu.to(device)

            # B,C,W,H

            # reconstruct image
            if args['type'] == 'sliding':
                recon_image = detection.ce_sliding_window(img, netG, input_cropped, args)
            else:
                input_cropped.resize_(img.size()).copy_(img)
                recon_image = input_cropped.clone()
                with torch.no_grad():
                    input_cropped.resize_(img.size()).copy_(img)
                    input_cropped[:, 0,
                    args['img_size'][0] // 4 + overlapSize:
                    args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize,
                    args['img_size'][0] // 4 + overlapSize:
                    args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize] \
                        = 0
                fake = netG(input_cropped)

                recon_image.data[:, :, args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4,
                args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4] = fake.data

            # generate prediction via naïve threshold
            mse = ((recon_image - img).square() * 2) - 1
            mse_threshold = mse > 0
            mse_threshold = (mse_threshold.float() * 2) - 1

            pred = torch.cat(
                    (img, recon_image.reshape(1, 1, *args["img_size"]), mse, mse_threshold,
                     img_mask)
                    )

            predictions.append(pred.cpu())

        # plot each image
        temp = os.listdir(f"./final-outputs/ARGS={args['arg_num']}")

        fig, subplots = plt.subplots(
                len(predictions), 5, sharex=True, sharey=True, constrained_layout=False, figsize=(5, len(predictions)),
                squeeze=False,
                gridspec_kw={'wspace': 0, 'hspace': 0}
                )
        plt.tick_params(
                top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False
                )

        for i, brain in enumerate(predictions):
            for plot in range(brain.shape[0]):
                if plot == 2:
                    subplots[i][plot].imshow(brain[plot].reshape(*brain.shape[-2:]).detach().cpu().numpy(), cmap="hot")

                else:
                    subplots[i][plot].imshow(brain[plot].reshape(*brain.shape[-2:]).detach().cpu().numpy(), cmap="gray")
                subplots[i][plot].tick_params(
                        top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False
                        )

        #  set axis labels and save
        for i, val in enumerate(["$x_0$", "Reconstruction", "Square Error", "Prediction", "Ground Truth"]):
            subplots[0][i].set_xlabel(f"{val}", fontsize=6)
            subplots[0][i].xaxis.set_label_position("top")

        plt.savefig(
                f'./final-outputs/ARGS={args["arg_num"]}/attempt'
                f'={len(temp) + 1}-predictions.png'
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
    import sys

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # add times new roman to mpl fonts
    font_path = "./times new roman.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()

    if str(sys.argv[2]):
        DATASET_PATH = str(sys.argv[2])
    else:
        DATASET_PATH = './DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plt.set_cmap('gray')

    plt.rcParams['figure.dpi'] = 600
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)

    # run different outputs based on model and dataset
    if str(sys.argv[1]) == "101" or str(sys.argv[1]) == "102" or str(sys.argv[1]) == "103" or str(sys.argv[1]) == \
            "104":
        make_gan_outputs()
    elif str(sys.argv[1]) == "23":
        make_varying_frequency_outputs()
    elif str(sys.argv[1]) == "26":
        make_videos()
    elif str(sys.argv[1]) == "28":
        make_videos()
    elif str(sys.argv[1]) == "30":
        make_videos()
    elif str(sys.argv[1]) == "1000":
        make_gauss_simplex_outputs("31", "30")
    elif str(sys.argv[1]) == "1001":
        make_gauss_simplex_outputs("33", "32")
    else:
        make_ano_outputs()

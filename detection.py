import random
import time

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


def anomalous_validation_1():
    """
    Iterates over 4 anomalous slices for each Volume, returning diffused video for it,
    the heatmap of that & detection method (A&B) or C
    :return:
    """
    args, output = load_parameters(device)

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
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )
    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass

    # make folder for each anomalous volume
    for i in ano_dataset.slices.keys():
        try:
            os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{i}')
        except OSError:
            pass

    dice_data = []
    start_time = time.time()
    for i in range(len(ano_dataset)):

        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
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
                    t_distance=timestep, denoise_fn=args["noise_fn"]
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
                            img_mask[slice_number, ...].reshape(1, 1, *args["img_size"]),
                            output_name + ".png"
                            )
                    )

            plt.close('all')

            if args["noise_fn"] == "gauss":
                dice = diff.detection_B(
                        unet, img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice_number].numpy()[0]),
                        img_mask[slice_number, ...].reshape(1, 1, *args["img_size"]), "gauss",
                        total_avg=3
                        )
                dice_data.append(dice)
            elif args["noise_fn"] == "simplex":
                dice = diff.detection_B(
                        unet, img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice_number].numpy()[0]),
                        img_mask[slice_number, ...].reshape(1, 1, *args["img_size"]), denoise_fn=args["noise_fn"],
                        total_avg=3
                        )
                dice_data.append(dice)
            elif args["noise_fn"] == "simplex_randParam":
                diff.detection_A(
                        unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice].numpy()[0]),
                        img_mask[slice_number, ...].reshape(1, 1, *args["img_size"])
                        )
                dice = diff.detection_B(
                        unet, img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                        args, (new["filenames"][0][-9:-4], new["slices"][slice_number].numpy()[0]),
                        img_mask[slice_number, ...].reshape(1, 1, *args["img_size"]), "octave",
                        total_avg=3
                        )
                dice_data.append(dice)
        time_taken = time.time() - start_time
        remaining_epochs = 22 - i
        time_per_epoch = time_taken / (i + 1)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)

        print(
                f"file: {new['filenames'][0][-9:-4]}, "
                f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining time: {hours}:{mins:02.0f}"
                )


def anomalous_dice_calculation():
    """
    Iterates over 4 anomalous slices for each Volume, returning diffused video for it,
    the heatmap of that & detection method (A&B) or C
    :return:
    """
    args, output = load_parameters(device)
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
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )
    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass

    # make folder for each anomalous volume
    for i in ano_dataset.slices.keys():
        try:
            os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{i}')
        except OSError:
            pass

    dice_data = []
    ssim_data = []
    IOU = []
    precision = []
    recall = []
    FPR = []
    start_time = time.time()
    for i in range(len(ano_dataset)):

        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        img_mask = img_mask.to(device)
        img_mask = (img_mask > 0).float()
        for slice_number in range(4):
            try:
                os.makedirs(
                        f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                        f'{new["slices"][slice_number].numpy()[0]}'
                        )
            except OSError:
                pass
            output = diff.forward_backward(
                    unet, img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                    see_whole_sequence=None,
                    t_distance=250, denoise_fn=args["noise_fn"]
                    )

            evaluation.heatmap(
                    img[slice_number, ...].reshape(1, 1, *args["img_size"]), output.to(device),
                    img_mask[slice_number, ...].reshape(1, 1, *args["img_size"]),
                    f"./diffusion-training-images/ARGS={args['arg_num']}/{new['filenames'][0][-9:-4]}"
                    f"-{new['slices'][slice_number].cpu().item()}.png",
                    save=True
                    )
            mse = (img - output).square()
            mse = (mse > 0.5).float()
            dice_data.append(evaluation.dice_coeff(img, output.to(device), img_mask, mse=mse).cpu().item())
            ssim_data.append(
                    evaluation.SSIM(img.reshape(*args["img_size"]), output.reshape(*args["img_size"]))
                    )
            precision.append(evaluation.precision(img_mask, mse).cpu().numpy())
            recall.append(evaluation.recall(img_mask, mse).cpu().numpy())
            IOU.append(evaluation.IoU(img_mask, mse))
            FPR.append(evaluation.FPR(img_mask, mse).cpu().numpy())

            plt.close('all')

        time_taken = time.time() - start_time
        remaining_epochs = 22 - i
        time_per_epoch = time_taken / (i + 1)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)

        print(
                f"file: {new['filenames'][0][-9:-4]}, "
                f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining time: {hours}:{mins:02.0f}"
                )

        print(f"Dice coefficient: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +-{np.std(ssim_data[-4:])}")
        print(f"Dice: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +- {np.std(ssim_data[-4:])}")
        print(f"Precision: {np.mean(precision[-4:])} +- {np.std(precision[-4:])}")
        print(f"Recall: {np.mean(recall[-4:])} +- {np.std(recall[-4:])}")
        print(f"FPR: {np.mean(FPR[-4:])} +- {np.std(FPR[-4:])}")
        print(f"IOU: {np.mean(IOU[-4:])} +- {np.std(IOU[-4:])}")

    print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(
            f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
            f" {np.std(ssim_data)}"
            )
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")


def graph_data():
    args, output = load_parameters(device)
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
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )
    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./metrics/')
    except OSError:
        pass

    try:
        os.makedirs(f'./metrics/ARGS={args["arg_num"]}')
    except OSError:
        pass
    t_range = np.linspace(0, 999, 200).astype(np.int32)

    start_time = time.time()
    for i in range(len(ano_dataset)):

        dice_data = []
        ssim_data = []
        precision = []
        recall = []
        IOU = []
        FPR = []

        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(img.shape[1], 1, *args["img_size"])
        img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        img_mask = img_mask.to(device)
        img_mask = (img_mask > 0).float()

        slice_number = 1
        img = img[slice_number, ...].reshape(1, 1, *args["img_size"])
        img_mask = img_mask[slice_number, ...].reshape(1, 1, *args["img_size"])
        # for slice_number in range(4):

        for t in t_range:
            output = diff.forward_backward(
                    unet, img,
                    see_whole_sequence=None,
                    t_distance=t, denoise_fn=args["noise_fn"]
                    )

            mse = (img - output).square()
            mse = (mse > 0.5).float()
            dice_data.append(evaluation.dice_coeff(img, output.to(device), img_mask, mse=mse).cpu().item())
            ssim_data.append(
                    evaluation.SSIM(img.reshape(*args["img_size"]), output.reshape(*args["img_size"]))
                    )
            precision.append(evaluation.precision(img_mask, mse).cpu().numpy())
            recall.append(evaluation.recall(img_mask, mse).cpu().numpy())
            IOU.append(evaluation.IoU(img_mask, mse))
            FPR.append(evaluation.FPR(img_mask, mse).cpu().numpy())

            if t in [0, 100, 200, 301, 401, 502, 602, 702, 803, 903, 999]:
                print(t, dice_data[-1], ssim_data[-1], precision[-1], recall[-1], IOU[-1])

            plt.plot(t_range[:len(dice_data)], dice_data, label="dice")
            plt.plot(t_range[:len(dice_data)], IOU, label="IOU")
            plt.plot(t_range[:len(dice_data)], precision, label="precision")
            plt.plot(t_range[:len(dice_data)], recall, label="recall")
            plt.legend(loc="upper right")
            ax = plt.gca()
            ax.set_ylim([0, 1])
            plt.savefig(f'./metrics/ARGS={args["arg_num"]}/{new["filenames"][0][-9:-4]}.png')
            plt.clf()

        time_taken = time.time() - start_time
        remaining_epochs = 22 - i
        time_per_epoch = time_taken / (i + 1)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)

        print(
                f"file: {new['filenames'][0][-9:-4]}, "
                f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining time: {hours}:{mins:02.0f}"
                )

        print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
        print(
                f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
                f" {np.std(ssim_data)}"
                )
        print(f"Dice: {np.mean(dice_data)} +- {np.std(dice_data)}")
        print(
                f"Structural Similarity Index (SSIM): {np.mean(ssim_data)} +-"
                f" {np.std(ssim_data)}"
                )
        print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
        print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
        print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")

        plt.plot(t_range, dice_data, label="dice")
        plt.plot(t_range, IOU, label="IOU")
        plt.plot(t_range, precision, label="precision")
        plt.plot(t_range, recall, label="recall")
        plt.legend(loc="upper right")
        ax = plt.gca()
        ax.set_ylim([0, 1])
        plt.savefig(f'./metrics/ARGS={args["arg_num"]}/{new["filenames"][0][-9:-4]}.png')
        plt.clf()
        with open(f'./metrics/ARGS={args["arg_num"]}/{new["filenames"][0][-9:-4]}.csv', mode="w") as f:

            f.write(",".join([f"{i:04}" for i in t_range]))
            f.write("\n")
            for i in [dice_data, ssim_data, IOU, precision, recall, FPR]:
                f.write(",".join([f"{j:.4f}" for j in i]))
                f.write("\n")


def unet_anomalous():
    args, output = load_parameters(device)
    args["Batch_Size"] = 1
    print(repr(args['arg_num']))
    unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )
    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass

    # make folder for each anomalous volume
    for i in ano_dataset.slices.keys():
        try:
            os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{i}')
        except OSError:
            pass

    dice_data = []
    ssim_data = []
    IOU = []
    precision = []
    recall = []
    FPR = []
    start_time = time.time()
    for i in range(len(ano_dataset)):

        new = next(loader)
        image = new["image"].reshape(new["image"].shape[1], 1, *args["img_size"])

        img_mask_whole = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        for slice_number in range(4):
            try:
                os.makedirs(
                        f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                        f'{new["slices"][slice_number].numpy()[0]}'
                        )
            except OSError:
                pass
            img = image[slice_number, ...].to(device).reshape(1, 1, *args["img_size"])
            img_mask = img_mask_whole[slice_number, ...].to(device)
            img_mask = (img_mask > 0).float().reshape(1, 1, *args["img_size"])

            t_batch = torch.tensor([1], device=device).repeat(img.shape[0])
            output = unet(img, t_batch)
            evaluation.heatmap(
                    img, output.to(device).reshape(1, *args["img_size"]),
                    img_mask,
                    f"./diffusion-training-images/ARGS={args['arg_num']}/{new['filenames'][0][-9:-4]}"
                    f"-{new['slices'][slice_number].cpu().item()}.png",
                    save=True
                    )
            mse = (img - output).square()
            mse = (mse > 0.5).float()
            dice_data.append(evaluation.dice_coeff(img, output.to(device), img_mask, mse=mse).detach().cpu().numpy())
            ssim_data.append(
                    evaluation.SSIM(img.reshape(*args["img_size"]), output.reshape(*args["img_size"]))
                    )
            precision.append(evaluation.precision(img_mask, mse).detach().cpu().numpy())
            recall.append(evaluation.recall(img_mask, mse).detach().cpu().numpy())
            IOU.append(evaluation.IoU(img_mask, mse))
            FPR.append(evaluation.FPR(img_mask, mse).detach().cpu().numpy())

            plt.close('all')

        time_taken = time.time() - start_time
        remaining_epochs = 22 - i
        time_per_epoch = time_taken / (i + 1)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)

        print(
                f"file: {new['filenames'][0][-9:-4]}, "
                f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining time: {hours}:{mins:02.0f}"
                )

        print(f"Dice coefficient: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +-{np.std(ssim_data[-4:])}")
        print(f"Dice: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +- {np.std(ssim_data[-4:])}")
        print(f"Precision: {np.mean(precision[-4:])} +- {np.std(precision[-4:])}")
        print(f"Recall: {np.mean(recall[-4:])} +- {np.std(recall[-4:])}")
        print(f"FPR: {np.mean(FPR[-4:])} +- {np.std(FPR[-4:])}")
        print(f"IOU: {np.mean(IOU[-4:])} +- {np.std(IOU[-4:])}")

    print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(
            f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
            f" {np.std(ssim_data)}"
            )
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")


def gan_anomalous():
    import CE
    args, output = load_parameters(device)
    args["Batch_Size"] = 1

    netG = CE.Generator(start_size=args['img_size'][0], out_size=args['inpaint_size'], dropout=args["dropout"])

    netG.load_state_dict(output["generator_state_dict"])
    netG.to(device)
    netG.eval()
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )

    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 1000

    overlapSize = args['overlap']
    input_cropped = torch.FloatTensor(args['Batch_Size'], 1, 256, 256)
    input_cropped = input_cropped.to(device)

    try:
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass

    # make folder for each anomalous volume
    for i in ano_dataset.slices.keys():
        try:
            os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{i}')
        except OSError:
            pass

    dice_data = []
    ssim_data = []
    IOU = []
    precision = []
    recall = []
    FPR = []
    start_time = time.time()
    for i in range(len(ano_dataset)):

        new = next(loader)
        image = new["image"].reshape(new["image"].shape[1], 1, *args["img_size"])

        img_mask_whole = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        for slice_number in range(4):
            try:
                os.makedirs(
                        f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                        f'{new["slices"][slice_number].numpy()[0]}'
                        )
            except OSError:
                pass
            img = image[slice_number, ...].to(device).reshape(1, 1, *args["img_size"])
            img_mask = img_mask_whole[slice_number, ...].to(device)
            img_mask = (img_mask > 0).float().reshape(1, 1, *args["img_size"])

            print(input_cropped.device, netG.device, img.device)
            recon_image = ce_sliding_window(img, netG, input_cropped, args)

            evaluation.heatmap(
                    img, recon_image.to(device).reshape(1, *args["img_size"]),
                    img_mask,
                    f"./diffusion-training-images/ARGS={args['arg_num']}/{new['filenames'][0][-9:-4]}"
                    f"-{new['slices'][slice_number].cpu().item()}.png",
                    save=True
                    )
            mse = (img - recon_image).square()
            mse = (mse > 0.5).float()
            dice_data.append(
                    evaluation.dice_coeff(img, recon_image.to(device), img_mask, mse=mse).detach().cpu().numpy()
                    )
            ssim_data.append(
                    evaluation.SSIM(img.reshape(*args["img_size"]), recon_image.reshape(*args["img_size"]))
                    )
            precision.append(evaluation.precision(img_mask, mse).detach().cpu().numpy())
            recall.append(evaluation.recall(img_mask, mse).detach().cpu().numpy())
            IOU.append(evaluation.IoU(img_mask, mse))
            FPR.append(evaluation.FPR(img_mask, mse).detach().cpu().numpy())

            plt.close('all')

        time_taken = time.time() - start_time
        remaining_epochs = 22 - i
        time_per_epoch = time_taken / (i + 1)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)

        print(
                f"file: {new['filenames'][0][-9:-4]}, "
                f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining time: {hours}:{mins:02.0f}"
                )

        print(f"Dice coefficient: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +-{np.std(ssim_data[-4:])}")
        print(f"Dice: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +- {np.std(ssim_data[-4:])}")
        print(f"Precision: {np.mean(precision[-4:])} +- {np.std(precision[-4:])}")
        print(f"Recall: {np.mean(recall[-4:])} +- {np.std(recall[-4:])}")
        print(f"FPR: {np.mean(FPR[-4:])} +- {np.std(FPR[-4:])}")
        print(f"IOU: {np.mean(IOU[-4:])} +- {np.std(IOU[-4:])}")

    print("\n\n")
    print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(
            f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
            f" {np.std(ssim_data)}"
            )
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")


def ce_sliding_window(img, netG, input_cropped, args):
    recon_image = input_cropped.clone()
    for center_offset_y in range(0, 200, args['inpaint_size']):

        for center_offset_x in range(0, 200, args['inpaint_size']):
            with torch.no_grad():
                input_cropped.resize_(img.size()).copy_(img)
                input_cropped[:, 0,
                16 + center_offset_x + args['overlap']:16 + args['inpaint_size'] + center_offset_x - args['overlap'],
                16 + center_offset_y + args['overlap']:16 + args['inpaint_size'] + center_offset_y - args[
                    'overlap']] = 0

            fake = netG(input_cropped)

            recon_image.data[:, :, 16 + center_offset_x:args['inpaint_size'] + 16 + center_offset_x,
            16 + center_offset_x:args['inpaint_size'] + 16 + center_offset_x] = fake.data

    return recon_image


if __name__ == "__main__":
    DATASET_PATH = './DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # anomalous_dice_calculation()
    # anomalous_validation_1()
    import sys

    if str(sys.argv[1]) == "100":
        unet_anomalous()
    elif str(sys.argv[1]) == "101" or str(sys.argv[1]) == "102" or str(sys.argv[1]) == "103" or str(sys.argv[1]) == \
            "104":
        gan_anomalous()
    else:
        graph_data()
        anomalous_dice_calculation()

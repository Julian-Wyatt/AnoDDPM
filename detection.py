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

            dice_data.append(
                    evaluation.heatmap(
                            img[slice_number, ...].reshape(1, 1, *args["img_size"]), output.to(device),
                            img_mask[slice_number, ...].reshape(1, 1, *args["img_size"]),
                            f"./diffusion-training-images/ARGS={args['arg_num']}/{new['filenames'][0][-9:-4]}"
                            f"-{new['slices'][slice_number].cpu().item()}.png",
                            save=True
                            )
                    )
            ssim_data.append(
                    evaluation.SSIM(
                            img[slice_number, ...].reshape(*args["img_size"]), output.reshape(*args["img_size"])
                            )
                    )

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
                f"remaining time: {hours}:{mins:02.0f}, dice {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}, "
                f"ssim {np.mean(ssim_data[-4:])} +-"
                f" {np.std(ssim_data[-4:])}"
                )

    print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(
            f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
            f" {np.std(ssim_data)}"
            )


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
    t_range = np.linspace(0, 999, 250).astype(np.int32)

    start_time = time.time()
    for i in range(len(ano_dataset)):

        dice_data = []
        ssim_data = []
        precision = []
        recall = []
        IOU = []

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

            if (t + 1) % 100 == 0 or (t - 1) % 100 == 0 or (t) % 100 == 0:
                print(t, dice_data[-1], ssim_data[-1], precision[-1], recall[-1], IOU[-1])

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

        with open(f'./metrics/ARGS={args["arg_num"]}/{new["filenames"][0][-9:-4]}.csv', mode="w") as f:
            f.write(",".join(t_range))
            f.write("\n")
            f.write(",".join(dice_data))
            f.write("\n")
            f.write(",".join(ssim_data))
            f.write("\n")
            f.write(",".join(IOU))
            f.write("\n")
            f.write(",".join(precision))
            f.write("\n")
            f.write(",".join(recall))
            f.write("\n")

        plt.plot(t_range, dice_data, label="dice")
        plt.plot(t_range, IOU, label="IOU")
        plt.plot(t_range, precision, label="precision")
        plt.plot(t_range, recall, label="recall")
        plt.legend(loc="upper right")
        plt.savefig(f'./metrics/ARGS={args["arg_num"]}/{new["filenames"][0][-9:-4]}.png')




if __name__ == "__main__":
    DATASET_PATH = './DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # anomalous_dice_calculation()
    # anomalous_validation_1()
    graph_data()

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


def anomalous_metric_calculation():
    """
    Iterates over 4 anomalous slices for each Volume, returning diffused video for it,
    the heatmap of that & detection method (A&B) or C
    :return:
    """
    args, output = load_parameters(device)
    in_channels = 1
    if args["dataset"].lower() == "leather":
        in_channels = 3

    print(f"args{args['arg_num']}")
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
        d_set_size = len(d_set)
    elif args["dataset"].lower() == "leather":
        d_set = dataset.MVTec(
                "./DATASETS/leather", anomalous=True, img_size=args["img_size"],
                rgb=True, include_good=False
                )
        d_set_size = len(d_set)
    else:
        d_set = dataset.AnomalousMRIDataset(
                ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
                slice_selection="iterateKnown_restricted", resized=False, cleaned=True
                )
        d_set_size = len(d_set) * 4
    loader = dataset.init_dataset_loader(d_set, args)
    plt.rcParams['figure.dpi'] = 200

    dice_data = []
    ssim_data = []
    IOU = []
    precision = []
    recall = []
    FPR = []
    AUC_scores = []

    start_time = time.time()
    for i in range(d_set_size):

        if args["dataset"].lower() != "carpet" and args["dataset"].lower() != "leather":
            if i % 4 == 0:
                new = next(loader)
                new["image"] = new["image"].reshape(new["image"].shape[1], 1, *args["img_size"])
                new["mask"] = new["mask"].reshape(new["mask"].shape[1], 1, *args["img_size"])
            image = new["image"][i % 4, ...].to(device).reshape(1, 1, *args["img_size"])
            mask = new["mask"][i % 4, ...].to(device).reshape(1, 1, *args["img_size"])
        else:
            new = next(loader)
            image = new["image"].to(device)
            mask = new["mask"].to(device)

        output = diff.forward_backward(
                unet, image,
                see_whole_sequence=None,
                t_distance=200, denoise_fn=args["noise_fn"]
                )

        mse = (image - output).square()
        fpr_simplex, tpr_simplex, _ = evaluation.ROC_AUC(mask.to(torch.uint8), mse)
        AUC_scores.append(evaluation.AUC_score(fpr_simplex, tpr_simplex))
        mse = (mse > 0.5).float()
        # print(img.shape, output.shape, img_mask.shape, mse.shape)
        dice_data.append(
                evaluation.dice_coeff(
                        image, output.to(device),
                        mask, mse=mse
                        ).cpu().item()
                )

        ssim_data.append(
                evaluation.SSIM(
                        image.permute(0, 2, 3, 1).reshape(*args["img_size"], image.shape[1]),
                        output.permute(0, 2, 3, 1).reshape(*args["img_size"], image.shape[1])
                        )
                )
        precision.append(evaluation.precision(mask, mse).cpu().numpy())
        recall.append(evaluation.recall(mask, mse).cpu().numpy())
        IOU.append(evaluation.IoU(mask, mse))
        FPR.append(evaluation.FPR(mask, mse).cpu().numpy())
        plt.close('all')

        if i % 8 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = d_set_size - i
            time_per_epoch = time_taken / (i + 1)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)

            print(
                    f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                    f"remaining time: {hours}:{mins:02.0f}"
                    )

        if i % 4 == 0 and (args["dataset"].lower() != "carpet" and args["dataset"].lower() != "leather"):
            print(f"file: {new['filenames'][0][-9:-4]}")
            print(f"Dice: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
            print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +- {np.std(ssim_data[-4:])}")
            print(f"Precision: {np.mean(precision[-4:])} +- {np.std(precision[-4:])}")
            print(f"Recall: {np.mean(recall[-4:])} +- {np.std(recall[-4:])}")
            print(f"FPR: {np.mean(FPR[-4:])} +- {np.std(FPR[-4:])}")
            print(f"IOU: {np.mean(IOU[-4:])} +- {np.std(IOU[-4:])}")
            print("\n")

    print()
    print("Overall: ")
    print(f"Dice coefficient: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data)} +- {np.std(ssim_data)}")
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")
    with open(f"./metrics/args{args['arg_num']}.csv", mode="w") as f:
        f.write("dice,ssim,iou,precision,recall,fpr,auc\n")
        for METRIC in [dice_data, ssim_data, IOU, precision, recall, FPR, AUC_scores]:
            f.write(f"{np.mean(METRIC):.4f} +- {np.std(METRIC):.4f},")


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
    t_range = np.linspace(0, 999, 1000).astype(np.int32)
    # t_range = np.linspace(0, 999, 1).astype(np.int32)

    start_time = time.time()
    files_to_complete = defaultdict(str, {"19691": False, "18756": False})
    for i in range(2):

        dice_data = []
        ssim_data = []
        precision = []
        recall = []
        IOU = []
        FPR = []
        new = next(loader)

        while (new['filenames'][0][-9:-4] not in files_to_complete) or (files_to_complete[new['filenames'][0][
                                                                                          -9:-4]] == True):
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

            if t in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]:
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
        files_to_complete[new['filenames'][0][-9:-4]] = True
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

            f.write(",".join(["timestep", "Dice", "SSIM", "IOU", "Precision", "Recall", "FPR"]))
            f.write("\n")
            for i in range(1000):
                f.write(
                        f"{i:04}," + ",".join(
                                [f"{j:.4f}" for j in [dice_data[i], ssim_data[i], IOU[i], precision[i],
                                                      recall[i], FPR[i]]]
                                )
                        )
                f.write("\n")


def roc_data():
    sys.argv[1] = "28"
    args_simplex, output_simplex = load_parameters(device)
    sys.argv[1] = "27"
    args_hybrid, output_hybrid = load_parameters(device)
    sys.argv[1] = "26"
    args_gauss, output_gauss = load_parameters(device)

    unet_simplex = UNetModel(
            args_simplex['img_size'][0], args_simplex['base_channels'], channel_mults=args_simplex['channel_mults']
            )
    unet_hybrid = UNetModel(
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

    unet_hybrid.load_state_dict(output_hybrid["ema"])
    unet_simplex.load_state_dict(output_simplex["ema"])
    unet_gauss.load_state_dict(output_gauss["ema"])
    unet_simplex.eval()
    unet_gauss.eval()

    import Comparative_models.CE as CE
    sys.argv[1] = "103"
    args_GAN, output_GAN = load_parameters(device)
    args_GAN["Batch_Size"] = 1
    print(args_GAN)
    netG = CE.Generator(
            start_size=args_GAN['img_size'][0], out_size=args_GAN['inpaint_size'], dropout=args_GAN["dropout"]
            )

    netG.load_state_dict(output_GAN["generator_state_dict"])
    netG.eval()
    ano_dataset_128 = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args_GAN['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )

    loader_128 = dataset.init_dataset_loader(ano_dataset_128, args_GAN, False)

    overlapSize = args_GAN['overlap']
    input_cropped = torch.FloatTensor(args_GAN['Batch_Size'], 1, 128, 128)

    ano_dataset_256 = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args_simplex['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )
    loader_256 = dataset.init_dataset_loader(ano_dataset_256, args_simplex, False)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./metrics/')
    except OSError:
        pass

    try:
        os.makedirs(f'./metrics/ROC_data_3/')
    except OSError:
        pass
    t_distance = 250

    simplex_sqe = []
    gauss_sqe = []
    GAN_sqe = []
    hybrid_sqe = []
    img_128 = []
    img_256 = []
    simplex_AUC = []
    gauss_AUC = []
    GAN_AUC = []
    hybrid_AUC = []
    for i in range(len(ano_dataset_256)):

        new_256 = next(loader_256)
        img_256_whole = new_256["image"].to(device)
        img_256_whole = img_256_whole.reshape(img_256_whole.shape[1], 1, *args_simplex["img_size"])
        img_mask_256_whole = dataset.load_image_mask(
                new_256['filenames'][0][-9:-4], args_simplex['img_size'],
                ano_dataset_256
                )
        img_mask_256_whole = img_mask_256_whole.to(device)
        img_mask_256_whole = (img_mask_256_whole > 0).float()

        new_128 = next(loader_128)
        img_128_whole = new_128["image"].to(device)
        img_128_whole = img_128_whole.reshape(img_128_whole.shape[1], 1, *args_GAN["img_size"])
        img_mask_128_whole = dataset.load_image_mask(
                new_128['filenames'][0][-9:-4], args_GAN['img_size'],
                ano_dataset_128
                )

        for slice_number in range(4):
            img = img_256_whole[slice_number, ...].reshape(1, 1, *args_simplex["img_size"])
            img_mask = img_mask_256_whole[slice_number, ...].reshape(1, 1, *args_simplex["img_size"])
            img_256.append(img_mask.detach().cpu().numpy().flatten())
            # for slice_number in range(4):
            unet_simplex.to(device)

            output_simplex = diff_simplex.forward_backward(
                    unet_simplex, img.reshape(1, 1, *args_simplex["img_size"]),
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_simplex["noise_fn"]
                    )

            unet_simplex.cpu()

            mse_simplex = (img - output_simplex).square()
            simplex_sqe.append(mse_simplex.detach().cpu().numpy().flatten())

            fpr_simplex, tpr_simplex, threshold_simplex = evaluation.ROC_AUC(img_mask, mse_simplex)
            simplex_AUC.append(evaluation.AUC_score(fpr_simplex, tpr_simplex))

            unet_hybrid.to(device)

            output_hybrid = diff_simplex.forward_backward(
                    unet_hybrid, img.reshape(1, 1, *args_hybrid["img_size"]),
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_hybrid["noise_fn"]
                    )

            unet_hybrid.cpu()

            mse_hybrid = (img - output_hybrid).square()
            hybrid_sqe.append(mse_hybrid.detach().cpu().numpy().flatten())

            fpr_hybrid, tpr_hybrid, threshold_hybrid = evaluation.ROC_AUC(img_mask, mse_hybrid)
            hybrid_AUC.append(evaluation.AUC_score(fpr_hybrid, tpr_hybrid))

            unet_gauss.to(device)
            output_gauss = diff_gauss.forward_backward(
                    unet_gauss, img.reshape(1, 1, *args_simplex["img_size"]),
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_gauss["noise_fn"]
                    )

            unet_gauss.cpu()

            mse_gauss = (img - output_gauss).square()
            gauss_sqe.append(mse_gauss.detach().cpu().numpy().flatten())
            fpr_gauss, tpr_gauss, threshold_gauss = evaluation.ROC_AUC(img_mask, mse_gauss)
            gauss_AUC.append(evaluation.AUC_score(fpr_gauss, tpr_gauss))

            img = img_128_whole[slice_number, ...].reshape(1, 1, *args_GAN["img_size"]).to(device)
            img_mask = img_mask_128_whole[slice_number, ...].to(device)
            img_mask = (img_mask > 0).float().reshape(1, 1, *args_GAN["img_size"])
            img_mask_center = img_mask[:, :,
                              args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4,
                              args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4]
            img_center = img[:, :, args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4,
                         args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4]
            img_128.append(img_mask_center.detach().cpu().numpy().flatten())
            input_cropped = input_cropped.to(device)
            netG.to(device)
            input_cropped.resize_(img.size()).copy_(img)
            # recon_image = input_cropped.clone()
            with torch.no_grad():
                input_cropped[:, 0,
                args_GAN['img_size'][0] // 4 + overlapSize:
                args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4 - overlapSize,
                args_GAN['img_size'][0] // 4 + overlapSize:
                args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4 - overlapSize] \
                    = 0

            fake = netG(input_cropped)
            # print(fake.shape, img.shape, recon_image.shape)
            # recon_image.data[:, :, args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4,
            # args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4] = fake.data

            mse_GAN = (img_center - fake).square()
            GAN_sqe.append(mse_GAN.detach().cpu().numpy().flatten())
            fpr_GAN, tpr_GAN, threshold_GAN = evaluation.ROC_AUC(img_mask_center, mse_GAN)
            GAN_AUC.append(evaluation.AUC_score(fpr_GAN, tpr_GAN))

            input_cropped.cpu()
            netG.cpu()

            plt.plot(fpr_gauss, tpr_gauss, ":", label=f"gauss AUC={gauss_AUC[-1]:.2f}")
            plt.plot(fpr_simplex, tpr_simplex, "-", label=f"simplex AUC={simplex_AUC[-1]:.2f}")
            plt.plot(fpr_GAN, tpr_GAN, "-.", label=f"GAN AUC={GAN_AUC[-1]:.2f}")
            plt.legend()
            ax = plt.gca()
            ax.set_ylim([0, 1])
            ax.set_xlim([0, 1])
            plt.savefig(
                    f'./metrics/ROC_data_3/{new_128["filenames"][0][-9:-4]}'
                    f'-{new_128["slices"][slice_number].cpu().item()}.png'
                    )
            plt.clf()

    simplex_sqe = np.array(simplex_sqe)
    gauss_sqe = np.array(gauss_sqe)
    GAN_sqe = np.array(GAN_sqe)
    hybrid_sqe = np.array(hybrid_sqe)
    img_256 = np.array(img_256)
    img_128 = np.array(img_128)

    fpr_simplex, tpr_simplex, _ = evaluation.ROC_AUC(img_256, simplex_sqe)
    fpr_gauss, tpr_gauss, _ = evaluation.ROC_AUC(img_256, gauss_sqe)
    fpr_GAN, tpr_GAN, _ = evaluation.ROC_AUC(img_128, GAN_sqe)
    fpr_hybrid, tpr_hybrid, _ = evaluation.ROC_AUC(img_256, hybrid_sqe)

    for model in [(fpr_simplex, tpr_simplex, "simplex"), (fpr_gauss, tpr_gauss, "gauss"), (fpr_GAN, tpr_GAN, "GAN"),
                  (fpr_hybrid, tpr_hybrid, "hybrid")]:

        with open(f'./metrics/ROC_data_2/overall_{model[2]}.csv', mode="w") as f:
            f.write(f"fpr, tpr, {evaluation.AUC_score(model[0], model[1])}")
            f.write("\n")
            for i in range(len(model[0])):
                f.write(",".join([f"{j:.4f}" for j in [model[0][i], model[1][i]]]))
                f.write("\n")

    plt.plot(fpr_gauss, tpr_gauss, ":", label=f"Gaussian AUC={evaluation.AUC_score(fpr_gauss, tpr_gauss):.3f}")
    plt.plot(
            fpr_simplex, tpr_simplex, "-",
            label=f"Simplex $\mathcal{{L}}_{{simple}}$ AUC={evaluation.AUC_score(fpr_simplex, tpr_simplex):.3f}"
            )
    plt.plot(
            fpr_hybrid, tpr_hybrid, "-", label=f"Simplex $\mathcal{{L}}_{{hybrid}}$ AUC"
                                               f"={evaluation.AUC_score(fpr_hybrid, tpr_hybrid):.3f}"
            )
    plt.plot(
            fpr_GAN, tpr_GAN, "-.",
            label=f"Adversarial Context Encoder AUC={evaluation.AUC_score(fpr_GAN, tpr_GAN):.3f}"
            )
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(f'./metrics/ROC_data_2/Overall.png')
    plt.clf()

    print(f"Simplex AUC {np.mean(simplex_AUC)} +- {np.std(simplex_AUC)}")
    print(f"Simplex hybrid AUC {np.mean(hybrid_AUC)} +- {np.std(hybrid_AUC)}")
    print(f"Gauss AUC {np.mean(gauss_AUC)} +- {np.std(gauss_AUC)}")
    print(f"CE AUC {np.mean(GAN_AUC)} +- {np.std(GAN_AUC)}")


def gan_anomalous():
    import Comparative_models.CE as CE
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

            if args['type'] == 'sliding':
                recon_image = ce_sliding_window(img, netG, input_cropped, args)
            else:
                input_cropped.resize_(img.size()).copy_(img)
                recon_image = input_cropped.clone()
                with torch.no_grad():
                    input_cropped[:, 0,
                    args['img_size'][0] // 4 + overlapSize:
                    args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize,
                    args['img_size'][0] // 4 + overlapSize:
                    args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize] \
                        = 0

                fake = netG(input_cropped)

                recon_image.data[:, :, args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4,
                args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4] = fake.data

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
        print("\n")

    print()
    print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(
            f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
            f" {np.std(ssim_data)}"
            )
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")

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

            img_center = img[:, :, args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4,
                         args['img_size'][0] // 4: args['inpaint_size'] + args['img_size'][0] // 4]
            img_mask_center = img_mask[:, :, args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4,
                              args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4]

            input_cropped.resize_(img.size()).copy_(img)
            with torch.no_grad():
                input_cropped[:, 0,
                args['img_size'][0] // 4 + overlapSize:
                args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize,
                args['img_size'][0] // 4 + overlapSize:
                args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize] \
                    = 0

            fake = netG(input_cropped)

            mse = (img_center - fake).square()
            mse = (mse > 0.5).float()
            dice_data.append(
                    evaluation.dice_coeff(img_center, fake, img_mask_center, mse=mse).detach().cpu().numpy()
                    )
            ssim_data.append(
                    evaluation.SSIM(
                            img_center.reshape(args["inpaint_size"], args["inpaint_size"]),
                            fake.reshape(args["inpaint_size"], args["inpaint_size"])
                            )
                    )
            precision.append(evaluation.precision(img_mask_center, mse).detach().cpu().numpy())
            recall.append(evaluation.recall(img_mask_center, mse).detach().cpu().numpy())
            IOU.append(evaluation.IoU(img_mask_center, mse))
            FPR.append(evaluation.FPR(img_mask_center, mse).detach().cpu().numpy())
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
        print("\n")

    print()
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
    input_cropped.resize_(img.size()).copy_(img)
    recon_image = input_cropped.clone()
    for center_offset_y in np.arange(0, 97, args['inpaint_size']):

        for center_offset_x in np.arange(0, 97, args['inpaint_size']):
            with torch.no_grad():
                input_cropped.resize_(img.size()).copy_(img)
                input_cropped[:, 0,
                center_offset_x + args['overlap']: args['inpaint_size'] + center_offset_x - args['overlap'],
                center_offset_y + args['overlap']: args['inpaint_size'] + center_offset_y - args[
                    'overlap']] = 0

            fake = netG(input_cropped)

            recon_image.data[:, :, center_offset_x:args['inpaint_size'] + center_offset_x,
            center_offset_y:args['inpaint_size'] + center_offset_y] = fake.data

    return recon_image


if __name__ == "__main__":
    import sys
    from matplotlib import font_manager

    font_path = "./times new roman.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    DATASET_PATH = './DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if str(sys.argv[1]) == "100":
    #     unet_anomalous()
    if str(sys.argv[1]) == "101" or str(sys.argv[1]) == "102" or str(sys.argv[1]) == "103" or str(sys.argv[1]) == \
            "104":
        gan_anomalous()
    elif str(sys.argv[1]) == "200":
        roc_data()
    elif str(sys.argv[1]) == "500":
        sys.argv[1] = "26"
        anomalous_metric_calculation()
        sys.argv[1] = "28"
        anomalous_metric_calculation()
        sys.argv[1] = "103"
        gan_anomalous()
    elif str(sys.argv[1]) == "201":
        sys.argv[1] = "26"
        graph_data()
        sys.argv[1] = "28"
        graph_data()
    else:

        anomalous_metric_calculation()

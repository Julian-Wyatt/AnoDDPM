import collections
import copy
import sys
import time
from random import seed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from torch import optim

import dataset
import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel, update_ema_params

torch.cuda.empty_cache()

ROOT_DIR = "./"


def train(training_dataset_loader, testing_dataset_loader, args, resume):
    """

    :param training_dataset_loader: cycle(dataloader) instance for training
    :param testing_dataset_loader:  cycle(dataloader) instance for testing
    :param args: dictionary of parameters
    :param resume: dictionary of parameters if continuing training from checkpoint
    :return: Trained model and tested
    """

    in_channels = 1
    if args["dataset"].lower() == "cifar" or args["dataset"].lower() == "leather":
        in_channels = 3

    if args["channels"] != "":
        in_channels = args["channels"]

    model = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )

    if resume:

        if "unet" in resume:
            model.load_state_dict(resume["unet"])
        else:
            model.load_state_dict(resume["ema"])

        ema = UNetModel(
                args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'],
                dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                in_channels=in_channels
                )
        ema.load_state_dict(resume["ema"])
        start_epoch = resume['n_epoch']

    else:
        start_epoch = 0
        ema = copy.deepcopy(model)

    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)
    model.to(device)
    ema.to(device)
    optimiser = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], betas=(0.9, 0.999))
    if resume:
        optimiser.load_state_dict(resume["optimizer_state_dict"])

    del resume
    start_time = time.time()
    losses = []
    vlb = collections.deque([], maxlen=10)
    iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(200)
    # iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(150)

    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []

        for i in iters:
            data = next(training_dataset_loader)
            if args["dataset"] == "cifar":
                # cifar outputs [data,class]
                x = data[0].to(device)
            else:
                x = data["image"]
                x = x.to(device)

            loss, estimates = diffusion.p_loss(model, x, args)

            noisy, est = estimates[1], estimates[2]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())

            if epoch % 50 == 0 and i == 0:
                row_size = min(8, args['Batch_Size'])
                training_outputs(
                        diffusion, x, est, noisy, epoch, row_size, save_imgs=args['save_imgs'],
                        save_vids=args['save_vids'], ema=ema, args=args
                        )

        losses.append(np.mean(mean_loss))
        if epoch % 200 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)

            vlb_terms = diffusion.calc_total_vlb(x, model, args)
            vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
            print(
                    f"epoch: {epoch}, most recent total VLB: {vlb[-1]} mean total VLB:"
                    f" {np.mean(vlb):.4f}, "
                    f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, vb: "
                    f"{torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, x_0_mse: "
                    f"{torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, mse: "
                    f"{torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f}"
                    f" time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                    f"est time remaining: {hours}:{mins:02.0f}\r"
                    )
            # else:
            #
            #     print(
            #             f"epoch: {epoch}, imgs trained: {(i + 1) * args['Batch_Size'] + epoch * 100}, last 20 epoch mean loss:"
            #             f" {np.mean(losses[-20:]):.4f} , last 100 epoch mean loss:"
            #             f" {np.mean(losses[-100:]) if len(losses) > 0 else 0:.4f}, "
            #             f"time per epoch {time_per_epoch:.2f}s, time elapsed {int(time_taken / 3600)}:"
            #             f"{((time_taken / 3600) % 1) * 60:02.0f}, est time remaining: {hours}:{mins:02.0f}\r"
            #             )

        if epoch % 1000 == 0 and epoch >= 0:
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)

    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)

    evaluation.testing(testing_dataset_loader, diffusion, ema=ema, args=args, model=model)


def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
    """
    Save model final or checkpoint
    :param final: bool for final vs checkpoint
    :param unet: unet instance
    :param optimiser: ADAM optim
    :param args: model parameters
    :param ema: ema instance
    :param loss: loss for checkpoint
    :param epoch: epoch for checkpoint
    :return: saved model
    """
    if final:
        torch.save(
                {
                    'n_epoch':              args["EPOCHS"],
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "ema":                  ema.state_dict(),
                    "args":                 args
                    # 'loss': LOSS,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
                )
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/checkpoint/diff_epoch={epoch}.pt'
                )


def training_outputs(diffusion, x, est, noisy, epoch, row_size, ema, args, save_imgs=False, save_vids=False):
    """
    Saves video & images based on args info
    :param diffusion: diffusion model instance
    :param x: x_0 real data value
    :param est: estimate of the noise at x_t (output of the model)
    :param noisy: x_t
    :param epoch:
    :param row_size: rows for outputs into torchvision.utils.make_grid
    :param ema: exponential moving average unet for sampling
    :param save_imgs: bool for saving imgs
    :param save_vids: bool for saving diffusion videos
    :return:
    """
    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}')
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}')
    except OSError:
        pass
    if save_imgs:
        if epoch % 100 == 0:
            # for a given t, output x_0, & prediction of x_(t-1), and x_0
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(ema, x_t, t)
            out = torch.cat(
                    (x[:row_size, ...].cpu(), temp["sample"][:row_size, ...].cpu(),
                     temp["pred_x_0"][:row_size, ...].cpu())
                    )
            plt.title(f'real,sample,prediction x_0-{epoch}epoch')
        else:
            # for a given t, output x_0, x_t, & prediction of noise in x_t & MSE
            out = torch.cat(
                    (x[:row_size, ...].cpu(), noisy[:row_size, ...].cpu(), est[:row_size, ...].cpu(),
                     (est - noisy).square().cpu()[:row_size, ...])
                    )
            plt.title(f'real,noisy,noise prediction,mse-{epoch}epoch')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap='gray')

        plt.savefig(f'./diffusion-training-images/ARGS={args["arg_num"]}/EPOCH={epoch}.png')
        plt.clf()
    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 500 == 0:
            plt.rcParams['figure.dpi'] = 200
            if epoch % 1000 == 0:
                out = diffusion.forward_backward(ema, x, "half", args['sample_distance'] // 2, denoise_fn="noise_fn")
            else:
                out = diffusion.forward_backward(ema, x, "half", args['sample_distance'] // 4, denoise_fn="noise_fn")
            imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            ani = animation.ArtistAnimation(
                    fig, imgs, interval=50, blit=True,
                    repeat_delay=1000
                    )

            ani.save(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/sample-EPOCH={epoch}.mp4')

    plt.close('all')


def main():
    """
        Load arguments, run training and testing functions, then remove checkpoint directory
    :return:
    """
    # make directories
    for i in ['./model/', "./diffusion-videos/", './diffusion-training-images/']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    # read file from argument
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")

    # resume from final or resume from most recent checkpoint -> ran from specific slurm script?
    resume = 0
    if files[0] == "RESUME_RECENT":
        resume = 1
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")
    elif files[0] == "RESUME_FINAL":
        resume = 2
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")

    # allow different arg inputs ie 25 or args15 which are converted into argsNUM.json
    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    else:
        raise ValueError("File Argument is not a json file")

    # load the json args
    with open(f'{ROOT_DIR}test_args/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    # make arg specific directories
    for i in [f'./model/diff-params-ARGS={args["arg_num"]}',
              f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint',
              f'./diffusion-videos/ARGS={args["arg_num"]}',
              f'./diffusion-training-images/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    print(file, args)
    if args["channels"] != "":
        in_channels = args["channels"]

    # if dataset is cifar, load different training & test set
    if args["dataset"].lower() == "cifar":
        training_dataset_loader_, testing_dataset_loader_ = dataset.load_CIFAR10(args, True), \
                                                            dataset.load_CIFAR10(args, False)
        training_dataset_loader = dataset.cycle(training_dataset_loader_)
        testing_dataset_loader = dataset.cycle(testing_dataset_loader_)
    elif args["dataset"].lower() == "carpet":
        training_dataset = dataset.DAGM(
                "./DATASETS/CARPET/Class1", False, args["img_size"],
                False
                )
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset = dataset.DAGM(
                "./DATASETS/CARPET/Class1", True, args["img_size"],
                False
                )
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    elif args["dataset"].lower() == "leather":
        if in_channels == 3:
            training_dataset = dataset.MVTec(
                    "./DATASETS/leather", anomalous=False, img_size=args["img_size"],
                    rgb=True
                    )
            testing_dataset = dataset.MVTec(
                    "./DATASETS/leather", anomalous=True, img_size=args["img_size"],
                    rgb=True, include_good=True
                    )
        else:
            training_dataset = dataset.MVTec(
                    "./DATASETS/leather", anomalous=False, img_size=args["img_size"],
                    rgb=False
                    )
            testing_dataset = dataset.MVTec(
                    "./DATASETS/leather", anomalous=True, img_size=args["img_size"],
                    rgb=False, include_good=True
                    )
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    else:
        # load NFBS dataset
        training_dataset, testing_dataset = dataset.init_datasets(ROOT_DIR, args)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    # if resuming, loaded model is attached to the dictionary
    loaded_model = {}
    if resume:
        if resume == 1:
            checkpoints = os.listdir(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint')
            checkpoints.sort(reverse=True)
            for i in checkpoints:
                try:
                    file_dir = f"./model/diff-params-ARGS={args['arg_num']}/checkpoint/{i}"
                    loaded_model = torch.load(file_dir, map_location=device)
                    break
                except RuntimeError:
                    continue

        else:
            file_dir = f'./model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
            loaded_model = torch.load(file_dir, map_location=device)

    # load, pass args
    train(training_dataset_loader, testing_dataset_loader, args, loaded_model)

    # remove checkpoints after final_param is saved (due to storage requirements)
    for file_remove in os.listdir(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint'):
        os.remove(os.path.join(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint', file_remove))
    os.removedirs(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)

    main()

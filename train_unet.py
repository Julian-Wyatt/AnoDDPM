import collections
import copy
import time
from random import seed

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from torch import optim

import dataset
import evaluation
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

    in_channels = 3 if args["dataset"].lower() == "cifar" else 1

    model = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels
            )

    start_epoch = 0
    ema = copy.deepcopy(model)

    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)
    model.to(device)
    ema.to(device)
    optimiser = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], betas=(0.9, 0.999))
    if resume:
        optimiser.load_state_dict(resume["optimizer_state_dict"])

    loss_criterion = torch.nn.MSELoss().to(device)
    del resume
    start_time = time.time()
    losses = []
    vlb = collections.deque([], maxlen=10)
    iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(256)
    # iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(150)

    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []

        for i in iters:
            data = next(training_dataset_loader)
            if args["dataset"] != "cifar":
                x = data["image"]
                x = x.to(device)
            else:
                # cifar outputs [data,class]
                x = data[0].to(device)

            t_batch = torch.tensor([1], device=x.device).repeat(x.shape[0])
            x_hat = model(x, t_batch)
            loss = loss_criterion(x_hat, x)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())

            if i == 0 and epoch % 50 == 0:
                training_outputs(x, x_hat, epoch, ema=ema, args=args, row_size=4, filename=f"training-epoch={epoch}")

        losses.append(np.mean(mean_loss))
        if epoch % 200 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)

            print(
                    f"epoch: {epoch}, imgs trained: {(i + 1) * args['Batch_Size'] + epoch * 100}, last 20 epoch mean loss:"
                    f" {np.mean(losses[-20:]):.4f} , last 100 epoch mean loss:"
                    f" {np.mean(losses[-100:]) if len(losses) > 0 else 0:.4f}, "
                    f"time per epoch {time_per_epoch:.2f}s, time elapsed {int(time_taken / 3600)}:"
                    f"{((time_taken / 3600) % 1) * 60:02.0f}, est time remaining: {hours}:{mins:02.0f}\r"
                    )

        if epoch % 1000 == 0 and epoch >= 0:
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)

    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)

    testing(testing_dataset_loader, args, ema, model)


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


def training_outputs(x, x_hat, epoch, row_size, ema, args, filename):
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
    ema.eval()
    if epoch % 100 == 0:
        t_batch = torch.tensor([1], device=x.device).repeat(x.shape[0])
        x_ema = ema(x, t_batch)
        # output real, x_hat, recon_ema, mse_ema
        out = torch.cat(
                (x[:row_size, ...], x_hat[:row_size, ...],
                 x_ema[:row_size, ...], (((x - x_ema).square() * 2) - 1)[:row_size, ...]
                 )
                ).cpu()
        plt.title(f'real,$x_{{hat}}$,$x_{{ema}},$mse_{{ema}}$-{epoch}epoch')
    else:
        # for a given t, output x_0, x_t, & prediction of noise in x_t & MSE
        out = torch.cat(
                (x[:row_size, ...], x_hat[:row_size, ...],
                 (((x - x_hat).square() * 2) - 1)[:row_size, ...]
                 )
                ).cpu()
        plt.title(f'real,$x_{{hat}}$,$mse_{{hat}}$-{epoch}epoch')
    plt.rcParams['figure.dpi'] = 150
    plt.grid(False)
    plt.imshow(gridify_output(out, row_size), cmap='gray')
    if not filename:
        filename = f"EPOCH={epoch}.png"
    plt.savefig(f'./diffusion-training-images/ARGS={args["arg_num"]}/{filename}')
    plt.clf()

    plt.close('all')
    ema.train()


def testing(testing_dataset_loader, args, ema, model):
    """
    Samples videos on test set & calculates some metrics such as PSNR & VLB.
    PSNR for diffusion is found by sampling x_0 to T//2 and then finding a prediction of x_0

    :param testing_dataset_loader: The cycle(dataloader) object for looping through test set
    :param diffusion: Gaussian Diffusion model instance
    :param args: parameters of the model
    :param ema: exponential moving average unet for sampling
    :param model: original unet for VLB calc
    :return: outputs:
                total VLB    mu +- sigma,
                prior VLB    mu +- sigma,
                vb -> T      mu +- sigma,
                x_0 mse -> T mu +- sigma,
                mse -> T     mu +- sigma,
                PSNR         mu +- sigma
    """
    import os
    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/')
    except OSError:
        pass
    ema.eval()
    model.eval()

    test_iters = 40

    # vlb = []
    # for epoch in range(test_iters // args["Batch_Size"] + 5):
    #     data = next(testing_dataset_loader)
    #     if args["dataset"] != "cifar":
    #         x = data["image"]
    #         x = x.to(device)
    #     else:
    #         # cifar outputs [data,class]
    #         x = data[0].to(device)
    #
    #     vlb_terms = diffusion.calc_total_vlb(x, model, args)
    #     vlb.append(vlb_terms)

    psnr = []
    for epoch in range(test_iters // args["Batch_Size"] + 5):
        data = next(testing_dataset_loader)

        x = data["image"]
        x = x.to(device)
        t_batch = torch.tensor([1], device=x.device).repeat(x.shape[0])
        x_hat = ema(x, t_batch)

        training_outputs(x, x_hat, epoch * 50, ema=ema, args=args, row_size=4, filename=f"testing-epoch={epoch}")

        psnr.append(evaluation.PSNR(x_hat, x))
    print(f"Test set PSNR: {np.mean(psnr)} +- {np.std(psnr)}")


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

    args = {
        "img_size":          [256, 256],
        "EPOCHS":            3000,
        "base_channels":     128,
        "channel_mults":     "",
        "Batch_Size":        1,
        "arg_num":           100,
        "dataset":           "mri",
        "num_heads":         2,
        "num_head_channels": -1,
        "dropout":           0.2,
        "lr":                1e-4,
        "weight_decay":      0.0,
        'random_slice':      True,
        }

    # make arg specific directories
    for i in [f'./model/diff-params-ARGS={args["arg_num"]}',
              f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint',
              f'./diffusion-videos/ARGS={args["arg_num"]}',
              f'./diffusion-training-images/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    # if dataset is cifar, load different training & test set
    if args["dataset"].lower() == "cifar":
        training_dataset_loader_, testing_dataset_loader_ = dataset.load_CIFAR10(args, True), \
                                                            dataset.load_CIFAR10(args, False)
        training_dataset_loader = dataset.cycle(training_dataset_loader_)
        testing_dataset_loader = dataset.cycle(testing_dataset_loader_)
    else:
        # load NFBS dataset
        training_dataset, testing_dataset = dataset.init_datasets(ROOT_DIR, args)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    # if resuming, loaded model is attached to the dictionary
    loaded_model = {}

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

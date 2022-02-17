import collections
import copy
import json
import os
import sys
import time
from random import seed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from torch import optim

import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel, update_ema_params

torch.cuda.empty_cache()

ROOT_DIR = "./"


def train(training_dataset_loader, testing_dataset_loader, args, resume):
    if args["dataset"].lower() == "cifar":
        in_channels = 3
    else:
        in_channels = 1
    model = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"]
            )

    if resume:
        if "unet" in resume:
            model.load_state_dict(resume["unet"])
        else:
            model.load_state_dict(resume["ema"])

        ema = UNetModel(
                args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                    "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                in_channels=in_channels
                )
        ema.load_state_dict(resume["ema"])
        start_epoch = resume['n_epoch']

    else:
        start_epoch = 0
        ema = copy.deepcopy(model)

    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)

    ema.to(device)
    model.to(device)

    optimiser = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], betas=(0.9, 0.999))
    if resume:
        optimiser.load_state_dict(resume["optimizer_state_dict"])
    # optimiser.to(device)
    del resume
    startTime = time.time()
    losses = []
    vlb = collections.deque([], maxlen=200)
    # tqdm_epoch = tqdm.trange(args['EPOCHS'])
    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []
        for i in range(100 // args['Batch_Size']):
            data = next(training_dataset_loader)
            x = data["image"]
            x = x.to(device)
            loss, estimates = diffusion.p_loss(model, x, args)
            loss_terms, noisy, est = estimates
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            update_ema_params(ema, model)

            if "vlb" in loss_terms:
                vlb.append(loss_terms["vlb"])

            mean_loss.append(loss.data.cpu())
            # print(f"imgs trained: {(1 + i) * args['Batch_Size'] + epoch * 100}, loss: {loss.data.cpu():.2f} ,"
            #       f"'last epoch mean loss': {losses[-1] if len(losses)>0 else 0:.4f}\r")
            # tqdm_epoch.set_postfix({"imgs trained": (1 + i) * args['Batch_Size'] + epoch * 100, "loss": loss.data.cpu() ,'last epoch mean loss': losses[-1] if len(losses)>0 else 0})
            if epoch % 50 == 0 and i == 0:
                row_size = min(8, args['Batch_Size'])
                training_outputs(
                        diffusion, x, est, noisy, epoch, row_size, save_imgs=args['save_imgs'],
                        save_vids=args['save_vids'], ema=ema
                        )
                # save(diffusion=diffusion, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)

        losses.append(np.mean(mean_loss))
        if epoch % 100 == 0:
            timeTaken = time.time() - startTime
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = timeTaken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)
            print(
                    f"epoch: {epoch}, imgs trained: {(i + 1) * args['Batch_Size'] + epoch * 100}, last 20 epoch mean loss:"
                    f" {np.mean(losses[-20:]):.4f} , last 100 epoch mean loss:"
                    f" {np.mean(losses[-100:]) if len(losses) > 0 else 0:.4f}, VLB: {np.mean(vlb)}"
                    f"time per epoch {time_per_epoch:.2f}s, time elapsed {int(timeTaken / 3600)}:"
                    f"{((timeTaken / 3600) % 1) * 60:02.0f}, est time remaining: {hours}:{mins:02.0f}\r"
                    )

        if epoch % 1000 == 0 and epoch >= 0:
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)

    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)

    testing(testing_dataset_loader, diffusion, ema=ema, args=args, device=device)


def testing(testing_dataset_loader, diffusion, args, ema, device=torch.device('cpu')):
    try:
        os.makedirs(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/test-set/')
    except OSError:
        pass
    plt.rcParams['figure.dpi'] = 200
    # for i in [args['sample_distance'], args['T'] / 4, None]:
    # for i in [*range(1,args['sample_distance']),args['T']]:
    vlb = []
    for i in [*range(25, args['sample_distance'], 25), args["T"], args["T"], args["T"], args["T"], args["T"]]:
        data = next(testing_dataset_loader)
        x = data["image"]
        x = x.to(device)
        row_size = min(5, args['Batch_Size'])

        fig, ax = plt.subplots()
        out = diffusion.forward_backward(ema, x, see_whole_sequence="half", t_distance=i)
        imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
        ani = animation.ArtistAnimation(
                fig, imgs, interval=200, blit=True,
                repeat_delay=1000
                )

        files = os.listdir(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/test-set/')
        ani.save(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/test-set/t={i}-attempts={len(files) + 1}.mp4')

        t_tensor = torch.tensor([i], device=x.device).repeat(x.shape[0])
        eps = diffusion.noise_fn(x, t_tensor)
        x_t = diffusion.sample_q(x, t_tensor, eps)
        vlb.append(diffusion.calc_vlb(ema, x, x_t, t_tensor))
        print(f"saved {i}, VLB: {vlb[-1].detach().cpu().numpy()}")
    print(f"Test set VLB: {np.mean(vlb)}")


def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
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



def training_outputs(diffusion, x, est, noisy, epoch, row_size, ema, save_imgs=False, save_vids=False):
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
                out = diffusion.forward_backward(ema, x, "whole", args['sample_distance'])
            else:
                out = diffusion.forward_backward(ema, x, "half", args['sample_distance'])
            imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            ani = animation.ArtistAnimation(
                    fig, imgs, interval=100, blit=True,
                    repeat_delay=1000
                    )

            ani.save(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/sample-EPOCH={epoch}.mp4')

    plt.close('all')



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)

    for i in ['./model/', "./diffusion-videos/", './diffusion-training-images/']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")

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

    file = files[0]
    if file[-5:] == ".json":
        with open(f'{ROOT_DIR}test_args/{file}', 'r') as f:
            args = json.load(f)
        args['arg_num'] = file[4:-5]
        args = defaultdict_from_json(args)
        for i in [f'./model/diff-params-ARGS={args["arg_num"]}',
                  f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint',
                  f'./diffusion-videos/ARGS={args["arg_num"]}',
                  f'./diffusion-training-images/ARGS={args["arg_num"]}']:
            try:
                os.makedirs(i)
            except OSError:
                pass
        print(file, args)
        if args["dataset"].lower() == "cifar":
            training_dataset, testing_dataset = dataset.load_CIFAR10(args, True), dataset.load_CIFAR10(args, False)
        else:
            training_dataset, testing_dataset = dataset.init_datasets(ROOT_DIR, args)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

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

        for f in os.listdir(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint'):
            os.remove(os.path.join(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint', f))
        os.removedirs(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint')

    else:
        raise ValueError("File Argument is not a json file")

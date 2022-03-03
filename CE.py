import time
from random import seed

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from torch import optim

import dataset
import evaluation
from helpers import *

torch.cuda.empty_cache()

ROOT_DIR = "./"


# https://towardsdatascience.com/inpainting-with-ai-get-back-your-images-pytorch-a68f689128e5

class Generator(torch.nn.Module):

    # generator model
    def __init__(self, start_size=256, out_size=64, dropout=0.3, bottleneck=4000):
        super(Generator, self).__init__()
        if start_size == 256:
            enc_channels = [1, 64, 64, 128, 256, 512, 1024]
        else:
            enc_channels = [1, 64, 128, 256, 512]

        if out_size == 64:
            dec_channels = [512, 256, 128, 1]
        else:
            dec_channels = [256, 128, 64, 1]

        self.encoder = []
        in_channels = enc_channels[0]
        for i in enc_channels[1:]:
            self.encoder.append(
                    torch.nn.Sequential(
                            torch.nn.Conv2d(
                                    in_channels=in_channels, out_channels=i, kernel_size=(4, 4), stride=2, padding=1
                                    ),
                            torch.nn.BatchNorm2d(i),
                            torch.nn.LeakyReLU(0.2)
                            )
                    )
            in_channels = i

        self.bottom = torch.nn.Sequential(
                torch.nn.Conv2d(enc_channels[-1], bottleneck, kernel_size=(4, 4)),  # bottleneck
                torch.nn.BatchNorm2d(bottleneck),
                torch.nn.ReLU()
                )

        self.decoder = [torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                        in_channels=bottleneck, out_channels=dec_channels[0], kernel_size=(4, 4), stride=1,
                        padding=0
                        ),
                torch.nn.BatchNorm2d(dec_channels[0]),
                torch.nn.ReLU()
                )]
        out_channels = dec_channels[0]
        for i in dec_channels[1:-1]:
            self.decoder.append(
                    torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(
                                    in_channels=out_channels, out_channels=i, kernel_size=(4, 4), stride=2, padding=1
                                    ),
                            torch.nn.BatchNorm2d(i),
                            torch.nn.ReLU()
                            )
                    )
            out_channels = i

        self.output = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                        in_channels=out_channels, out_channels=1, kernel_size=(4, 4), stride=2,
                        padding=1
                        ),
                torch.nn.Tanh()
                )
        self.dropout = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Sequential(*self.encoder)
        self.decoder = torch.nn.Sequential(*self.decoder)

    def forward(self, x):

        x = self.encoder(x)
        x = self.bottom(x)
        x = self.decoder(x)
        return self.output(x)


class Discriminator(torch.nn.Module):

    # discriminator model
    def __init__(self, in_size=64, dropout=0.3, ):
        super(Discriminator, self).__init__()

        if in_size == 64:
            channels = [1, 64, 128, 256, 512, 1]
        elif in_size == 32:
            channels = [1, 64, 128, 256, 1]

        self.encode = []
        in_channels = channels[0]
        for i in channels[1:-1]:
            self.encode.append(
                    torch.nn.Sequential(
                            torch.nn.Conv2d(
                                    in_channels=in_channels, out_channels=i, kernel_size=(4, 4), stride=2, padding=1
                                    ),
                            torch.nn.LeakyReLU(0.2),
                            torch.nn.Dropout(dropout)
                            )
                    )
            in_channels = i

        self.encode.append(
                torch.nn.Sequential(
                        torch.nn.Conv2d(
                                in_channels=in_channels, out_channels=channels[-1], kernel_size=(4, 4), stride=1,
                                padding=0
                                ),
                        torch.nn.Sigmoid()
                        )
                )

        self.encode = torch.nn.Sequential(*self.encode)

    def forward(self, x):
        return self.encode(x)  # output of discriminator



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(training_dataset_loader, testing_dataset_loader, args, resume):
    """

    :param training_dataset_loader: cycle(dataloader) instance for training
    :param testing_dataset_loader:  cycle(dataloader) instance for testing
    :param args: dictionary of parameters
    :param resume: dictionary of parameters if continuing training from checkpoint
    :return: Trained model and tested
    """

    netG = Generator(start_size=args["img_size"][0], out_size=args["inpaint_size"], dropout=args["dropout"])
    netG.apply(weights_init)

    netD = Discriminator(args["inpaint_size"], dropout=args["dropout"])
    netD.apply(weights_init)

    criterion = torch.nn.BCELoss()
    criterionMSE = torch.nn.MSELoss()

    l2wt = args['l2wt']
    overlapSize = args["overlap"]

    input_real = torch.FloatTensor(args['Batch_Size'], 1, 256, 256)
    input_cropped = torch.FloatTensor(args['Batch_Size'], 1, 256, 256)
    label = torch.FloatTensor(args['Batch_Size'])
    real_label = 1
    fake_label = 0

    real_center = torch.FloatTensor(args['Batch_Size'], 1, 64, 64)
    netD.to(device)
    netG.to(device)
    criterion.to(device)
    criterionMSE.to(device)
    input_real, input_cropped, label = input_real.to(device), input_cropped.to(device), label.to(device)
    real_center = real_center.to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=args['lr'], betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args['lr'], betas=(0.5, 0.999))

    in_shape = args["img_size"][0]
    start_epoch = 0
    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)

    start_time = time.time()
    disc_losses = []
    gen_losses = []
    l2_losses = []
    iters = range(20)

    # dataset loop
    for epoch in tqdm_epoch:
        disc_mean_loss = []
        gen_mean_loss = []
        l2_mean_loss = []
        # "Disc Loss":errD.data,"Gen Loss":errG_D.data,"l2 Loss":errG_l2.data.item()
        for i in iters:
            data = next(training_dataset_loader)

            x_cpu = data["image"]
            x = x_cpu.to(device)

            # B,C,W,H

            # B,C,W,H
            if args['type'] == 'sliding':
                center_offset_y = np.random.choice(np.arange(0, 97, args['inpaint_size']))
                center_offset_x = np.random.choice(np.arange(0, 97, args['inpaint_size']))
                x_start, x_end = overlapSize + center_offset_x, center_offset_x - overlapSize + args['inpaint_size']
                y_start, y_end = overlapSize + center_offset_y, center_offset_y - overlapSize + args['inpaint_size']
            else:
                x_start, x_end = args['img_size'][0] // 4 + overlapSize, args['inpaint_size'] + args['img_size'][
                    0] // 4 - overlapSize
                y_start, y_end = args['img_size'][0] // 4 + overlapSize, args['inpaint_size'] + args['img_size'][
                    0] // 4 - overlapSize

            real_crop_cpu = x_cpu[:, :, x_start - overlapSize:x_end + overlapSize,
                            y_start - overlapSize:y_end + overlapSize]

            with torch.no_grad():
                input_real.resize_(x_cpu.size()).copy_(x_cpu)
                input_cropped.resize_(x_cpu.size()).copy_(x_cpu)
                real_center.resize_(real_crop_cpu.size()).copy_(real_crop_cpu)

                input_cropped[:, 0,
                x_start:x_end, y_start:y_end] = 0

            # start the discriminator by training with real data---
            netD.zero_grad()
            with torch.no_grad():
                label.resize_(args['Batch_Size']).fill_(real_label)

            output = netD(real_center)
            errD_real = criterion(output.reshape(args['Batch_Size']), label)
            errD_real.backward()
            D_x = output.data.mean()

            # train the discriminator with fake data---
            fake = netG(input_cropped)

            label.data.fill_(fake_label)
            output = netD(fake.detach())

            errD_fake = criterion(output.reshape(args['Batch_Size']), label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            # train the generator now---
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG_D = criterion(output.reshape(args['Batch_Size']), label)

            l2wtMatrix = real_center.clone()
            l2wtMatrix.data.fill_(l2wt * 10)
            l2wtMatrix.data[:, :, overlapSize:args['inpaint_size'] - overlapSize,
            overlapSize:args['inpaint_size'] - overlapSize] = l2wt

            errG_l2 = (fake - real_center).pow(2)
            errG_l2 = errG_l2 * l2wtMatrix
            errG_l2 = errG_l2.mean()

            with torch.no_grad():
                recon_image = input_cropped.clone()
                recon_image.data[:, :, x_start - overlapSize:x_end + overlapSize,
                y_start - overlapSize:y_end + overlapSize] = fake.data

            # plt.imshow(gridify_output(torch.cat((input_cropped[:8], recon_image[:8])), 4))
            # plt.show()

            errG = (1 - l2wt) * errG_D + l2wt * errG_l2

            errG.backward()

            D_G_z2 = output.data.mean()
            optimizerG.step()

            l2_mean_loss.append(errG_l2.detach().cpu().numpy())
            disc_mean_loss.append(errD.detach().cpu().numpy())
            gen_mean_loss.append(errG_D.detach().cpu().numpy())
            if i == 0 and epoch % 50 == 0:
                training_outputs(
                        x, recon_image, epoch, netG=netG, args=args, row_size=4,
                        filename=f"training-epoch={epoch}.png"
                        )

        disc_losses.append(np.mean(disc_mean_loss))
        gen_losses.append(np.mean(gen_mean_loss))
        l2_losses.append(np.mean(l2_mean_loss))
        if epoch % 200 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)

            print(
                    f"epoch: {epoch}, imgs trained: {(i + 1) * args['Batch_Size'] + epoch * 100},"
                    f" last 20 epoch mean losses - l2={np.mean(l2_losses[-20:]):.4f}, disc="
                    f"{np.mean(disc_losses[-20:])}, "
                    f"gen={np.mean(gen_losses[-20:])}"
                    f"time per epoch {time_per_epoch:.2f}s, time elapsed {int(time_taken / 3600)}:"
                    f"{((time_taken / 3600) % 1) * 60:02.0f}, est time remaining: {hours}:{mins:02.0f}\r"
                    )

        if epoch % 1000 == 0 and epoch >= 0:
            save(args=args, final=False, netD=netD, netG=netG, optimD=optimizerD, optimG=optimizerG, epoch=epoch)

    save(args=args, final=True, netD=netD, netG=netG, optimD=optimizerD, optimG=optimizerG)

    testing(testing_dataset_loader, args, netG)


def save(final, netG, netD, optimG, optimD, args, loss=0, epoch=0):
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
                    'n_epoch':                        args["EPOCHS"],
                    'generator_state_dict':           netG.state_dict(),
                    'generator_optim_state_dict':     optimG.state_dict(),
                    'discriminator_state_dict':       netD.state_dict(),
                    'discriminator_optim_state_dict': optimD.state_dict(),
                    "args":                           args
                    # 'loss': LOSS,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
                )
    else:
        torch.save(
                {
                    'n_epoch':                        epoch,
                    'generator_state_dict':           netG.state_dict(),
                    'generator_optim_state_dict':     optimG.state_dict(),
                    'discriminator_state_dict':       netD.state_dict(),
                    'discriminator_optim_state_dict': optimD.state_dict(),
                    "args":                           args,
                    'loss':                           loss,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/checkpoint/diff_epoch={epoch}.pt'
                )


def training_outputs(x, x_hat, epoch, row_size, netG, args, filename):
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
    netG.eval()

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
    netG.train()


def testing(testing_dataset_loader, args, netG, in_shape=256, overlapSize=4):
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
    netG.eval()

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
    input_cropped = torch.FloatTensor(args['Batch_Size'], 1, 256, 256).to(device)
    psnr_whole = []
    psnr = []
    SSIM = []
    for epoch in range(test_iters // args["Batch_Size"] + 5):
        data = next(testing_dataset_loader)
        x_cpu = data["image"]
        x = x_cpu.to(device)

        # B,C,W,H

        if args['type'] == 'sliding':
            center_offset_y = np.random.choice(np.arange(0, 97, args['inpaint_size']))
            center_offset_x = np.random.choice(np.arange(0, 97, args['inpaint_size']))
            x_start, x_end = overlapSize + center_offset_x, center_offset_x - overlapSize + args['inpaint_size']
            y_start, y_end = overlapSize + center_offset_y, center_offset_y - overlapSize + args['inpaint_size']
        else:
            x_start, x_end = args['img_size'][0] // 4 + overlapSize, args['inpaint_size'] + args['img_size'][
                0] // 4 - overlapSize
            y_start, y_end = args['img_size'][0] // 4 + overlapSize, args['inpaint_size'] + args['img_size'][
                0] // 4 - overlapSize

        with torch.no_grad():
            input_cropped.resize_(x_cpu.size()).copy_(x_cpu)
            input_cropped[:, 0, x_start:x_end, y_start:y_end] = 0

        fake = netG(input_cropped)

        recon_image = input_cropped.clone()
        recon_image.data[:, :, x_start - overlapSize:x_end + overlapSize,
        y_start - overlapSize:y_end + overlapSize] = fake.data

        real_center = x_cpu[:, :, x_start - overlapSize:x_end + overlapSize,
                      y_start - overlapSize:y_end + overlapSize].to(device)

        training_outputs(
                x, recon_image, epoch * 50, netG=netG, args=args, row_size=4, filename=f"testing-epoch={epoch}.png"
                )

        psnr_whole.append(evaluation.PSNR(recon_image, x))
        psnr.append(
                evaluation.PSNR(
                        fake, real_center
                        )
                )

        SSIM.append(
                [evaluation.SSIM(
                        real_center[i].reshape(args['inpaint_size'], args['inpaint_size']),
                        fake[i].reshape(args['inpaint_size'], args['inpaint_size'])
                        ) for i in range(args["Batch_Size"])]
                )
    print(f"Test set PSNR: {np.mean(psnr)} +- {np.std(psnr)}")
    print(f"Test set SSIM: {np.mean(SSIM)} +- {np.std(SSIM)}")


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
        "img_size":     [128, 128],
        "inpaint_size": 32,
        "overlap":      6,
        "EPOCHS":       5000,
        "Batch_Size":   16,
        "dataset":      "mri",
        "dropout":      0.2,
        "lr":           0.0002,
        "l2wt":         0.999,
        "weight_decay": 0.0,
        "type":         'centered',
        'random_slice': True,
        }
    if str(sys.argv[1]) == "101":
        args['inpaint_size'] = 64
        args['arg_num'] = 101
        args["type"] = 'sliding'
    elif str(sys.argv[1]) == "102":
        args['inpaint_size'] = 32
        args['arg_num'] = 102
        args["type"] = 'sliding'

    elif str(sys.argv[1]) == "103":
        args['inpaint_size'] = 64
        args['arg_num'] = 103
    elif str(sys.argv[1]) == "104":
        args['inpaint_size'] = 32
        args['arg_num'] = 104

    print(f"inpaint size: {args['inpaint_size']}, arg_num {args['arg_num']}")
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
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)

    main()


# model

# train

# test

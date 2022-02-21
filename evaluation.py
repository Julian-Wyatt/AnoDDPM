import matplotlib.pyplot as plt
import torch

from helpers import gridify_output


def main():
    pass


def heatmap(real: torch.Tensor, recon: torch.Tensor, mask, filename):
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = mse > 0
    mse_threshold = (mse_threshold.float() * 2) - 1
    output = torch.cat((real, recon, mse, mse_threshold, mask))
    plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
    plt.axis('off')
    plt.savefig(filename)
    plt.clf()

    dice = dice_coeff(real, recon, mse=mse, real_mask=mask)
    return dice


# for anomalous dataset - metric of crossover
def dice_coeff(real: torch.Tensor, recon: torch.Tensor, real_mask: torch.Tensor, mse=None, smooth=1):
    if mse == None:
        mse = ((real - recon).square() * 2) - 1
    intersection = torch.sum(mse * real_mask, dim=[1, 2, 3])
    union = torch.sum(mse, dim=[1, 2, 3]) + torch.sum(real_mask, dim=[1, 2, 3])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    return dice


def PSNR(recon, real):
    se = (real - recon).square()
    mse = torch.mean(se, dim=list(range(len(real.shape))))
    psnr = 20 * torch.log10(torch.max(real) / torch.sqrt(mse))
    return psnr.detach().cpu().numpy()


def FID():
    pass


def sFID():
    pass


def testing(testing_dataset_loader, diffusion, args, ema, model):
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

    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/')
    except OSError:
        pass
    ema.eval()
    model.eval()

    plt.rcParams['figure.dpi'] = 200
    for i in [*range(100, args['sample_distance'], 100)]:
        data = next(testing_dataset_loader)
        if args["dataset"] != "cifar":
            x = data["image"]
            x = x.to(device)
        else:
            # cifar outputs [data,class]
            x = data[0].to(device)

        row_size = min(5, args['Batch_Size'])

        fig, ax = plt.subplots()
        out = diffusion.forward_backward(ema, x, see_whole_sequence="half", t_distance=i)
        imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
        ani = animation.ArtistAnimation(
                fig, imgs, interval=200, blit=True,
                repeat_delay=1000
                )

        files = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/')
        ani.save(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/t={i}-attempts={len(files) + 1}.mp4')

    test_iters = 40

    vlb = []
    for epoch in range(test_iters // args["Batch_Size"] + 5):
        data = next(testing_dataset_loader)
        if args["dataset"] != "cifar":
            x = data["image"]
            x = x.to(device)
        else:
            # cifar outputs [data,class]
            x = data[0].to(device)

        vlb_terms = diffusion.calc_total_vlb(x, model, args)
        vlb.append(vlb_terms)

    psnr = []
    for epoch in range(test_iters // args["Batch_Size"] + 5):
        data = next(testing_dataset_loader)
        if args["dataset"] != "cifar":
            x = data["image"]
            x = x.to(device)
        else:
            # cifar outputs [data,class]
            x = data[0].to(device)

        out = diffusion.forward_backward(ema, x, see_whole_sequence=None, t_distance=args["T"] // 2)
        psnr.append(PSNR(out, x))

    print(
            f"Test set total VLB: {np.mean([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +- {np.std([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])}"
            )
    print(
            f"Test set prior VLB: {np.mean([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +-"
            f" {np.std([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])}"
            )
    print(
            f"Test set vb @ t=200: {np.mean([i['vb'][0][199].cpu().item() for i in vlb])} "
            f"+- {np.std([i['vb'][0][199].cpu().item() for i in vlb])}"
            )
    print(
            f"Test set x_0_mse @ t=200: {np.mean([i['x_0_mse'][0][199].cpu().item() for i in vlb])} "
            f"+- {np.std([i['x_0_mse'][0][199].cpu().item() for i in vlb])}"
            )
    print(
            f"Test set mse @ t=200: {np.mean([i['mse'][0][199].cpu().item() for i in vlb])}"
            f" +- {np.std([i['mse'][0][199].cpu().item() for i in vlb])}"
            )
    print(f"Test set PSNR: {np.mean(psnr)} +- {np.std(psnr)}")


def main():
    args, output = load_parameters()
    print(f"args{args['arg_num']}")

    in_channels = 3 if args["dataset"].lower() == "cifar" else 1
    unet = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )
    ema = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"]
            )

    ema.load_state_dict(output["ema"])
    ema.to(device)
    ema.eval()

    unet.load_state_dict(output["model_state_dict"])
    unet.to(device)
    unet.eval()
    training_dataset, testing_dataset = dataset.init_datasets("./", args)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    testing(testing_dataset_loader, diff, args, ema, unet)



if __name__ == '__main__':
    import dataset
    import os
    import matplotlib.animation as animation
    import numpy as np
    from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
    from UNet import UNetModel
    from detection import load_parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()

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
    if not mse:
        mse = ((recon - real).square() * 2) - 1
    intersection = torch.sum(mse * real_mask, dim=[1, 2, 3])
    union = torch.sum(mse, dim=[1, 2, 3]) + torch.sum(real_mask, dim=[1, 2, 3])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    return dice


def PSNR(recon, real):
    mse = torch.mean((recon - real).square(), dim=0)
    psnr = 10 * torch.log10(mse.pow(-1))
    return psnr


def FID():
    pass


def sFID():
    pass



if __name__ == '__main__':
    main()

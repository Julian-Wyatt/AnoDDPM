import torch


def main():
    pass




def dice_coeff(real: torch.Tensor, recon: torch.Tensor, real_mask: torch.Tensor, smooth=1):
    mse = (recon - real).square()
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

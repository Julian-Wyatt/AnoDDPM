import json
import torch
from diffusion import testing, init_datasets, init_dataset_loader
from models import UNet, GaussianDiffusion, get_beta_schedule

# multiple samples from varying encoding distances


if __name__=='__main__':

    ROOT_DIR='./'
    file = f'{ROOT_DIR}model/diff-params-29-11-750epochs/'

    with open(f'{file}args.json','r') as f:
        args = json.load(f)
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = UNet(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusion(unet.to(device), args['img_size'], betas, loss_weight=args['loss_weight'],loss_type=args[
        'loss-type'])

    diffusion.load_state_dict(torch.load(f'{file}params',map_location=device))

    training_dataset,testing_dataset = init_datasets(args)
    testing_dataset_loader = init_dataset_loader(testing_dataset, args)

    testing(testing_dataset,diffusion, args, device, use_ddim=False)

from models import *
import dataset

import json
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision



def testing(testing_dataset,diffusion, device=torch.device('cpu'), root_dir='./',use_ddim = False, args={'T':500,'sample_distance':10}):
    testing_dataset_loader = dataset.cycle(torch.utils.data.DataLoader(testing_dataset,
                                                               batch_size=args['Batch_Size'], shuffle=True,
                                                               num_workers=2))
    plt.rcParams['figure.dpi'] = 200
    for i in [args['sample_distance'], args['T'] / 4, None]:
        data = next(testing_dataset_loader)
        x = data["image"]
        x = x.to(device)
        rowSize = min(5,args['Batch_Size'])
        fig, ax = plt.subplots()
        out = diffusion.forward_backward(x, see_whole_sequence=True, use_ddim=use_ddim, t_distance=i)
        imgs = [[ax.imshow(
            torchvision.utils.make_grid(((x+1)*127.5).clamp(0,255).to(torch.uint8), nrow=rowSize).cpu().data.permute(0, 2, 1).contiguous().permute(2,
                                                                                                                     1,
                                                                                                                     0),
            animated=True)] for x in out]
        # out = out.permute(0,2,3,1)
        ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,
                                        repeat_delay=1000)

        ani.save(
            f'{root_dir}diffusion-videos/test-set-{time.gmtime().tm_mday}-{time.gmtime().tm_mon}-{i}-ddim={use_ddim}-{args["EPOCHS"]}epochs.mp4')
        plt.title(
            f'x,forward_backward with video: {root_dir}diffusion-videos/test-set-{time.gmtime().tm_mday}-{time.gmtime().tm_mon}-{i}-{args["EPOCHS"]}epochs.mp4')
        print("saved")
        plt.grid(False)
        plt.imshow(
            torchvision.utils.make_grid(((out[-1]+1)*127.5).clamp(0,255).to(torch.uint8), nrow=rowSize).cpu().data.permute(0, 2,
                                                                                             1).contiguous().permute(
                2, 1, 0),
            cmap='gray')

        plt.show()
        plt.pause(0.0001)


if __name__=='__main__':

    file = './model/diff-params-29-11-750epochs/'

    with open(f'{file}args.json','r') as f:
        args = json.load(f)
    print(args)
    img_size = (32,32)
    root_dir='./'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = UNet(args['base_channels'], channel_mults=args['channel_mults'])
    # unet.apply(weights_init)

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusion(unet.to(device), img_size, betas, loss_weight=args['loss_weight'],loss_type=args['loss-type'])
    # diffusion.apply(weights_init)

    diffusion.load_state_dict(torch.load(f'{file}params',map_location=device))
    diffusion.eta = 0.4

    testing_dataset = dataset.MRIDataset(root_dir=f'{root_dir}Test/', img_size=img_size)
    args['Batch_Size']=5
    testing(testing_dataset,diffusion, device,args=args,use_ddim=True)
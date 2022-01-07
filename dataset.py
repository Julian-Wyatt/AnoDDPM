import os
from random import randint
import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

############ Dataset animation creation ###########

# fig = plt.figure()
# ims = []
# for i in range(256):
#     tempImg = image.slicer[:,i:i+1,:]
#     im = plt.imshow(tempImg.dataobj.reshape(256,192),cmap='gray', animated=True)
#     ims.append([im])
#
# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)
#
# ani.save('dynamic_images.mp4')

# plt.show()

# for file in files:
#     if file[-4:]==".nii":
#
#         # use slice 35-55
#         img = nib.load(image_path+"/"+file)
#
#         image = img.get_fdata()
#         image = np.rot90(image)
#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# image = np.load("./Train/A00028185/A00028185.npy")
# fig, ax = plt.subplots()
#
# ims = []
# print(image.shape)
# def update(i):
#     print(i)
#     tempImg = image[:, i:i + 1,: ]
#     ax.set_title(f"slice: {i}")
#     ax.imshow(tempImg.reshape(image.shape[0],image.shape[2]),cmap='gray', animated=True)
#
#
#
# ani = animation.FuncAnimation(fig, update,frames=image.shape[1], interval=10, blit=False,
#                                 repeat=False)
#
# ani.save(f'A00028185-dim1.mp4')
# plt.close(fig)

class MRIDataset(Dataset):
    """Healthy MRI dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32,32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.CenterCrop(240),
                                transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                                # transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))
                                ]) if not transform else transform

        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if os.path.exists(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy")):
            image = np.load(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"))
            pass
        else:
            img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], f"sub-{self.filenames[idx]}_ses-NFB3_T1w.nii.gz")
            # random between 40 and 130
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()

            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"),image.astype(
                np.float32))
        if self.random_slice:
            slice_idx = randint(40, 100)
        else:
            slice_idx = 80
        image = image[:, slice_idx:slice_idx + 1, :].reshape(256, 192).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, "filenames": self.filenames[idx]}
        return sample


class AnomalousMRIDataset(Dataset):
    """Anomalous MRI dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32,32), slice_selection="random"):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            img_size: size of each 2D dataset image
            slice_selection: "random" = randomly selects a slice from the image
                             "iterateKnown" = iterates between ranges of tumour using slice data
                             "iterateUnKnown" = iterates through whole MRI volume
        """
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.CenterCrop(240),
                                transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                                # transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))
                                ]) if not transform else transform

        self.slices = {"17904": range(15, 21), "18428": range(42, 61), "18582": range(30, 63), "18638": range(36, 60),
                       "18675": range(37, 61), "18716": range(29, 56), "18756": range(39, 60), "18863": range(33, 61),
                       "18886": range(14, 30), "18975": range(55, 61), "19015": range(43, 61), "19085": range(41, 61),
                       "19275": range(23, 31), "19277": range(33, 61), "19357": range(31, 58), "19398": range(27, 51),
                       "19423": range(31, 61), "19567": range(20, 31), "19628": range(32, 63), "19691": range(41, 63),
                       "19723": range(12, 24), "19849": range(23, 46)}

        self.filenames = self.slices.keys()
        self.filenames = list(map(lambda name: f"{ROOT_DIR}/{name}.nii",self.filenames))
        # self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.slice_selection = slice_selection

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if os.path.exists(os.path.join(f"{self.filenames[idx]}.npy")):
            image = np.load(os.path.join( f"{self.filenames[idx]}.npy"))
            pass
        else:
            img_name = os.path.join(self.filenames[idx])
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()
            image = np.rot90(image)

            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(os.path.join(f"{self.filenames[idx]}.npy"),image.astype(
                np.float32))

        if self.slice_selection == "random":
            temp_range = self.slices[self.filenames[idx][-9:-4]]
            slice_idx = randint(temp_range.start,temp_range.stop)
            image = image[:, :,slice_idx:slice_idx + 1].reshape(image.shape[0],image.shape[1]).astype(np.float32)
            if self.transform:
                image = [self.transform(image)]
        elif self.slice_selection == "iterateKnown":
            temp_range = self.slices[self.filenames[idx][-9:-4]]
            output = []
            for i in temp_range:
                temp = image[:,: i:i + 1].reshape(image.shape[0],image.shape[1]).astype(np.float32)
                if self.transform:
                    temp = self.transform(temp)
                output.append(temp)
            image = output

        elif self.slice_selection == "iterateUnKnown":
            output = []
            for i in range(image.shape[2]):
                temp = image[:, : i:i + 1].reshape(image.shape[0], image.shape[1]).astype(np.float32)
                if self.transform:
                    temp = self.transform(temp)
                output.append(temp)
            image = output

        sample = {'image': image, "filenames": self.filenames[idx]}
        return sample

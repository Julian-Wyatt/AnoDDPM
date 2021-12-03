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

class MRIDataset(Dataset):
    """MRI dataset."""

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
            slice_idx = randint(40, 130)
        else:
            slice_idx = 80
        image = image[:, slice_idx:slice_idx + 1, :].reshape(256, 192).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, "filenames": self.filenames[idx]}
        return sample

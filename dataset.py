import os
from random import randint

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import Dataset
from torchvision import transforms

# helper function to make getting another batch of data easier
import helpers


# from diffusion_training import output_img


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


############ Dataset animation creation ###########


# image = np.load("./Train/A00028185/A00028185.npy")
# print(image.shape)
#
# image = np.load("./Cancerous Dataset/EdinburghDataset/Anomalous/raw/17904.nii.npy")
# print(image.shape)
#
# image = nib.load(
#         "./Cancerous Dataset/EdinburghDataset/19423/tissue_classes"
#         "/y_anon_128401136192244359611861638207501334216725958.nii"
#         ).get_fdata()
# image = np.rot90(image)
#
# fig, ax = plt.subplots()
#
# ims = []
# print(image.shape)

# plt.imsave(
#         "./CancerousDataset/EdinburghDataset/19423/tissue_classes"
#         "/y_anon_128401136192244359611861638207501334216725958.png", image[100, ...]
#         )


# def update(i):
#     print(i)
#     tempImg = image[:, i:i + 1, :]
#     ax.set_title(f"slice: {i}")
#     ax.imshow(tempImg.reshape(image.shape[0], image.shape[2]), cmap='gray', animated=True)

def main(save_videos=False):
    DATASET = "./CancerousDataset/EdinburghDataset"
    patients = os.listdir(DATASET)

    for patient in patients:
        try:
            patient_data = os.listdir(f"{DATASET}/{patient}")
        except:
            print(f"{DATASET}/{patient} Not a directory")
            continue
        for data_folder in patient_data:
            if "COR_3D" in data_folder:
                try:
                    T1_files = os.listdir(f"{DATASET}/{patient}/{data_folder}")
                except:
                    print(f"{patient}/{data_folder} not a directory")
                    continue
                for t1 in T1_files:
                    if t1[-4:] == ".nii":
                        # try:
                        # use slice 35-55
                        img = nib.load(f"{DATASET}/{patient}/{data_folder}/{t1}")
                        image = img.get_fdata()
                        image = np.rot90(image, 3, (0, 2))
                        image_mean = np.mean(image)
                        image_std = np.std(image)
                        img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
                        image = np.clip(image, img_range[0], img_range[1])
                        image = image / (img_range[1] - img_range[0])
                        image = np.transpose(image, (1, 0, 2))

                        np.save(
                                f"{DATASET}/Anomalous-T1/raw/{patient}.npy", image.astype(
                                        np.float32
                                        )
                                )
                        print(f"Saved {DATASET}/Anomalous-T1/raw/{patient}.npy")
                        if save_videos:

                            fig, ax = plt.subplots()

                            def update(i):
                                # print(i)
                                tempImg = image[i, :, :].reshape(image.shape[0], image.shape[2])
                                ax.set_title(f"slice: {i}")
                                ax.imshow(tempImg, cmap='gray', animated=True)


                            ani = animation.FuncAnimation(
                                    fig, update, frames=image.shape[1], interval=50, blit=False,
                                    repeat=False
                                    )

                            ani.save(f"{DATASET}/Anomalous-T1/video/{patient}.mp4")

                            print(f"Saved {DATASET}/Anomalous-T1/raw/{patient}.mp4")

                        outputImg = np.zeros((256, 256, 310))
                        for i in range(image.shape[0]):
                            tempImg = image[i:i + 1, :, :].reshape(image.shape[1], image.shape[2])
                            img_sm = cv2.resize(tempImg, (310, 256), interpolation=cv2.INTER_CUBIC)
                            outputImg[i, :, :] = img_sm

                        image = outputImg
                        print(f"Resized:  {DATASET}/Anomalous-T1/raw/{patient}")

                        if save_videos:
                            fig = plt.figure()
                            ims = []
                            for i in range(image.shape[0]):
                                tempImg = image[i:i + 1, :, :]
                                im = plt.imshow(
                                        tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True
                                        )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                    fig, ims, interval=50, blit=True,
                                    repeat_delay=1000
                                    )

                            ani.save(f"{DATASET}/Anomalous-T1/video/{patient}-resized.mp4")
                            plt.close(fig)

                        np.save(
                                f"{DATASET}/Anomalous-T1/raw/{patient}-resized.npy", image.astype(
                                        np.float32
                                        )
                                )

                        print(f"Saved resized  {DATASET}/Anomalous-T1/raw/{patient}")


def checkDataSet():
    resized = False
    mri_dataset = AnomalousMRIDataset(
            "CancerousDataset/EdinburghDataset/Anomalous-T1/raw", img_size=(256, 256),
            slice_selection="iterateUnknown", resized=resized
            # slice_selection="random"
            )

    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    mri_dataset,
                    batch_size=22, shuffle=True,
                    num_workers=2, drop_last=True
                    )
            )

    new = next(dataset_loader)

    image = new["image"]

    print(image.shape)
    from helpers import gridify_output
    print("Making Video for resized =", resized)
    fig = plt.figure()
    ims = []
    for i in range(0, image.shape[1], 2):
        tempImg = image[:, i, ...].reshape(image.shape[0], 1, image.shape[2], image.shape[3])
        im = plt.imshow(
                gridify_output(tempImg, 5), cmap='gray',
                animated=True
                )
        ims.append([im])

    ani = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True,
            repeat_delay=1000
            )

    ani.save(f"./CancerousDataset/EdinburghDataset/Anomalous-T1/video-resized={resized}.mp4")


def output_videos_for_dataset():
    folders = os.listdir("/Users/jules/Downloads/19085/")
    folders.sort()
    print(f"Folders: {folders}")
    for folder in folders:
        try:
            files_folder = os.listdir("/Users/jules/Downloads/19085/" + folder)
        except:
            print(f"{folder} not a folder")
            exit()

        for file in files_folder:
            try:
                if file[-4:] == ".nii":
                    # try:
                    # use slice 35-55
                    img = nib.load("/Users/jules/Downloads/19085/" + folder + "/" + file)
                    image = img.get_fdata()
                    image = np.rot90(image, 3, (0, 2))
                    print(f"{folder}/{file} has shape {image.shape}")
                    outputImg = np.zeros((256, 256, 310))
                    for i in range(image.shape[1]):
                        tempImg = image[:, i:i + 1, :].reshape(image.shape[0], image.shape[2])
                        img_sm = cv2.resize(tempImg, (310, 256), interpolation=cv2.INTER_CUBIC)
                        outputImg[i, :, :] = img_sm

                    image = outputImg
                    print(f"{folder}/{file} has shape {image.shape}")
                    fig = plt.figure()
                    ims = []
                    for i in range(image.shape[0]):
                        tempImg = image[i:i + 1, :, :]
                        im = plt.imshow(tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True)
                        ims.append([im])

                    ani = animation.ArtistAnimation(
                            fig, ims, interval=50, blit=True,
                            repeat_delay=1000
                            )

                    ani.save("/Users/jules/Downloads/19085/" + folder + "/" + file + ".mp4")
                    plt.close(fig)

            except:
                print(
                        f"--------------------------------------{folder}/{file} FAILED TO SAVE VIDEO ------------------------------------------------"
                        )


def get_segmented_labels(save_videos=True):
    DATASET = "./CancerousDataset/EdinburghDataset"
    patients = os.listdir(DATASET)

    for patient in patients:
        try:
            patient_data = os.listdir(f"{DATASET}/{patient}")
        except:
            print(f"{DATASET}/{patient} Not a directory")
            continue
        for data_folder in patient_data:
            if "tissue_classes" in data_folder:
                try:
                    masks = os.listdir(f"{DATASET}/{patient}/{data_folder}")
                except:
                    print(f"{patient}/{data_folder} not a directory")
                    continue

                for mask in masks:

                    if "cleaned" in mask and mask[-4:] == ".nii":
                        # try:
                        # use slice 35-55
                        img = nib.load(f"{DATASET}/{patient}/{data_folder}/{mask}")
                        image = img.get_fdata()
                        image = np.rot90(image, 3, (0, 2))
                        image_mean = np.mean(image)
                        image_std = np.std(image)
                        img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
                        image = np.clip(image, img_range[0], img_range[1])
                        image = image / (img_range[1] - img_range[0])
                        image = np.transpose(image, (1, 0, 2))

                        np.save(
                                f"{DATASET}/Anomalous-T1/mask/{patient}.npy", image.astype(
                                        np.float32
                                        )
                                )
                        print(image.shape)
                        print(f"Saved {DATASET}/Anomalous-T1/raw/{patient}.npy")
                        if save_videos:
                            fig = plt.figure()
                            ims = []
                            for i in range(image.shape[0]):
                                tempImg = image[i:i + 1, :, :]
                                im = plt.imshow(
                                        tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True
                                        )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                    fig, ims, interval=50, blit=True,
                                    repeat_delay=1000
                                    )

                            ani.save(f"{DATASET}/Anomalous-T1/mask/videos/{patient}.mp4")
                            plt.close(fig)

                        outputImg = np.zeros((256, 256, 310))
                        for i in range(image.shape[0]):
                            tempImg = image[i:i + 1, :, :].reshape(image.shape[1], image.shape[2])
                            img_sm = cv2.resize(tempImg, (310, 256), interpolation=cv2.INTER_CUBIC)
                            outputImg[i, :, :] = img_sm

                        image = outputImg
                        print(f"Resized:  {DATASET}/Anomalous-T1/raw/{patient}")

                        if save_videos:
                            fig = plt.figure()
                            ims = []
                            for i in range(image.shape[0]):
                                tempImg = image[i:i + 1, :, :]
                                im = plt.imshow(
                                        tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True
                                        )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                    fig, ims, interval=50, blit=True,
                                    repeat_delay=1000
                                    )

                            ani.save(f"{DATASET}/Anomalous-T1/mask/videos/{patient}-resized.mp4")
                            plt.close(fig)

                        np.save(
                                f"{DATASET}/Anomalous-T1/mask/{patient}-resized.npy", image.astype(
                                        np.float32
                                        )
                                )

                        print(f"Saved resized  {DATASET}/Anomalous-T1/raw/{patient}")


def load_datasets_for_test():
    args = {'img_size': (256, 256), 'random_slice': True, 'Batch_Size': 20}
    training, testing = init_datasets("./", args)

    ano_dataset = AnomalousMRIDataset(
            ROOT_DIR=f'./CancerousDataset/EdinburghDataset/Anomalous-T1', img_size=args['img_size'],
            slice_selection="random", resized=False
            )

    train_loader = init_dataset_loader(training, args)
    ano_loader = init_dataset_loader(ano_dataset, args)

    for i in range(5):
        new = next(train_loader)
        new_ano = next(ano_loader)
        output = torch.cat((new["image"][:10], new_ano["image"][:10]))
        plt.imshow(helpers.gridify_output(output, 5), cmap='gray')
        plt.show()
        plt.pause(0.0001)


def init_datasets(ROOT_DIR, args):
    training_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}Train/', img_size=args['img_size'], random_slice=args['random_slice']
            )
    testing_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}Test/', img_size=args['img_size'], random_slice=args['random_slice']
            )
    # testing_dataset = MRIDataset(ROOT_DIR='/content/drive/MyDrive/dissertation data/Anomalous/',transform=transform)
    return training_dataset, testing_dataset


def init_dataset_loader(mri_dataset, args):
    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    mri_dataset,
                    batch_size=args['Batch_Size'], shuffle=True,
                    num_workers=0, drop_last=True
                    )
            )

    return dataset_loader


class MRIDataset(Dataset):
    """Healthy MRI dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomAffine(3, translate=(0.02, 0.09)),
                 transforms.CenterCrop(235),
                 transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 # transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))
                 ]
                ) if not transform else transform

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
            img_name = os.path.join(
                    self.ROOT_DIR, self.filenames[idx], f"sub-{self.filenames[idx]}_ses-NFB3_T1w.nii.gz"
                    )
            # random between 40 and 130
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()

            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(
                    os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"), image.astype(
                            np.float32
                            )
                    )
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

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), slice_selection="random", resized=False):
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
        self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.CenterCrop((175, 240)),
                 # transforms.RandomAffine(0, translate=(0.02, 0.1)),
                 transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 # transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))
                 ]
                ) if not transform else transform
        self.img_size = img_size
        self.resized = resized
        self.slices = {
            "17904": range(160, 205), "18428": range(177, 213), "18582": range(160, 190), "18638": range(160, 212),
            "18675": range(140, 200), "18716": range(135, 190), "18756": range(150, 205), "18863": range(130, 190),
            "18886": range(120, 180), "18975": range(170, 194), "19015": range(158, 195), "19085": range(155, 195),
            "19275": range(184, 213), "19277": range(158, 209), "19357": range(158, 210), "19398": range(164, 200),
            "19423": range(142, 200), "19567": range(160, 200), "19628": range(147, 210), "19691": range(155, 200),
            "19723": range(140, 187), "19849": range(150, 180)
            }

        self.filenames = self.slices.keys()
        self.filenames = list(map(lambda name: f"{ROOT_DIR}/raw/{name}.npy", self.filenames))
        # self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.slice_selection = slice_selection

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if os.path.exists(os.path.join(f"{self.filenames[idx][:-4]}.npy")):
            if self.resized:
                image = np.load(os.path.join(f"{self.filenames[idx][:-4]}-resized.npy"))
            else:
                image = np.load(os.path.join(f"{self.filenames[idx][:-4]}.npy"))
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
            np.save(
                    os.path.join(f"{self.filenames[idx]}.npy"), image.astype(
                            np.float32
                            )
                    )
        sample = {}
        if self.slice_selection == "random":
            temp_range = self.slices[self.filenames[idx][-9:-4]]
            slice_idx = randint(temp_range.start, temp_range.stop)
            image = image[slice_idx:slice_idx + 1, :, :].reshape(image.shape[1], image.shape[2]).astype(np.float32)
            if self.transform:
                image = self.transform(image)
                # image = transforms.functional.rotate(image, -90)
            sample["slices"] = slice_idx
        elif self.slice_selection == "iterateKnown":
            temp_range = self.slices[self.filenames[idx][-9:-4]]
            output = torch.empty(temp_range.stop - temp_range.start, *self.img_size)
            # print(output.shape, image.shape, temp_range)
            for i in temp_range:
                temp = image[i, ...].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                if self.transform:
                    temp = self.transform(temp)
                    # temp = transforms.functional.rotate(temp, -90)
                output[i - temp_range.start, ...] = temp
            image = output
            sample["slices"] = temp_range

        elif self.slice_selection == "iterateUnknown":

            output = torch.empty(image.shape[0], *self.img_size)
            for i in range(image.shape[0]):
                temp = image[i:i + 1, :, :].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                if self.transform:
                    temp = self.transform(temp)
                    # temp = transforms.functional.rotate(temp, -90)
                output[i, ...] = temp

            image = output
            sample["slices"] = image.shape[0]

        sample["image"] = image
        sample["filenames"] = self.filenames[idx]
        # sample = {'image': image, "filenames": self.filenames[idx], "slices":slice_idx}
        return sample


def load_image_mask(file, img_size, Ano_Dataset_Class):
    transform = Ano_Dataset_Class.transform
    img_mask = np.load(f"{Ano_Dataset_Class.ROOT_DIR}/mask/{file}-resized.npy")
    output = torch.empty(img_mask.shape[0], *img_size)
    for i in range(img_mask.shape[0]):
        temp = img_mask[i:i + 1, :, :].reshape(img_mask.shape[1], img_mask.shape[2]).astype(np.float32)
        if transform:
            temp = transform(temp)
            # temp = transforms.functional.rotate(temp, -90)
        output[i, ...] = temp

    return output


if __name__ == "__main__":
    load_datasets_for_test()
    # get_segmented_labels(False)
    # main()

from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval

from align_images import get_frame_transformation_matrix


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def renorm_image(image):
    image -= np.nanmedian(image)
    image /= np.nanstd(image)
    return image


class ImageDataset(Dataset):
    def __init__(self, device, N=1000):
        super().__init__()
        self.images = []
        self.targets = []
        self.device = device
        self.raw_images = []

        self.fullimages = []
        self.fulltargets = []

        filenames = sorted(glob("../training_data/*_L_*.fit"))
        print(f"Loading {len(filenames)} images from disk")
        for filename in filenames:
            hdu = fits.open(filename)
            img_data = hdu[0].data.astype(np.float32)
            self.raw_images.append(renorm_image(img_data))


        print("Aligning consecutive images")
        for ii in range(len(filenames)-1):
            H = get_frame_transformation_matrix(self.raw_images[ii], self.raw_images[ii+1])
            
            nx, ny = np.shape(self.raw_images[ii])
            outshape = (ny,nx)
            transformed = cv.warpAffine(self.raw_images[ii+1], H, outshape)
            self.fullimages.append(self.raw_images[ii])
            self.fulltargets.append(transformed)



        print("Creating matching cutouts")
        cutout_size = 128
        for ii in tqdm(range(N)):
            ind = np.random.randint(0, len(self.raw_images)-1)

            ny, nx = np.shape(self.raw_images[0])
            cx = np.random.randint(cutout_size, nx-cutout_size)
            cy = np.random.randint(cutout_size, ny-cutout_size)
            cutout1 = Cutout2D(self.fullimages[ind], (cx, cy), (cutout_size, cutout_size)).data
            cutout2 = Cutout2D(self.fulltargets[ind], (cx, cy), (cutout_size, cutout_size)).data

            self.images.append(torch.from_numpy(cutout1).unsqueeze(0).to(device))
            self.targets.append(torch.from_numpy(cutout2).unsqueeze(0).to(device))



    def test_cutouts(self):
        Ntest = 5
        fig, axs = plt.subplots(Ntest, 3, sharex="row", sharey="row")
        scaler = ZScaleInterval()
        for ii in range(Ntest):
            ind = np.random.randint(0, len(self.raw_images)-1)

            ny, nx = np.shape(self.raw_images[0])
            cx = np.random.randint(128, nx-128)
            cy = np.random.randint(128, ny-128)
            cutout1 = Cutout2D(self.fullimages[ind], (cx, cy), (128, 128)).data
            cutout2 = Cutout2D(self.fulltargets[ind], (cx, cy), (128, 128)).data

            limits = scaler.get_limits(cutout1)

            axs[ii][0].imshow(cutout1, vmin=limits[0], vmax=limits[1])
            axs[ii][1].imshow(cutout2, vmin=limits[0], vmax=limits[1])
            limits = scaler.get_limits(cutout2-cutout1)
            axs[ii][2].imshow(cutout2-cutout1, vmin=limits[0], vmax=limits[1])
        plt.show()



    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images[idx], self.targets[idx]


if __name__ == "__main__":
    device = get_device()
    data = ImageDataset(device)

    dataloader = DataLoader(data, batch_size=8, shuffle=True)

    data_batch, labels_batch = next(iter(dataloader))
    print(f"Batch shape: {data_batch.shape}")
    print(f"Labels shape: {labels_batch.shape}")


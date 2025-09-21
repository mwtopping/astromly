from tqdm import tqdm
from glob import glob
import os
from itertools import islice
from pathlib import Path


from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip


import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval

from align_images import get_frame_transformation_matrix


def median_binner(a,bin_x,bin_y):
    m,n = np.shape(a)
    strided_reshape = np.lib.stride_tricks.as_strided(a,shape=(bin_x,bin_y,m//bin_x,n//bin_y),strides = a.itemsize*np.array([(m // bin_x) * n, (n // bin_y), n, 1]))
    return np.array([np.median(col) for row in strided_reshape for col in row]).reshape(bin_x,bin_y)


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



def load_rgb_image(fname):

    hdu = fits.open(fname)
    image_data = hdu[0].data
    header = hdu[0].header

    color_image = cv.demosaicing(image_data, cv.COLOR_BayerBG2BGR)

    h, w = image_data.shape
    r = color_image[0::2, 0::2, 0]  # Extract R from RGGB pattern
    g = (color_image[0::2, 1::2, 1] + color_image[1::2, 0::2, 1]) / 2  # Average the two G channels
    b = color_image[1::2, 1::2, 2]  # Extract B

    # Create a non-interpolated RGB image (half resolution)
    non_interpolated = np.zeros((h//2, w//2, 3), dtype=color_image.dtype)
    non_interpolated[:, :, 0] = r
    non_interpolated[:, :, 1] = g
    non_interpolated[:, :, 2] = b

        #lum = cv.cvtColor(non_interpolated, cv.COLOR_BGR2GRAY)
    lum = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    
    return lum, header





BASE_DIR = Path(__file__).parent.parent

class ImageDataset(Dataset):
    def __init__(self, device, N=1024*8):
        super().__init__()
        self.images = []
        self.targets = []
        self.device = device
        self.raw_images = []
        self.headers = []

        self.fullimages = []
        self.fulltargets = []
        self.N = N
        self.niter = 0

        print(BASE_DIR)

        #filenames = sorted(BASE_DIR.glob("training_data/*_L_*.fit"))
        #filenames = sorted(glob("/Volumes/Seagate/data/asic*"))
        filenames = [i.path for i in islice(os.scandir("/Volumes/Seagate/data/"), 10)]
        print(f"Loading {len(filenames)} images from disk")
        for filename in filenames:
            print(filename)
#            hdu = fits.open(filename)
#            nx, ny = np.shape(hdu[0].data)
#            print(nx, ny)

#            img_data = median_binner(hdu[0].data.astype(np.float32), int(nx/2), int(ny/2))
#            img_data = hdu[0].data.astype(np.float32)
            img_data, header = load_rgb_image(filename)
            img_data = img_data.astype(np.float32)
            self.headers.append(header)
            scaler = ZScaleInterval()
            limits = scaler.get_limits(img_data)
#            fig, axs = plt.subplots(1,2)
#            axs[0].imshow(img_data, vmin=limits[0], vmax=limits[1])
            print(np.shape(img_data))
            img_data = self.preprocess(img_data)
            self.raw_images.append(img_data)
            limits = scaler.get_limits(img_data)
#            axs[1].imshow(img_data, vmin=limits[0], vmax=limits[1])
#            plt.show()


        print("Aligning consecutive images")
        for ii in tqdm(range(len(filenames)-1)):
            H = get_frame_transformation_matrix(self.raw_images[ii], self.raw_images[ii+1])

            # assign new header values to next image
            self.headers[ii+1]['a00'] = H[0][0]
            self.headers[ii+1]['a01'] = H[0][1]
            self.headers[ii+1]['a10'] = H[1][0]
            self.headers[ii+1]['a11'] = H[1][1]
            self.headers[ii+1]['b00'] = H[0][2]
            self.headers[ii+1]['b10'] = H[1][2]

            # update the image
            fits.HDUList([fits.PrimaryHDU(data=self.raw_images[ii+1], header=self.headers[ii+1])
            ]).writeto(filenames[ii+1], overwrite=True)

            nx, ny = np.shape(self.raw_images[ii])
            outshape = (ny,nx)
            transformed = cv.warpAffine(self.raw_images[ii+1], H, outshape)
            self.fullimages.append(self.raw_images[ii].copy())
            self.fulltargets.append(transformed)

        exit()



    def preprocess(self, img):
        img = renorm_image(img)

        sigma_clip = SigmaClip(sigma=5.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(img, (50, 50), filter_size=(3, 3),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        img = img - bkg.background

        return img


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

    def generate_cutout(self):
        self.niter += 1
        if self.niter % 100 == 0:
            print(f"{self.niter}/{self.N}")
        cutout_size = 128
        Nhotpix = 80
        hotpixmax = 100
        ind = np.random.randint(0, len(self.raw_images)-1)

        ny, nx = np.shape(self.raw_images[0])
        cx = np.random.randint(cutout_size, nx-cutout_size)
        cy = np.random.randint(cutout_size, ny-cutout_size)
        cutout1 = Cutout2D(self.fullimages[ind], (cx, cy), (cutout_size, cutout_size), copy=True).data
        cutout2 = Cutout2D(self.fulltargets[ind], (cx, cy), (cutout_size, cutout_size), copy=True).data

#        for jj in range(Nhotpix):
#            value = hotpixmax*np.random.random()
#            x = np.random.randint(cutout_size)
#            y = np.random.randint(cutout_size)
#            cutout1[x,y] += value



        return torch.from_numpy(cutout1).unsqueeze(0).to(self.device), torch.from_numpy(cutout2).unsqueeze(0).to(self.device)



    def __len__(self):
        return self.N
#        return len(self.targets)

    def __getitem__(self, idx):
        inp_tensors = []
        tar_tensors = []

        if torch.is_tensor(idx):
            idx = idx.tolist()
            for ii in idx:
                inp, tar = self.generate_cutout()
                inp_tensors = torch.cat((inp_tensors, inp), dim=0)
                tar_tensors = torch.cat((tar_tensors, tar), dim=0)

        else:
            inp, tar = self.generate_cutout()
            inp_tensors = inp
            tar_tensors = tar

        return inp_tensors, tar_tensors 


if __name__ == "__main__":
    device = get_device()
    data = ImageDataset(device)

    data.test_cutouts()

    dataloader = DataLoader(data, batch_size=8, shuffle=True)

    data_batch, labels_batch = next(iter(dataloader))
    print(f"Batch shape: {data_batch.shape}")
    print(f"Labels shape: {labels_batch.shape}")


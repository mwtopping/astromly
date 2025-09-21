import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import cv2 as cv
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground


if __name__ == "__main__":
    #DATA_DIR = "../data/2025-06-29_05_18_50Z/"
    DATA_DIR = "../data/2025-07-18_04_57_46Z/"
    #fnames = sorted(glob(f"{DATA_DIR}/*FIT"))
    DATA_DIR = "../data/rpicam/"
    fnames = sorted(glob(f"{DATA_DIR}/*fits"))
    block_names = sorted(glob("/Users/michael/data/asicam/ASI_BLOCK*fits"))

    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    limits = None
    frameno=0

    scaler = ZScaleInterval()

    for block in tqdm(block_names):
        hdul = fits.open(block)

        for hdu1 in hdul[1:]: 
#    scaler = ZScaleInterval(contrast=0.15)
#
#    hdu = fits.open(fnames[0])
#    img_data = hdu[1].data # index 1 for rpicam, 0 for zwo
#    color_image = cv.demosaicing(img_data, cv.COLOR_BayerBG2BGR)
#    lum = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
#    fig, axs = plt.subplots(1, 3)
#
#    bkg = Background2D(lum, (50, 50), filter_size=(3, 3),
#                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
#
    #lum3 = lum.astype(np.float64) - bkg.background
    #limits = scaler.get_limits(lum3)
    #axs[2].imshow(lum3, vmin=-5000, vmax=20000, cmap="Greys_r", aspect='auto')
    #print(limits)
#   # axs[2].imshow(lum3, cmap="Greys_r", aspect='auto')
    #plt.show()
#    cv.imshow("first", color_image)
#    cv.waitKey(0)
#    height, width, channels = color_image.shape
#    print(height, width)
#    lum /= np.nanmedian(lum)
#    fourcc = cv.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#    out = cv.VideoWriter("temp.mp4", fourcc, 30.0, (width, height))
            #fig, ax = plt.subplots(frameon=False)
            fig = plt.figure(frameon=False)
            ax = fig.add_axes([0., 0., 1., 1.])
    #        fname1 = fnames[ii]
    #        fname2 = fnames[ii+1]
    #        hdu1 = fits.open(fname1)
            img_data1 = hdu1.data
    #        hdu2 = fits.open(fname2)
    #        img_data2 = hdu2[1].data

            color_image1 = cv.demosaicing(img_data1, cv.COLOR_BayerBG2BGR)
    #        color_image2 = cv.demosaicing(img_data2, cv.COLOR_BayerBG2BGR)
            lum = np.float64(cv.cvtColor(color_image1, cv.COLOR_BGR2GRAY))
    #        lum2 = cv.cvtColor(color_image2, cv.COLOR_BGR2GRAY)
    #        lum = 0.5*lum1 + 0.5 * lum2


            #print(np.nanmedian(lum))
            lum = np.float64(lum) / np.float64(np.nanmedian(lum))

            bkg = Background2D(lum, (50, 50), filter_size=(3, 3),
                               sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

            lum = lum.astype(np.float64) - bkg.background

            if limits is None:
                limits = scaler.get_limits(lum)
            #cv.imshow("next", color_image)
            #cv.waitKey(0)
            #out.write(color_image)
            
            ax.imshow(lum, vmin=-0.15, vmax=0.45, cmap="Greys_r", aspect='auto')
            ax.set_axis_off()
            fig.savefig(f"/Users/michael/data/asicam/out/{frameno}.png", dpi=200)
    #        plt.show()
            plt.close('all')
            frameno += 1
        #out.release()
        #cv.destroyAllWindows()

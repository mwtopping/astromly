from astropy.io import fits
from glob import glob
import numpy as np
import sys





hdul = fits.HDUList([fits.PrimaryHDU()])
filenames = glob("/Users/michael/data/asicam/asicam_*fits")
print(filenames)

Nblock = 0
for filename in sorted(filenames):
    hdu = fits.open(filename)
    newhdu = fits.ImageHDU(data=hdu[0].data, header=hdu[0].header)
    newhdu.header["filename"] = filename
    hdul.append(newhdu)
    total_size = np.sum([hdu.data.nbytes for hdu in hdul[1:]])/1024/1024/1024
    print(total_size)
    if total_size > 1:
        hdul.writeto(f"/Users/michael/data/asicam/ASI_BLOCK_{Nblock}.fits", overwrite=True)
        Nblock += 1
        hdul = fits.HDUList([fits.PrimaryHDU()])
    hdu.close()

if len(hdul) > 1:
    hdul.writeto(f"/Users/michael/data/asicam/ASI_BLOCK_{Nblock}.fits", overwrite=True)

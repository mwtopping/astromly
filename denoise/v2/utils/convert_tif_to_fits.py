from astropy.io import fits
from PIL import Image
import sys
import numpy as np



if len(sys.argv) <= 1:
    print("Please input filename")
    sys.exit(1)

fname = sys.argv[1]
im = Image.open(fname)

hdu = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(im)])

hdu.writeto(fname.replace("TIF", 'fits'), overwrite=True)

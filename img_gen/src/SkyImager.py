import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import numpy as np
import cv2
from tqdm import tqdm

from parameters import get_stellar_density, sample_star_mags
from perlin import perlin_octaves



class SkyImager:
    def __init__(self, sensor_cfg = {}, optics_cfg = {}, exposure_cfg = {}):
        self.sensor_cfg = sensor_cfg
        self.optics_cfg = optics_cfg
        self.exposure_cfg = exposure_cfg

        self.fill_defaults()
        self.calculate_other_params()

    def fill_defaults(self):
        sensor_cfg_default = {
            "nx":976,
            "ny":1304,
            "pix_size":3.75, #in micron
            "bias": 30,
            "dark": 0,
            "read": 0.8,
        }

        optics_cfg_default = {
            "focal_length":12, # in mm
            "aperture": 8, # in mm
        }

        exposure_cfg_default = {
            "gain": 100,
            "exp_time": 1, # in sec
        }

        for key in sensor_cfg_default:
            if key not in self.sensor_cfg:
                self.sensor_cfg[key] = sensor_cfg_default[key]
        for key in optics_cfg_default:
            if key not in self.optics_cfg:
                self.optics_cfg[key] = optics_cfg_default[key]
        for key in exposure_cfg_default:
            if key not in self.exposure_cfg:
                self.exposure_cfg[key] = exposure_cfg_default[key]

    def calculate_other_params(self):
        self.pixel_scale = self.sensor_cfg["pix_size"] / self.optics_cfg["focal_length"] / 1000 * 206265
        self.fov_width = self.sensor_cfg["nx"] * self.pixel_scale
        self.fov_height = self.sensor_cfg["ny"] * self.pixel_scale
        self.fov_area = self.fov_width * self.fov_height / (206265**2)
        print(self.fov_area)
    


    def __repr__(self):

        result = "".join([f"{key}:  {cfg[key]}\n" 
                    for cfg in [self.sensor_cfg, self.optics_cfg, self.exposure_cfg] 
                        for key in cfg])
        return result.strip()


    def generate_image(self):
        image = np.zeros((self.sensor_cfg["nx"], self.sensor_cfg["ny"]))

        image += self.sensor_cfg["bias"]

        mag_bins, total_star_mags = get_stellar_density()
        mags = sample_star_mags(mag_bins, total_star_mags, self.fov_area, mag_limit=12)

        xoffset = 2**15*np.random.rand()
        yoffset = 2**15*np.random.rand()

        noise_scale = 2.0

        for m in tqdm(mags):

            test = np.random.rand()
            x = np.random.randint(self.sensor_cfg["nx"])
            y = np.random.randint(self.sensor_cfg["ny"])

            while perlin_octaves(x/self.sensor_cfg["nx"]*noise_scale+xoffset, 
                                 y/self.sensor_cfg["ny"]*noise_scale+xoffset, 2) > test:
                test = np.random.rand()
                x = np.random.randint(self.sensor_cfg["nx"])
                y = np.random.randint(self.sensor_cfg["ny"])




            image = add_star(image, (x, y), m, flux_scale=200)

    #for ii in range(400):
    #    for jj in range(400):
    #        arr[ii][jj] = perlin_octaves(ii/400.0+xoffset, jj/400.0+xoffset, 8)


        # add noise
        image += (np.sqrt(image)*np.random.random(size=np.shape(image)))


        return image


def psf_convolve(image):

    image_conv = cv2.GaussianBlur(image, (65, 65), 0.3, 0.3) # instrument psf
    image_conv = cv2.GaussianBlur(image, (65, 65), 1.0, 1.0) # seeing

    return image_conv


def add_star(image, loc, m, flux_scale=1):
    cutout_size = 65
    #flux = flux_scale*50*np.random.pareto(1)
    flux = flux_scale * 3631 * 10**(-0.4*m)
    star_cutout = np.zeros((cutout_size, cutout_size))

    star_cutout[cutout_size //2, cutout_size //2] = flux

    star_cutout = psf_convolve(star_cutout)



    star_min_xedge = loc[0] - cutout_size // 2
    star_max_xedge = loc[0] + cutout_size // 2+1
    star_min_yedge = loc[1] - cutout_size // 2
    star_max_yedge = loc[1] + cutout_size // 2+1

    cutout_min_xedge = 0
    cutout_max_xedge = cutout_size 
    cutout_min_yedge = 0
    cutout_max_yedge = cutout_size 

    img_shape = np.shape(image)

    if star_min_xedge < 0:
        cutout_min_xedge -= star_min_xedge
        star_min_xedge = 0

    if star_max_xedge > img_shape[0]:
        cutout_max_xedge -= (star_max_xedge - img_shape[0])
        star_max_xedge = img_shape[0]

    if star_min_yedge < 0:
        cutout_min_yedge -= star_min_yedge
        star_min_yedge = 0

    if star_max_yedge > img_shape[1]:
        cutout_max_yedge -= (star_max_yedge - img_shape[1])
        star_max_yedge = img_shape[1]

    image[star_min_xedge:star_max_xedge, star_min_yedge:star_max_yedge] += (
        star_cutout[cutout_min_xedge:cutout_max_xedge, cutout_min_yedge:cutout_max_yedge])
    return image


if __name__ == "__main__":
    ski = SkyImager()

    img = ski.generate_image()
    fig, ax = plt.subplots()
    scaler = ZScaleInterval()
    limits = scaler.get_limits(img)
    ax.imshow(img, origin='lower', vmin=limits[0], vmax=1.2*limits[1], cmap="Greys_r")
    plt.show()



